# -*- coding: utf-8 -*-
"""calibration/optimizer —— 参考文本闭环校准的搜索器

在可控制的生成参数空间内，以「生成语音 vs 黄金样本」相似度 S 为目标做迭代搜索
(随机/网格扰动 + 精英保留)，逐情感 bucket 求出让 S 最高的参数配置，落盘为参数最优解。

不粗暴重训模型：通过调生成参数 + 选 adapter/参考音频，把生成拉向训练样本的
语速/音高/能量起伏/音色，直到相似度最高。即"生成→回测→修正"流程。
"""
import json
import os
import random
import re

from calibration.metrics import (composite, prosody_features, speaker_emb,
                                 edit_similarity)

_TAG = re.compile(r"\[/?emotion[^\]]*\]")   # 剥 [emotion]…[/emotion]

def strip_tag(text):
    return _TAG.sub("", text or "").strip()


# 可控生成参数空间(每个可采样项给离散/区间)
DEFAULT_SPACE = {
    "adapter": ["voice", "emotion"],
    "temperature": (0.70, 1.10),      # (lo,hi)
    "top_k": [40, 60, 80],
    "top_p": (0.85, 1.00),
    "repetition_penalty": (1.00, 1.10),
}

# 每个情感 bucket 固定给 best k 条做评估
DEFAULT_DEV_K = 3
DEFAULT_ROUNDS = 8   # 每个 bucket 评估的参数组数

# ---------------------------------------------------------------
# 统一强化（整句，不分情感独立训练）
#   一句话 = 复合情感，不能拆成单情感独立训练。
#   做法：把"推理权重"做进化式搜索(精英+变异+随机)，用整池分数做奖励，
#   分数回灌 → 下一轮权重 → 反复，直到整池相似度最高。
#   情感/语速注入不再靠往文本塞标签（tokenizer 会读出来），而是：
#     - emotion_ref: 选哪一个情感参考音频承载句中情感
#     - rate:        time-stretch 后处理注入语速(>1 更快)
#     - adapter/采样参数: 承载整体律动/语气
# ---------------------------------------------------------------
REFS = ["开心", "平静", "悲伤", "激动", "撒娇", "俏皮", "温柔"]

UNIFIED_SPACE = {
    "adapter": ["emotion", "voice"],
    "emotion_ref": REFS,                  # 情感参考音频（句中情感承载面）
    "temperature": (0.70, 1.10),
    "top_k": [40, 60, 80],
    "top_p": (0.85, 1.00),
    "repetition_penalty": (1.00, 1.10),
    "rate": (0.85, 1.15),                 # 语速注入 (time-stretch)
    "tone_variation": (0.0, 0.8),         # 随机语气强度(避免固定语气)
}


def sample_config(rng, space=None):
    s = space or DEFAULT_SPACE
    cfg = {}
    for k, v in s.items():
        if isinstance(v, tuple):
            cfg[k] = round(rng.uniform(v[0], v[1]), 3)
        else:
            cfg[k] = rng.choice(v)
    return cfg


def basic_space(space=None):
    """网格扰动基线：每个维度取中点/关键档。"""
    s = space or DEFAULT_SPACE
    out = []
    for adapter in s["adapter"]:
        for top_k in s["top_k"]:
            out.append({
                "adapter": adapter, "top_k": top_k,
                "temperature": 0.9, "top_p": 0.92, "repetition_penalty": 1.05,
            })
    return out


def _generate(engine, text, emotion, cfg):
    """用引擎生成候选语音，返回 (wav, sr, meta)；engine=None 用于 dry。"""
    if engine is None:
        return None
    return engine.synthesize(
        strip_tag(text), emotion, adapter=cfg.get("adapter", "emotion"),
        tone_variation=0.0,   # 校准要稳定可比，关闭句内随机
        temperature=cfg.get("temperature"), top_k=cfg.get("top_k"),
        top_p=cfg.get("top_p"), repetition_penalty=cfg.get("repetition_penalty"),
    )


def optimize_bucket(engine, items, emotion, rounds, dev_k,
                    seed=0, space=None):
    """对一个情感 bucket 迭代搜索，返回 (best_cfg, best_score, summary)。
    summary: {cfg, scores:[{text, composite, parts}]}
    注意：不再需要 wav_root；items 的 _wav 为绝对路径。"""
    rng = random.Random(seed)
    pool = [it for it in items[:dev_k] if os.path.isfile(it.get("_wav", ""))]
    if not pool or engine is None:
        return None, None, {}

    # 预计算黄金样本特征（一次）
    for it in pool:
        it["_text_clean"] = strip_tag(it["text"])
        it["_pros"] = prosody_features(it["_wav"])
        it["_emb"] = speaker_emb(engine, it["_wav"])

    cands = basic_space(space) + [sample_config(rng, space) for _ in range(max(rounds, 4))]
    best_cfg, best_score, best_summary = None, -1.0, None
    for cfg in cands:
        summaries = []
        for it in pool:
            out = _generate(engine, it["_text_clean"], emotion, cfg)
            if out is None:
                continue
            wav, sr, _ = out
            import numpy as np
            arr = np.asarray(wav, dtype="float32")
            gen_pros = prosody_features((arr, sr))
            gen_emb = speaker_emb(engine, (arr, sr))
            report = composite(
                gen_pros=gen_pros, gt_pros=it["_pros"],
                gen_nchars=len(it["_text_clean"]),
                gen_emb=gen_emb, gt_emb=it["_emb"], asr_sim=None, report={})
            summaries.append({"text": it["_text_clean"], "composite": report["composite"],
                              "parts": report["parts"], "_wav": wav, "_sr": sr,
                              "_cfg": dict(cfg)})
        if summaries:
            m = sum(x["composite"] for x in summaries) / len(summaries)
            if m > best_score:
                best_score, best_cfg, best_summary = m, cfg, summaries
    return best_cfg, best_score, best_summary


# ==================== 统一强化（整句，不对情感独立训练） ====================

def mutate(rng, cfg, space):
    """对推理权重做局部扰动（连续维 ±25% 跨度，离散维重选）。"""
    out = dict(cfg)
    for k, v in space.items():
        if isinstance(v, tuple):
            lo, hi = v
            base = lo if isinstance(out.get(k), (int, float)) else (lo + hi) / 2
            nv = float(out.get(k, (lo + hi) / 2)) + rng.uniform(-0.25, 0.25) * (hi - lo)
            out[k] = round(min(hi, max(lo, nv)), 3)
        else:
            out[k] = rng.choice(v)
    return out


def anchors(space=None):
    """确定性种子配置：覆盖两种 adapter、三类代表情感参考、三档 top_k，其余取中点。"""
    s = space or UNIFIED_SPACE
    refs = (s["emotion_ref"] or REFS)[:3]
    out = []
    for adapter in s["adapter"]:
        for ref in refs:
            for top_k in (s["top_k"] or [40, 60, 80])[:2]:
                out.append({
                    "adapter": adapter, "emotion_ref": ref, "top_k": top_k,
                    "temperature": 0.9, "top_p": 0.92,
                    "repetition_penalty": 1.05, "rate": 1.0, "tone_variation": 0.0,
                })
    return out


def apply_rate(wav, sr, rate):
    """语速注入：rate>1 加速(时长变短)。改动后返回的数据可用于直接打分。"""
    src_sr = int(sr)
    if rate is None or abs(float(rate) - 1.0) < 1e-4:
        return wav, src_sr, True
    try:
        import librosa
        import numpy as np
        y = np.asarray(wav, dtype="float32")
        warped = librosa.effects.time_stretch(y, rate=float(rate))
        return warped.astype("float32"), src_sr, True
    except Exception:
        return wav, src_sr, False


def _gen_unified(engine, text, cfg):
    """按统一权重 cfg 合成候选语音（情感+语速注入），返回 (wav, sr)。"""
    if engine is None:
        return None
    out = engine.synthesize(
        strip_tag(text),
        emotion=cfg.get("emotion_ref", "平静"),
        adapter=cfg.get("adapter", "emotion"),
        tone_variation=cfg.get("tone_variation", 0.0),
        temperature=cfg.get("temperature"), top_k=cfg.get("top_k"),
        top_p=cfg.get("top_p"), repetition_penalty=cfg.get("repetition_penalty"),
    )
    if out is None:
        return None
    wav, sr, _ = out
    import numpy as np
    wav = np.asarray(wav, dtype="float32")
    return apply_rate(wav, sr, cfg.get("rate", 1.0))


def _eval_pool(engine, pool, cfg, capture=False):
    """在整池上测一个推理权重 cfg，返回 (mean_reward, n, details)。
    details: [{text, emotion, composite, parts, _cfg, (_wav,_sr 当 capture)}]"""
    import numpy as np
    tot, n = 0.0, 0
    detail = []
    for it in pool:
        r = _gen_unified(engine, it["_text_clean"], cfg)
        if r is None:
            continue
        arr, sr, _ = r
        gen_pros = prosody_features((arr, sr))
        gen_emb = speaker_emb(engine, (arr, sr))
        rep = composite(
            gen_pros=gen_pros, gt_pros=it["_pros"],
            gen_nchars=len(it["_text_clean"]),
            gen_emb=gen_emb, gt_emb=it["_emb"], asr_sim=None, report={})
        tot += rep["composite"]; n += 1
        d = {"text": it["_text_clean"], "emotion": it.get("emotion", ""),
             "composite": rep["composite"], "parts": rep["parts"], "_cfg": dict(cfg)}
        if capture:
            d["_wav"] = arr; d["_sr"] = sr
        detail.append(d)
    return (tot / n if n else None), n, detail


def optimize_pool(engine, pool, rounds=12, pop_size=None, elite=3,
                  seed=0, space=None, log=False):
    """统一强化循环（核心）：
        1. 初代 = 锚点 + 随机
        2. 每代在整池评分 → 奖励 = 跨情感整池平均相似度
        3. 精英保留 + 变异 + 随机补位 → 下一代
        4. 反复，直到 rounds 满；返回 (best_cfg, best_reward, best_detail, history)
    best_detail 里带最佳权重在这些样本上的重生成音频(_wav/_sr=24000)，可直接落盘回档。
    """
    s = space or UNIFIED_SPACE
    rng = random.Random(seed)
    pop_size = pop_size or min(6 + max(rounds, 4), 16)
    pool = [it for it in pool if os.path.isfile(it.get("_wav", ""))]
    if not pool or engine is None:
        return None, None, None, []

    pop = anchors(s) + [sample_config(rng, s) for _ in range(pop_size)]
    best_cfg = best_reward = best_detail = None
    history = []
    for rnd in range(max(rounds, 1)):
        scored = []
        for cfg in pop:
            rew, ncn, _ = _eval_pool(engine, pool, cfg)
            if rew is None:
                continue
            scored.append((rew, cfg, ncn))
        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[0]
        if best_cfg is None or top[0] > best_reward:
            best_reward, best_cfg = top[0], top[1]
        history.append({"round": rnd, "best_reward": round(top[0], 4),
                        "elite": [(round(r, 4), c) for r, c, _ in scored[:elite]]})
        if log:
            print(f"  [round {rnd + 1}/{rounds}] reward={top[0]:.3f} cfg={top[1]}", flush=True)
        # 下一代 = 精英 + 变异精英 + 随机
        elites = [c for _, c, _ in scored[:elite]]
        muts = [mutate(rng, c, s) for c in elites for _ in range(2)]
        pop = elites + muts + [sample_config(rng, s) for _ in range(max(pop_size // 2, 1))]
    # 对最终最佳权重重评估一次，带出音频供落盘
    if best_cfg is not None:
        _, _, best_detail = _eval_pool(engine, pool, best_cfg, capture=True)
    return best_cfg, best_reward, best_detail, history