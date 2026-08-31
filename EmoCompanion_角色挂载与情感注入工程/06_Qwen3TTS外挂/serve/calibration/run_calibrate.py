# -*- coding: utf-8 -*-
"""整句统一强化闭环入口 —— 模型生成文本(含情感+语速)→TTS → 对比原始样本 → 评分 → 推理权重优化 → 反复

逻辑（一句话=复合情感，不做按情感分桶独立训练）:
  A. 建黄金池:  从打标 labels.jsonl 读 文本+情感+绝对wav，跨全部情感分层抽样 dev_k 条短样本
  A2. 模型外挂: 可选(--text-base + --aug-k)用 04 文本引擎把各情感最短 golden 句改写为
               同情感新句 → 加入整池(泛化验收)，其余仍 golden 复述(韵律校准)
  B. 统一生成:  每个"推理权重"(adapter + 情感参考 + 采样参数 + 语速time-stretch)整池生成，
     用 composite(语速/音高/能量/音色) 打分，整池均值即奖励
  C. 强化:      精英保留 + 变异 + 随机 → 下一代权重，反复 rounds 轮；落盘 best_weights + 音频 + 分数，写回档指针

用法(run 时保持 RVC 并行→建议小 rounds/dev-k):
  python -m calibration.run_calibrate --rounds 8 --dev-k 2 \
      --labels data/emotion_train_raw.jsonl --out-dir calibration/out --max-chars 60
  启用"模型外挂新句泛化"(需 04 文本引擎已启动，--text-base 指向其根地址):
  python -m calibration.run_calibrate --rounds 6 --dev-k 2 --aug-k 2 \
      --text-base http://127.0.0.1:<文本引擎端口> --labels data/emotion_train_raw.jsonl
  也可限定情感: --emotions 开心,悲伤 （此时仍统一训练，只是只采样这些情感）

回档: 每次运行写入 out/checkpoints/<时间戳>/best_weights.json + final_*.wav
      out/checkpoints/CURRENT 指向最近一次；要回退到历史，覆盖最佳权重配置即可。
"""
import argparse
import io
import json
import os
import random
import time

import numpy as np

CAL_DIR = os.path.dirname(os.path.abspath(__file__))
SERVE_DIR = os.path.dirname(CAL_DIR)

DEFAULT_LABELS = os.path.join(os.path.dirname(SERVE_DIR), "data", "emotion_train_raw.jsonl")


def probe(paths):
    """CPU 自检：打印每个 wav 的韵律描述。"""
    from calibration.metrics import prosody_features
    print("== probe(韵律管线自检) ==")
    for p in paths:
        print(f"  {os.path.basename(p)}: {prosody_features(p)}")


def load_buckets(labels, max_chars=80, k=6, emotions=None):
    """读打标 jsonl → {emotion:[{text,_wav,_chars}...]}。文本剥 [emotion] 标签、只保绝对 wav、
    限定文本长度与情感白名单，取 k 条最短。"""
    from calibration.optimizer import strip_tag
    buckets = {}
    with open(labels, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = strip_tag(rec.get("text", ""))
            wav = rec.get("audio", rec.get("wav", ""))
            emo = rec.get("emotion", "平静")
            if emotions and emo not in emotions:
                continue
            if not (text and wav and os.path.isfile(wav)):
                continue
            n = len(text)
            if not (6 <= n <= max_chars):
                continue
            buckets.setdefault(emo, []).append({"text": text, "_wav": wav, "_chars": n})
    # 每桶按文本长度升序取最短 k 条（短样本更利于快速/可比）
    return {e: sorted(v, key=lambda x: x["_chars"])[:k] for e, v in buckets.items()}


def _write_wav(path, arr, sr):
    import wave
    pcm = (np.clip(np.asarray(arr, dtype="float32"), -1, 1) * 32767).astype("int16")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs="*")
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--dev-k", type=int, default=2)
    ap.add_argument("--aug-k", type=int, default=0,
                    help="模型外挂：每情感用文本引擎改写的新句子数(0=关闭,仅golden复述)")
    ap.add_argument("--text-base", default=None,
                    help="04 文本引擎根地址,如 http://127.0.0.1:8001；提供且在线才启用新句泛化")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=80)
    ap.add_argument("--emotions", default=None, help="逗号分隔，缺省=全部")
    ap.add_argument("--out-dir", default=os.path.join(CAL_DIR, "out"))
    a = ap.parse_args()

    if a.probe:
        probe(a.probe)
        return

    random.seed(a.seed)
    emos = [e.strip() for e in a.emotions.split(",") if e.strip()] if a.emotions else None
    from calibration.optimizer import strip_tag, optimize_pool, prosody_features, speaker_emb
    from tts_engine import get_engine

    # 读全量打标数据（跨全部情感，一句话=复合情感，不做单情感独立训练）
    records = {}
    with open(a.labels, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = strip_tag(rec.get("text", ""))
            wav = rec.get("audio", rec.get("wav", ""))
            emo = rec.get("emotion", "平静")
            if emos and emo not in emos:
                continue
            if not (text and wav and os.path.isfile(wav)):
                continue
            if not (6 <= len(text) <= a.max_chars):
                continue
            records.setdefault(emo, []).append({"text": text, "emotion": emo, "_wav": wav})
    by_emo = {e: sorted(v, key=lambda x: len(x["text"])) for e, v in records.items()}
    if not by_emo:
        raise SystemExit(f"没有可用样本(检查 {a.labels} / max_chars / emotions){emos}")

    # 跨情感分层抽样：每情感取最短 dev_k 条，整池覆盖多种情感
    gold_pool = [it for emo in sorted(by_emo, key=lambda e: -len(by_emo[e]))
                 for it in by_emo[emo][:a.dev_k]]
    pool = list(gold_pool)
    print(f"== 统一强化：{len(pool)} 句 × {len(by_emo)} 种情感(不按情感独立训练) ==")
    print(f"   情感分布: { {e: len(v) for e, v in by_emo.items()} }")

    # 模型外挂：可选地用文本引擎改写新句加入整池(泛化验收)；不可用/未给则纯 golden 复述
    aug_pool, aug_online = [], False
    if a.aug_k and a.aug_k > 0:
        from calibration import text_gen as tg
        by_emo_order = sorted(by_emo.items(), key=lambda kv: -len(kv[1]))
        aug_pool, aug_online = tg.make_aug_set(a.text_base, by_emo_order,
                                               a.max_chars, a.aug_k, seed=a.seed)
        if aug_online:
            pool = aug_pool + gold_pool
            print(f"== 模型外挂·新句泛化: 新增 {len(aug_pool)} 句 "
                  f"(情感 {sorted({it['emotion'] for it in aug_pool})}) → 整池 {len(pool)} 句 "
                  "(golden复述校准 + 新句泛化统一训练)")
        else:
            print(f"!! --text-base 不可用({a.text_base})，跳过新句泛化，仅 golden 复述校准")
    else:
        print("== 未启用新句泛化(--text-base/--aug-k 缺省)，仅 golden 复述校准 ==")

    print("== 加载 TTS(常驻 4.4GB，与 RVC 并行) ==")
    eng = get_engine(); eng.load()

    # 预计算黄金样本特征（一次）
    for it in pool:
        it["_text_clean"] = strip_tag(it["text"])
        it["_pros"] = prosody_features(it["_wav"])
        it["_emb"] = speaker_emb(eng, it["_wav"])

    ts = time.strftime("%Y%m%d_%H%M%S")
    cp = os.path.join(a.out_dir, "checkpoints", ts)
    os.makedirs(cp, exist_ok=True)

    print("== 统一强化闭环（文本+情感+语速 → TTS → 对比原始样本 → 评分 → 权重进化 → 反复） ==")
    cfg, reward, detail, history = optimize_pool(eng, pool, rounds=a.rounds,
                                                 seed=a.seed, log=True)

    report = {"time": ts, "seed": a.seed, "rounds": a.rounds,
              "pool_size": len(pool), "aug_online": aug_online,
              "aug_n": len(aug_pool),
              "emotions": sorted(by_emo),
              "emotion_counts": {e: len(v) for e, v in by_emo.items()},
              "best_cfg": cfg, "best_reward": reward,
              "history": history, "samples": []}

    if cfg is not None:
        aug_set = {it["text"] for it in aug_pool}
        # 保存最终权重在这些样本上的语音（可回档/试听）
        for i, s in enumerate(detail):
            kind = "aug" if s["text"] in aug_set else "gold"
            if "_wav" in s and "_sr" in s:
                _write_wav(os.path.join(cp, f"final_{i}_{s['emotion']}_{kind}.wav"),
                           s["_wav"], s["_sr"])
            report["samples"].append({"kind": kind, "text": s["text"],
                                      "emotion": s["emotion"],
                                      "composite": s["composite"], "parts": s["parts"]})
        with open(os.path.join(cp, "best_weights.json"), "w", encoding="utf-8") as f:
            json.dump({"best_cfg": cfg, "best_reward": reward,
                       "samples": report["samples"]}, f, ensure_ascii=False, indent=2)
        print(f"  最终权重(可回档/复用): {dict(cfg)}")
        print(f"  整池最佳相似度 reward: {reward:.4f}")
    else:
        print("  无可用样本参与评估，未产出权重。")

    # 回档指针 + 汇总报告
    current = os.path.join(a.out_dir, "checkpoints", "CURRENT.json")
    with open(current, "w", encoding="utf-8") as f:
        json.dump({"latest_checkpoint": ts, "path": cp, "time": ts,
                   "best_reward": reward}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(cp, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[ok] 存档(可回档): {cp}")
    print(f"[ok] CURRENT → {current}")
    print("判定: reward 越高整句越贴近原样本；历史 checkpoint 在 out/checkpoints/ 下可随时回滚。")


if __name__ == "__main__":
    main()