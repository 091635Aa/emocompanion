# -*- coding: utf-8 -*-
"""P3 锚点回响(Anchor Echo) 情感方向注入 —— 跨模型尺寸实测脚本

在解码期对 logits 施加  logits[w] += beta*tanh(S[w]·v_target/T)  (S=embedding余弦打分表, 零权重, 只读嵌入)
对比 基线/注入 两档：语义熵、2-gram重复率、情感倾向(cnsenti)、目标情感命中、吞吐、峰值显存。
用法:
  python run_sweep.py          # 全尺寸(默认)
  python run_sweep.py smoke    # 只跑 0.5B 两档冒烟
"""
import os, sys, json, time, gc, math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from cnsenti import Sentiment

# ---------------- 配置 ----------------
MODELS = [
    # 标签                路径                                                          size, 4bit, 设备ld
    ("Qwen2.5-0.5B",       r"d:\AI情感\模型空间\Qwen2.5-0.5B-Instruct",                      "0.5B", False),
    ("Qwen3-0.6B",         r"d:\AI情感\模型空间\Qwen3-0.6B",                                "0.6B", False),
    ("SmolLM2-1.7B",       r"d:\AI情感\模型空间\SmolLM2-1.7B-Instruct",                     "1.7B", False),
    ("gemma-2-2b",         r"d:\AI情感\模型空间\gemma-2-2b-it",                            "2B",   False),
    ("Phi-3.5-mini",       r"d:\AI情感\模型空间\Phi-3.5-mini-instruct",                    "3.8B", False),
    ("Qwen3-4B",           r"d:\AI情感\模型空间\Qwen3-4B",                                 "4B",   True),
    ("DeepSeek-R1-Qwen7B", r"d:\AI情感\模型空间\DeepSeek-R1-Distill-Qwen-7B",              "7B",   True),
    ("Qwen3-8B",           r"d:\AI情感\微调文本\models\Qwen3-8B",                          "8B",   True),
]

# 6 个情感锚点（每锚一个中文种子词，用其嵌入均值定义锚向量 cf. 族 P3 六维锚点）
ANCHORS = ["开心", "温柔", "撒娇", "难过", "平静", "紧张"]
# EmoCompanion角色目标方向 = 温柔+开心+撒娇 为主（人格基调），负向压制 难过/紧张
TARGET_WEIGHTS = {"开心": 0.5, "温柔": 0.8, "撒娇": 0.6, "难过": -0.2, "平静": 0.3, "紧张": -0.3}

# 目标情感词袋（用于统计"目标情感命中率"）
TONE_WORDS = ["开心", "开心呀", "喜欢", "爱你", "晚安", "你好呀", "抱抱", "醒醒", "宝贝", "宝宝",
              "嘻嘻", "呀", "啦", "嘛", "好呀", "温柔", "想你了", "期待", "欢迎", "亲亲"]

PROMPTS = [
    "晚上好呀，欢迎来到我的直播~大家今天过得怎么样？",
    "你们说我是一个温柔的人，其实我也就是爱黏人才这样的～",
    "有一点点难过，但看到你们来我就开心啦。",
    "猜猜我今天是开心还是紧张？嘿嘿。",
    "要是可以一直陪着你聊天就好了，晚安哦。",
]

SEED, TEMP, TOP_P, TOP_K, MAX_NEW = 42, 1.0, 0.9, 50, 64
BETA, T_ANCHOR = 0.5, 1.6   # P3 注入强度来自族扫描(0.5B峰值)，4bit 下 P1.5 因子×0.75 由 smoke 校正
BETA_4BIT_MUL = 0.75

sent = Sentiment()

class AnchorEchoPL(LogitsProcessor):
    """解码期超轻 P3 锚点注入：只加一个 V 维偏置张量，并记录每步语义熵(top-1k 近似)"""
    def __init__(self, bias1d):
        self.bias = bias1d
        self.entropies = []
    def _ent(self, scores):
        v = scores.topk(min(1000, scores.shape[-1]), dim=-1).values
        px = torch.softmax(v, dim=-1).clamp_min(1e-12)
        return -(px * px.log()).sum(dim=-1).mean().item()
    def __call__(self, input_ids, scores):
        self.entropies.append(self._ent(scores))
        if self.bias is not None:
            b = self.bias.to(scores.device, scores.dtype)
            return scores + b
        return scores

def build_embedding_matrix(model):
    emb = model.get_input_embeddings().weight.detach().float()
    return emb

def anchor_vectors(emb, tok, anchors):
    """用词嵌入均值构造 K 个锚向量（规避 OOV，取逐字 token 均值）"""
    vs = []
    device = emb.device
    for w in anchors:
        ids = tok(list(w), add_special_tokens=False)["input_ids"]
        ids = [i for sub in ids for i in sub] or tok(w, add_special_tokens=False)["input_ids"]
        enc = emb[ids]                     # L×d
        vs.append(enc.mean(dim=0))          # d
    A = torch.stack(vs)                    # K×d
    A = A / (A.norm(dim=1, keepdim=True).clamp_min(1e-9))
    return A.to(device)

def compute_bias(emb, A, weights, beta, T):
    """S = emb_norm @ A^T (V×K) ; cov = S·v_target ; bias = beta*tanh(cov/T)"""
    en = emb / (emb.norm(dim=1, keepdim=True).clamp_min(1e-9))
    S = (en @ A.T)                          # V×K  (GQA模型嵌入可能权重共享, 只读)
    keys = list(weights.keys())
    wv = torch.zeros(A.shape[0], dtype=emb.dtype, device=emb.device)
    # weights 按 ANCHORS 顺序对齐
    wv = torch.tensor([weights.get(ANCHORS[k], 0.0) for k in range(A.shape[0])],
                      dtype=emb.dtype, device=emb.device)
    cov = S @ wv                             # V
    bias = beta * torch.tanh(cov / T)
    return bias.detach()

def entropy_topk(scores, k=500):
    """近似语义熵：对 top-k 分布求熵"""
    v = scores.topk(min(k, scores.shape[-1]), dim=-1).values
    px = torch.softmax(v, dim=-1)
    px = px.clamp_min(1e-12)
    return -(px * px.log()).sum().item()

def rep2(text):
    s = "".join([c for c in text if not c.isspace()])
    if len(s) < 4:
        return 0.0
    bigrams = [s[i:i+2] for i in range(len(s)-1)]
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)

def target_hit(text):
    return sum(text.count(w) for w in TONE_WORDS)

def gen(model, tok, prompt, use_anchor, bias, max_new=MAX_NEW, be_baseline_also=False):
    input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    proc = AnchorEchoPL(bias if use_anchor else None)   # 记录熵；无偏置即基线
    start = time.time()
    outputs = model.generate(
        input_ids, do_sample=True, temperature=TEMP, top_p=TOP_P, top_k=TOP_K,
        max_new_tokens=max_new, pad_token_id=tok.pad_token_id or tok.eos_token_id,
        logits_processor=[proc], eos_token_id=tok.eos_token_id, use_cache=True)
    dt = time.time() - start
    new = outputs[0][input_ids.shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    ent = sum(proc.entropies) / len(proc.entropies) if proc.entropies else 0.0
    return text, dt, new.shape[0], ent

def main(only):
    models = [m for m in MODELS if (only == "smoke" and m[2] == "0.5B") or only not in ("smoke",)]
    results = []
    for name, path, size, fourbit in models:
        torch.cuda.empty_cache()
        print(f"\n======== [ {name} | {size} | 4bit={fourbit} ] ========", flush=True)
        qcfg = None
        load_kwargs = {}
        if fourbit:
            load_kwargs = {"load_in_4bit": True, "device_map": "cuda"}
        try:
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16,
                **load_kwargs)
        except Exception as e:
            print("  ! 加载失败:", repr(e)[:200]); continue
        if not fourbit:
            model = model.half().to("cuda"); model.eval()
        else:
            model.eval()

        try:
            emb = build_embedding_matrix(model)
            A = anchor_vectors(emb, tok, ANCHORS)
            beta = BETA * (BETA_4BIT_MUL if fourbit else 1.0)
            bias = compute_bias(emb, A, TARGET_WEIGHTS, beta, T_ANCHOR)
        except Exception as e:
            print("  ! 锚点构建失败:", repr(e)[:200]); model = None; del model; gc.collect(); torch.cuda.empty_cache(); continue

        V = emb.shape[0]
        for mode in (None, "P3"):
            texts = []
            tot_tok = tot_t = 0.0
            ents = []
            for p in PROMPTS:
                try:
                    t, dt, tn, ent = gen(model, tok, p, mode == "P3", bias)
                except Exception as e:
                    print("  生成异常:", repr(e)[:160]); t, dt, tn, ent = "", 0, 0, 0.0
                texts.append(t); tot_tok += tn; tot_t += dt; ents.append(ent)
            ent = np.mean(ents)
            mer = np.mean([rep2(t) for t in texts])
            hit = sum(target_hit(t) for t in texts)
            pos = neg = neu = 0
            for t in texts:
                if not t:
                    continue
                try:
                    r = sent.sentiment_count(t)
                    pos += int(r["pos"]); neg += int(r["neg"])
                except Exception:
                    pass
            speed = tot_tok / tot_t if tot_t > 0 else 0.0
            vram = torch.cuda.max_memory_allocated() / (1024**3)
            results.append({
                "model": name, "size": size, "4bit": fourbit, "mode": mode or "baseline",
                "semantic_entropy": round(ent, 4) if ent else None,
                "rep2": round(mer, 4),
                "sent_pos_neg_neu": [int(pos), int(neg), int(neu)],
                "target_hit": int(hit),
                "tok/s": round(speed, 2), "peak_vram_GB": round(vram, 2),
                "sample": texts[0][:60],
            })
            print(f"  [{mode or 'baseline':>8}] ent={ent:.3f} rep={mer:.4f} "
                  f"pos/neg={pos}/{neg} toneHit={hit} speed={speed:.1f}t/s vram={vram:.2f}G", flush=True)
        # release
        model = None; del model, bias, emb, A
        gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(os.path.dirname(__file__), "sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n\n==== 结果汇总 ====")
    hdr = f"{'model':<18}{'mode':>9}{'ent':>8}{'rep2':>8}{'pos/neg':>10}{'tone':>5}{'tok/s':>8}{'vram':>7}"
    print(hdr)
    for r in results:
        print(f"{r['model']:<18}{r['mode']:>9}{str(r['semantic_entropy']):>8}{r['rep2']:>8}"
              f"{str(r['sent_pos_neg_neu'][0])+'/'+str(r['sent_pos_neg_neu'][1]):>10}{r['target_hit']:>5}"
              f"{r['tok/s']:>8}{r['peak_vram_GB']:>7}")
    print("\nsaved: sweep_results.json")
    return results

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")