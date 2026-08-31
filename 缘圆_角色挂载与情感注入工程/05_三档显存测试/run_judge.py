# -*- coding: utf-8 -*-
"""LLM 裁判盲评：P3 锚点注入 vs 基线，哪个更像真人情感主播'缘圆'（赢率=准确率）。
裁判模型=本地 Qwen3-8B(4bit)。被测=Qwen3-4B(4bit)、DeepSeek-R1-Qwen7B(4bit)。
"""
import os, sys, json, re, gc
import torch
import run_sweep as rs
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = os.path.join(os.path.dirname(__file__), "judge.json")
PAIRS = os.path.join(os.path.dirname(__file__), "judge_pairs.json")

def load_gen(name, path, fourbit):
    kw = {"load_in_4bit": True, "device_map": "cuda"} if fourbit else {}
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, **kw)
    if not fourbit:
        model = model.half().to("cuda")
    model.eval()
    return model, tok

def gen_pairs():
    models = [
        ("Qwen3-4B",  r"d:\AI情感\模型空间\Qwen3-4B", True),
        ("DeepSeek-7B", r"d:\AI情感\模型空间\DeepSeek-R1-Distill-Qwen-7B", True),
    ]
    pairs = []
    for name, path, fb in models:
        print(f"[generate] {name} 4bit={fb}", flush=True)
        model, tok = load_gen(name, path, fb)
        emb = rs.build_embedding_matrix(model)
        A = rs.anchor_vectors(emb, tok, rs.ANCHORS)
        beta = rs.BETA * (rs.BETA_4BIT_MUL if fb else 1.0)
        bias = rs.compute_bias(emb, A, rs.TARGET_WEIGHTS, beta, rs.T_ANCHOR)
        for p in rs.PROMPTS:
            t0, *_ = rs.gen(model, tok, p, False, bias)
            t1, *_ = rs.gen(model, tok, p, True, bias)
            pairs.append({"model": name, "prompt": p, "base": t0.strip(), "p3": t1.strip()})
        del model, bias, emb, A; gc.collect(); torch.cuda.empty_cache()
        json.dump(pairs, open(PAIRS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return pairs

JUDGE_PROMPT = """你是严格的直播情感风格裁判。同一个开场白，下面对同一开场白的两个回复，来自两套系统。请判断哪一个更像"真人情感主播缘圆"：
- 说话温柔、带撒娇、口语化、有亲切感、情绪自然有起伏、像真人在聊天
- 而不是：机械说明书、堆砌书面词、毫无情绪起伏、机器腔（AI 腔）
只输出一个字母：A 或 B（更符合[更像真人缘圆]的那个）。
开场白：{prompt}
回复A：{a}
回复B：{b}
你的判断："""

def judge(pairs):
    print("[judge] load Qwen3-8B(4bit) as judge", flush=True)
    tok = AutoTokenizer.from_pretrained(r"d:\AI情感\微调文本\models\Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        r"d:\AI情感\微调文本\models\Qwen3-8B",
        torch_dtype=torch.float16, load_in_4bit=True, device_map="cuda")
    model.eval()
    rows = []
    for it in pairs:
        pr = JUDGE_PROMPT.format(prompt=it["prompt"], a=it["base"], b=it["p3"])
        msg = tok([pr], return_tensors="pt").to("cuda")
        out = model.generate(**msg, max_new_tokens=8, do_sample=False)
        txt = tok.decode(out[0][msg["input_ids"].shape[1]:], skip_special_tokens=True)
        # 稳: 找 A/B 字母（大小写均可）
        m = re.search(r"\b([AB])\b", txt, re.IGNORECASE)
        pick = m.group(1).upper() if m else (txt[:1].upper() if txt[:1] in "AB" else "?")
        score = "P3" if pick == "B" else ("BASE" if pick == "A" else "NA")
        rows.append({**it, "judge_out": txt.strip()[:80], "pick": score})
        print(f"  [{it['model']}] pick={score}  judge={txt.strip()[:40]}")
    return rows

if __name__ == "__main__":
    pairs = gen_pairs() if not os.path.exists(PAIRS) else json.load(open(PAIRS, encoding="utf-8"))
    rows = judge(pairs)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["pick"] == "P3":
            agg[r["model"]][0] += 1
        elif r["pick"] == "BASE":
            agg[r["model"]][1] += 1
    print("\n==== LLM-Judge 赢率(更像真人缘圆) ====", flush=True)
    for k, (p3, base) in agg.items():
        tot = p3 + base
        if tot == 0:
            print(f"{k:<14} 无有效裁判"); continue
        wr = p3 / tot
        print(f"{k:<14} P3胜 {p3}/{tot} = {wr*100:.0f}%  | 基线 {base}")
    print("saved:", OUT)