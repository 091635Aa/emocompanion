# -*- coding: utf-8 -*-
"""无偏置裁判：随机交换 A/B 顺序,消除"永远选第二个"的位置偏置,结果映射回 P3 真实赢率。"""
import os, json, re, random, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import run_judge as rj

PAIRS = os.path.join(os.path.dirname(__file__), "judge_pairs.json")
OUT   = os.path.join(os.path.dirname(__file__), "judge_unbiased.json")
random.seed(2026)

JUDGE_PROMPT = rj.JUDGE_PROMPT

def main():
    pairs = json.load(open(PAIRS, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(r"d:\AI情感\微调文本\models\Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        r"d:\AI情感\微调文本\models\Qwen3-8B",
        torch_dtype=torch.float16, load_in_4bit=True, device_map="cuda")
    model.eval()
    rows = []
    for it in pairs:
        # 随机决定 P3 放 A 还是 B
        p3_as_a = random.random() < 0.5
        a = it["p3"] if p3_as_a else it["base"]
        b = it["base"] if p3_as_a else it["p3"]
        pr = JUDGE_PROMPT.format(prompt=it["prompt"], a=a, b=b)
        msg = tok([pr], return_tensors="pt").to("cuda")
        out = model.generate(**msg, max_new_tokens=8, do_sample=False)
        txt = tok.decode(out[0][msg["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\b([AB])\b", txt, re.IGNORECASE)
        pick = m.group(1).upper() if m else (txt[:1].upper() if txt[:1] in "AB" else "?")
        judge_picked_p3 = (pick == "A" and p3_as_a) or (pick == "B" and not p3_as_a)
        rows.append({**it, "p3_as_a": p3_as_a, "judge_out": txt.strip()[:80],
                     "pick": "P3" if judge_picked_p3 else ("BASE" if pick in "AB" else "NA")})
        print(f"[{it['model']}] p3_as_a={p3_as_a} judgePick={pick} -> {rows[-1]['pick']}", flush=True)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])  # [p3, base, na]
    for r in rows:
        agg[r["model"]][{"P3":0,"BASE":1,"NA":2}[r["pick"]]] += 1
    print("\n==== 无偏置裁判：P3 真实赢率 ====", flush=True)
    tot_p3 = 0; tot_judge = 0
    for k, (p3, base, na) in agg.items():
        j = p3 + base
        print(f"{k:<14} P3胜 {p3}/{j} = {p3/j*100:.0f}%  （基线 {base}, 无效 {na}）")
        tot_p3 += p3; tot_judge += j
    if tot_judge:
        print(f"TOTAL        P3胜 {tot_p3}/{tot_judge} = {tot_p3/tot_judge*100:.0f}%")
    print("saved:", OUT)

if __name__ == "__main__":
    main()