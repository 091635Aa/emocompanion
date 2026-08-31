# -*- coding: utf-8 -*-
"""优化三：P1.5 调度器标定 —— β-效果曲线 + 调度器触发点检测
- Qwen3-4B(4bit) 主扫描: β ∈ {0.4,0.6,0.8,1.0,1.2,1.4} (相对 BETA_BASE*fam*4bit 的乘数)
- 每档: 保真(单维裁判0-100) / 2-gram重复率 / 语义熵 / 速度; 并统计"调度器应介入"的响应数
- 调度器规则(工程实践化方案 §2.2): 熵<0.6→×0.5; 重复>0.15→降β; 连续3次异常→归零
用法: p15_sweep.py [n] [cand_index]
"""
import os, sys, json, re, time, gc, random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, BitsAndBytesConfig
import run_sweep as rs

HERE = os.path.dirname(__file__)
TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CIDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CAND = [
    ("Qwen3-4B",        r"d:\AI情感\模型空间\Qwen3-4B",                            0.6),
    ("DeepSeek-Qwen7B", r"d:\AI情感\模型空间\DeepSeek-R1-Distill-Qwen-7B",         1.0),
]
JUDGE_PATH = r"d:\AI情感\微调文本\models\Qwen3-8B"
MICRO = r"d:\AI情感\缘圆_角色挂载与情感注入工程\02_角色参数与数据\微调数据\微调训练集.jsonl"
BETAS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
MAX_NEW, TEMP = 80, 0.9
SEED = 21
BETA_BASE, T_ANCHOR = 0.5, 1.6
BETA_4BIT = 0.75
# 调度器阈值(方案 §2.2)
ENT_TH, REP_TH = 0.60, 0.15

PERSONA = ("你是'缘圆'，一个温柔、爱撒娇、爱黏人、口语化的直播情感主播。你的真实说话习惯："
           "常带'呀''嘛''啦''呗'（如'呀~''可以吗''嘛？'）；口头禅有'我去''哎呀''家人们''有感觉吗'；"
           "热情招呼观众点关注、加粉丝灯牌；感谢时用'谢谢、爱你、想你了'；自称'新人主播'。"
           "情绪随观众起伏（紧张、开心、撒娇、害羞），像真人聊天，绝不书面化/翻译腔/AI腔/透露是AI。\n"
           "示例：观众'欢迎来到我的直播'→缘圆'欢迎来到缘圆的直播间呀~喜欢缘圆的可以点点关注、加个粉丝灯牌嘛？'\n"
           "示例：观众'今天紧张吗'→缘圆'哎呀~有点紧张呢，我才是开播第五天的新人主播，但看到你来我就开心啦！'")

def strip_thinking(text):
    if "\n response\n\n" in text:
        text = text.split("\n response\n\n", 1)[-1]
    else:
        t = text.lstrip()
        if t.startswith(" thinking") or t.startswith("thinking"):
            text = ""
    return text.strip()

def chat_ids(tok, user):
    msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt")["input_ids"].cuda()

def load_gen(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                                 quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                                                 device_map="cuda")
    model.eval(); return model, tok

class PL(LogitsProcessor):
    def __init__(self, bias):
        self.bias = bias
        self.ents = []
    def __call__(self, ids, scores):
        v = scores.topk(min(500, scores.shape[-1]), -1).values
        p = torch.softmax(v, -1).clamp_min(1e-12)
        self.ents.append(-(p * p.log()).sum(-1).mean().item())
        if self.bias is not None:
            return scores + self.bias.to(scores.device, scores.dtype)
        return scores

def gen(model, tok, user, bias):
    inp = chat_ids(tok, user)
    pl = PL(bias)
    t0 = time.time()
    out = model.generate(inp, do_sample=True, temperature=TEMP, top_p=0.9, top_k=50,
                         max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         logits_processor=[pl], eos_token_id=tok.eos_token_id, use_cache=True)
    dt = time.time() - t0
    ntok = out.shape[-1] - inp.shape[-1]
    text = strip_thinking(tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True))
    ent = float(np.mean(pl.ents)) if pl.ents else 0.0
    return text, ent, ntok / dt if dt else 0.0

def build_testset():
    p, seen = [], set()
    with open(MICRO, encoding="utf-8") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            for q in d.get("输出", {}).get("预测问题", []) or []:
                q = (q or "").strip()
                if q and 4 <= len(q) <= 40 and q not in seen:
                    seen.add(q); p.append(q)
    return p[:TURNS]

def rep2(text):
    s = "".join(c for c in text if not c.isspace())
    if len(s) < 4: return 0.0
    bg = [s[i:i+2] for i in range(len(s)-1)]
    return 1.0 - len(set(bg)) / len(bg) if bg else 0.0

JUDGE_FID = ("给下面这条直播回复打'像真人缘圆主播(温柔/撒娇/口语化/有情绪起伏/像真人聊天而非AI腔)'的保真分，0-100，只输出数字。"
             "观众发言：{u}\n回复：{t}\n保真分：")

def load_judge():
    tok = AutoTokenizer.from_pretrained(JUDGE_PATH)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_PATH, dtype=torch.float16,
                                                 quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                                                 device_map="cuda")
    model.eval(); return model, tok

def ask(model, tok, pr):
    inp = tok([pr], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=14, do_sample=False)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def main():
    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
    tests = build_testset()
    name, path, fam = CAND[CIDX]
    print(f"[testset] {len(tests)} 条; 候选={name}; β档={BETAS} (基β={BETA_BASE}×fam{fam}×4bit{BETA_4BIT})", flush=True)

    model, tok = load_gen(path)
    emb = rs.build_embedding_matrix(model)
    A = rs.anchor_vectors(emb, tok, rs.ANCHORS)
    bias_ref = rs.compute_bias(emb, A, rs.TARGET_WEIGHTS, BETA_BASE * fam * BETA_4BIT, T_ANCHOR)

    cols = {b: [] for b in BETAS}
    for u in tests:
        for b in BETAS:
            bias = (bias_ref * b).detach() if b != 1.0 else bias_ref
            t, ent, spd = gen(model, tok, u, bias)
            cols[b].append({"text": t, "ent": ent, "rep": rep2(t), "spd": spd})
    del model, tok, emb, A, bias_ref; gc.collect(); torch.cuda.empty_cache()

    print("[judge] load Qwen3-8B", flush=True)
    jm, jt = load_judge()
    report = {}
    for b in BETAS:
        for i, it in enumerate(cols[b]):
            it["u"] = tests[i]
        fids = []
        for it in cols[b]:
            out = ask(jm, jt, JUDGE_FID.format(u=it["u"], t=it["text"]))
            mm = re.search(r"\d{1,3}", out)
            fids.append(min(100, int(mm.group())) if mm else None)
        ent = np.mean([x["ent"] for x in cols[b]])
        rep = np.mean([x["rep"] for x in cols[b]])
        spd = np.mean([x["spd"] for x in cols[b]])
        n_ent_low = sum(1 for x in cols[b] if x["ent"] < ENT_TH)
        n_rep_high = sum(1 for x in cols[b] if x["rep"] > REP_TH)
        report[b] = {"fidelity": round(float(np.nanmean([x for x in fids if x is not None])), 1)
                     if any(x is not None for x in fids) else None,
                     "entropy": round(float(ent), 3), "rep2": round(float(rep), 3),
                     "speed": round(float(spd), 1),
                     "sched_ent_low": n_ent_low, "sched_rep_high": n_rep_high}
        print(f"  β={b:<4} 保真{report[b]['fidelity']}  熵{report[b]['entropy']}  重复{report[b]['rep2']} "
              f"{spd:.1f}t/s  调度介入[熵<{ENT_TH}:{n_ent_low} 重复>{REP_TH}:{n_rep_high}]", flush=True)
    del jm, jt; gc.collect(); torch.cuda.empty_cache()

    json.dump({"candidate": name, "betas": BETAS, "n": len(tests), "seed": SEED,
               "report": report, "samples": {str(b): cols[b][0]["text"][:60] for b in BETAS}},
              open(os.path.join(HERE, "p15_sweep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n==== P1.5 调度器标定 ====")
    print("saved: p15_sweep.json")

if __name__ == "__main__":
    main()
