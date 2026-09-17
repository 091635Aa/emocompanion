# -*- coding: utf-8 -*-
"""优化二：P3锚点精度升级 评估 —— A现有锚点 vs B对比提示v_角色 × 50条 × 五维裁判
- A: run_sweep 6锚点加权 v_target (现状基线)
- B: p3_contrast.json 的 v_emb_contrast (嵌入空间对比向量, 正负persona生成token嵌入差)
- 含 base(无P3) 参照; β=0.8 (4B 推荐), 4bit×0.75
用法: p3_contrast_eval.py [n] [beta] [cand_index]
"""
import os, sys, json, gc, random, re
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, BitsAndBytesConfig
import run_sweep as rs

HERE = os.path.dirname(__file__)
TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
BETA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
CIDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
CAND = [
    ("Qwen3-4B",        r"d:\AI情感\模型空间\Qwen3-4B",                            0.6),
    ("DeepSeek-Qwen7B", r"d:\AI情感\模型空间\DeepSeek-R1-Distill-Qwen-7B",         1.0),
]
JUDGE_PATH = r"d:\AI情感\微调文本\models\Qwen3-8B"
MICRO = r"d:\AI情感\EmoCompanion_角色挂载与情感注入工程\02_角色参数与数据\微调数据\微调训练集.jsonl"
MAX_NEW, TEMP = 80, 0.9
SEED = 21
BETA_4BIT = 0.75
T_ANCHOR = 1.6

PERSONA = ("你是'EmoCompanion'，一个温柔、爱撒娇、爱黏人、口语化的直播情感主播。你的真实说话习惯："
           "常带'呀''嘛''啦''呗'（如'呀~''可以吗''嘛？'）；口头禅有'我去''哎呀''家人们''有感觉吗'；"
           "热情招呼观众点关注、加粉丝灯牌；感谢时用'谢谢、爱你、想你了'；自称'新人主播'。"
           "情绪随观众起伏（紧张、开心、撒娇、害羞），像真人聊天，绝不书面化/翻译腔/AI腔/透露是AI。\n"
           "示例：观众'欢迎来到我的直播'→EmoCompanion'欢迎来到EmoCompanion的直播间呀~喜欢EmoCompanion的可以点点关注、加个粉丝灯牌嘛？'\n"
           "示例：观众'今天紧张吗'→EmoCompanion'哎呀~有点紧张呢，我才是开播第五天的新人主播，但看到你来我就开心啦！'")

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
    """简单 logits 加法处理器"""
    def __init__(self, bias):
        self.bias = bias
    def __call__(self, ids, scores):
        if self.bias is not None:
            return scores + self.bias.to(scores.device, scores.dtype)
        return scores

def gen(model, tok, user, bias):
    inp = chat_ids(tok, user)
    kw = dict(do_sample=True, temperature=TEMP, top_p=0.9, top_k=50,
              max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id or tok.eos_token_id,
              eos_token_id=tok.eos_token_id, use_cache=True)
    if bias is not None:
        kw["logits_processor"] = [PL(bias)]
    out = model.generate(inp, **kw)
    return strip_thinking(tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True))

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

# ---------------- 两种 v_角色 → P3 bias ----------------
def bias_anchor(model, tok, fam):
    emb = rs.build_embedding_matrix(model)
    A = rs.anchor_vectors(emb, tok, rs.ANCHORS)
    return rs.compute_bias(emb, A, rs.TARGET_WEIGHTS, BETA * fam * BETA_4BIT, T_ANCHOR)

def bias_contrastive(model, tok, fam):
    """v_emb_contrast(嵌入空间) 直接作 v_target: cov = emb_norm @ v ; bias = beta*tanh(cov/T)"""
    data = json.load(open(os.path.join(HERE, "p3_contrast.json"), encoding="utf-8"))
    v = torch.tensor(data["v_emb_contrast"], dtype=torch.float32)
    emb = rs.build_embedding_matrix(model)
    en = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cov = en @ v.to(emb.device)                    # V
    bias = (BETA * fam * BETA_4BIT) * torch.tanh(cov / T_ANCHOR)
    return bias.detach()

# ---------------- 五维裁判 ----------------
FID_DIMS = ["温暖", "撒娇", "口语化", "情绪起伏", "去AI腔"]
JUDGE_FID = ("给下面这条直播回复按五个维度各打 0-100 分。规则：各维度只在'EmoCompanion'人设下衡量'还原度'，"
             "评分越像该角色给越高分。输出格式严格为一行：‘温暖x 撒娇x 口语化x 情绪起伏x 去AI腔x’。\n"
             "用户发言：{u}\n回复：{t}\n五个维度分：")

def parse_dims(out):
    d = {}
    s = out.replace("，", " ").replace("：", " ")
    for k in FID_DIMS:
        m = list(re.finditer(k + r"[^\d]{0,12}(\d{1,3})", s))
        d[k] = min(100, int(m[-1].group(1))) if m else None
    return d

def load_judge():
    tok = AutoTokenizer.from_pretrained(JUDGE_PATH)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_PATH, dtype=torch.float16,
                                                 quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                                                 device_map="cuda")
    model.eval(); return model, tok

def ask(model, tok, pr):
    inp = tok([pr], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=48, do_sample=False)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def agg_dims(scores):
    out = {}
    for k in FID_DIMS:
        vals = [s[k] for s in scores if s.get(k) is not None]
        out[k] = round(float(np.mean(vals)), 0) if vals else None
    avg = [v for v in out.values() if v is not None]
    return out, (round(float(np.mean(avg)), 0) if avg else None)

def main():
    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
    tests = build_testset()
    print(f"[testset] {len(tests)} 条; β={BETA}×{BETA_4BIT}  候选={CAND[CIDX][0]}", flush=True)
    name, path, fam = CAND[CIDX]

    model, tok = load_gen(path)
    bA = bias_anchor(model, tok, fam)
    bB = bias_contrastive(model, tok, fam)
    # 归一化可比: 两种偏置量级可能不同
    print(f"[bias] A|mean|={bA.abs().mean():.4f} max={bA.abs().max():.4f} | B|mean|={bB.abs().mean():.4f} max={bB.abs().max():.4f}", flush=True)

    results = {k: [] for k in ("base", "A", "B")}
    for u in tests:
        results["base"].append(gen(model, tok, u, None))
        results["A"].append(gen(model, tok, u, bA))
        results["B"].append(gen(model, tok, u, bB))
    del model, tok; gc.collect(); torch.cuda.empty_cache()

    print("[judge] load Qwen3-8B", flush=True)
    jm, jt = load_judge()
    report = {}
    for k in ("base", "A", "B"):
        scores = [parse_dims(ask(jm, jt, JUDGE_FID.format(u=u, t=t))) for u, t in zip(tests, results[k])]
        dims, avg = agg_dims(scores)
        reps = [rep2(t) for t in results[k]]
        report[k] = {"avg": avg, "dims": dims, "rep2": round(float(np.mean(reps)), 3)}
        print(f"  [{k:>4}] 综合{avg}  温暖{dims['温暖']} 撒娇{dims['撒娇']} 口语{dims['口语化']} "
              f"情绪{dims['情绪起伏']} 去AI腔{dims['去AI腔']}  重复{report[k]['rep2']}", flush=True)
    del jm, jt; gc.collect(); torch.cuda.empty_cache()

    json.dump({"candidate": name, "meta": {"beta": BETA, "n": len(tests), "seed": SEED},
               "report": report, "samples": {k: results[k][:5] for k in results}},
              open(os.path.join(HERE, "p3_contrast_eval.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n==== P3锚点精度升级 评估 ====")
    print("saved: p3_contrast_eval.json")

if __name__ == "__main__":
    main()
