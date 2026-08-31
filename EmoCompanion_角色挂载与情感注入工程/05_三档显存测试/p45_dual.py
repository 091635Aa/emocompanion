# -*- coding: utf-8 -*-
"""优化四：P4/P5 双通道叠加 —— 手动生成循环 + P3偏置 + P4 KV共振 + P5 内心态
- P3 单通道:  logits += β·tanh(emb_norm @ v_角色 / T)               (偏置)
- P3+P4:     上述偏置 + 对生成流情感token(g(p)=clip(cov,0,1)>0)位置 in-place K 缩放 (1+κ·g)
- P3+P4+P5:  P4 + 偏置改用 v_eff = norm(0.7·v_角色 + 0.3·v_用户)   (角色感知锚定)
- v_角色 = 优化二对比提示提取 (p3_contrast.json); 手动循环管理 past_key_values
用法: p45_dual.py [n] [cand]
"""
import os, sys, json, gc, random, re
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import run_sweep as rs

HERE = os.path.dirname(__file__)
TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CIDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CAND = [
    ("Qwen3-4B",        r"d:\AI情感\模型空间\Qwen3-4B",                            0.6),
    ("DeepSeek-Qwen7B", r"d:\AI情感\模型空间\DeepSeek-R1-Distill-Qwen-7B",         1.0),
]
JUDGE_PATH = r"d:\AI情感\微调文本\models\Qwen3-8B"
MICRO = r"d:\AI情感\EmoCompanion_角色挂载与情感注入工程\02_角色参数与数据\微调数据\微调训练集.jsonl"
MAX_NEW, TEMP, TOP_P, TOP_K = 80, 0.9, 0.9, 50
SEED = 21
BETA = 0.8 * 0.6 * 0.75     # 4B 推荐 β1.0 × fam0.6 × 4bit0.75
T_ANCHOR = 1.6
KAPPA = 0.3                 # P4 KV 缩放强度
GAMMA = 0.3                 # P5 用户分量占比

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

def load_gen(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                                 quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                                                 device_map="cuda")
    model.eval(); return model, tok

def sample(scores):
    scores = scores / TEMP
    topk = torch.topk(scores, TOP_K, -1).values
    scores = scores.masked_fill(scores < topk[0][-1], -float("inf"))
    sp = torch.sort(scores, descending=True).values
    cum = torch.softmax(sp, -1).cumsum(-1)
    keep = cum <= TOP_P
    if keep.any():
        scores = scores.masked_fill(scores < sp[keep].min(), -float("inf"))
    probs = torch.softmax(scores, -1)
    return torch.multinomial(probs, 1)

def scale_k(past, pos, g, kappa):
    """对 cache 中 pos 位置的所有 K 做 in-place 缩放 (1+kappa*g)  [transformers>=5: layers[l].keys]"""
    for l in range(len(past.layers)):
        k = past.layers[l].keys
        k[:, :, pos, :] = k[:, :, pos, :] * (1 + kappa * g)

def manual_gen(model, tok, user, bias, cov, mode):
    """mode: 'p3' / 'p3p4' / 'p3p4p5'; cov: V 维情感门(g=clip(cov,0,1)); bias 可为 None"""
    inp = chat_ids(tok, user)
    L = inp.shape[1]
    past, cur = None, inp
    gen_tokens, ents = [], []
    prev_tok, prev_pos = None, None
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    use_p4 = mode in ("p3p4", "p3p4p5") and cov is not None
    with torch.no_grad():
        for step in range(MAX_NEW):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            # P4: 缩放上一步采样 token 的位置(刚进入 cache) → 后续注意更多关注情感词
            if use_p4 and prev_pos is not None:
                g = float(torch.clamp(cov[prev_tok], 0.0, 1.0))
                if g > 0:
                    scale_k(past, prev_pos, g, KAPPA)
            logits = out.logits[:, -1, :]
            if bias is not None:
                logits = logits + bias.to(logits.device, logits.dtype)
            v = logits.topk(min(500, logits.shape[-1]), -1).values
            p = torch.softmax(v, -1).clamp_min(1e-12)
            e = -(p * p.log()).sum(-1).mean().item()
            ents.append(e if e == e else 0.0)   # nan 防护
            nxt = sample(logits)
            if nxt.item() == tok.eos_token_id:
                break
            gen_tokens.append(nxt.item())
            prev_tok = nxt.item()
            prev_pos = L + len(gen_tokens) - 1
            cur = nxt.reshape(1, 1)
    t1.record(); torch.cuda.synchronize()
    spd = len(gen_tokens) / max(t0.elapsed_time(t1) / 1000.0, 1e-6)
    text = strip_thinking(tok.decode(gen_tokens, skip_special_tokens=True))
    return text, (float(np.mean(ents)) if ents else 0.0), spd

def rep2(text):
    s = "".join(c for c in text if not c.isspace())
    if len(s) < 4: return 0.0
    bg = [s[i:i+2] for i in range(len(s)-1)]
    return 1.0 - len(set(bg)) / len(bg) if bg else 0.0

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
    name, path, fam = CAND[CIDX]
    print(f"[testset] {len(tests)} 条; 候选={name}; β={BETA:.3f} κ={KAPPA} γ={GAMMA}", flush=True)

    model, tok = load_gen(path)
    emb = rs.build_embedding_matrix(model).float()
    en = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-9)
    en_cpu = en.cpu()

    cdata = json.load(open(os.path.join(HERE, "p3_contrast.json"), encoding="utf-8"))
    v_role_c = torch.tensor(cdata["v_emb_contrast"], dtype=torch.float32)
    v_role = v_role_c.to("cuda")
    cov = en @ v_role                          # V 情感门
    bias_p3 = (BETA * fam) * torch.tanh(cov / T_ANCHOR)

    def v_user_cpu(user):
        ids = tok(user, add_special_tokens=False)["input_ids"]
        v = en_cpu[ids].mean(0)
        return v / v.norm().clamp_min(1e-9)

    def bias_p5(user):
        v_eff = 0.7 * v_role_c + 0.3 * v_user_cpu(user)
        v_eff = v_eff / v_eff.norm().clamp_min(1e-9)
        return (BETA * fam) * torch.tanh((en_cpu @ v_eff) / T_ANCHOR)

    b_p5 = {u: bias_p5(u) for u in tests}
    del emb; gc.collect(); torch.cuda.empty_cache()

    results = {k: [] for k in ("base", "p3", "p3p4", "p3p4p5")}
    for u in tests:
        results["base"].append(manual_gen(model, tok, u, None, None, "base"))
        results["p3"].append(manual_gen(model, tok, u, bias_p3, None, "p3"))
        results["p3p4"].append(manual_gen(model, tok, u, bias_p3, cov, "p3p4"))
        results["p3p4p5"].append(manual_gen(model, tok, u, b_p5[u], cov, "p3p4p5"))
    del model, tok; gc.collect(); torch.cuda.empty_cache()

    print("[judge] load Qwen3-8B", flush=True)
    jm, jt = load_judge()
    report = {}
    for k in ("base", "p3", "p3p4", "p3p4p5"):
        scores = [parse_dims(ask(jm, jt, JUDGE_FID.format(u=u, t=t[0]))) for u, t in zip(tests, results[k])]
        dims, avg = agg_dims(scores)
        reps = [rep2(t[0]) for t in results[k]]
        ents = [t[1] for t in results[k]]
        spds = [t[2] for t in results[k]]
        report[k] = {"avg": avg, "dims": dims, "rep2": round(float(np.mean(reps)), 3),
                     "entropy": round(float(np.mean(ents)), 3), "speed": round(float(np.mean(spds)), 1)}
        print(f"  [{k:>6}] 综合{avg}  温暖{dims['温暖']} 撒娇{dims['撒娇']} 口语{dims['口语化']} "
              f"情绪{dims['情绪起伏']} 去AI腔{dims['去AI腔']}  重复{report[k]['rep2']} "
              f"熵{report[k]['entropy']} {report[k]['speed']}t/s", flush=True)
    del jm, jt; gc.collect(); torch.cuda.empty_cache()

    json.dump({"candidate": name, "meta": {"beta": BETA, "kappa": KAPPA, "gamma": GAMMA,
               "n": len(tests), "seed": SEED}, "report": report,
               "samples": {k: [x[0] for x in results[k]][:5] for k in results}},
              open(os.path.join(HERE, "p45_dual.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n==== P4/P5 双通道叠加 ====")
    print("saved: p45_dual.json")

if __name__ == "__main__":
    main()
