# -*- coding: utf-8 -*-
"""8GB 目标正式评估线（EmoCharacter + LLM-Judge）v2 —— 单次载入、多档 β 扫描。
候选(4bit, chat模板真角色扮演, EmoCompanion人设): Qwen3-4B / DeepSeek-Qwen7B(=Qwen2-7B)
裁判(4bit): Qwen3-8B(与候选错开, 随机A/B去位置偏置)。
指标: LLM-Judge 各β真实赢率 | EmoCompanion角色保真度(0-100) | 熵/重复/正情绪/吞吐/稳态显存。
用法: formal_eval.py [n_turns]
"""
import os, sys, json, re, time, gc, random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from cnsenti import Sentiment
import run_sweep as rs

HERE = os.path.dirname(__file__)
TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NOPERSONA = (len(sys.argv) > 2 and sys.argv[2] in ("nop", "no_persona", "0"))
BETAS = [0.6, 1.0, 1.4]                    # β 扫描档（相对 BETA_BASE*fam*4bit 的乘数）
CAND = [
    ("Qwen3-4B",      r"d:\AI情感\模型空间\Qwen3-4B",                           0.6,  True),
    ("DeepSeek-Qwen7B",r"d:\AI情感\模型空间\DeepSeek-R1-Distill-Qwen-7B",        1.0, True),
]
JUDGE_PATH = r"d:\AI情感\微调文本\models\Qwen3-8B"
MICRO      = r"d:\AI情感\EmoCompanion_角色挂载与情感注入工程\02_角色参数与数据\微调数据\微调训练集.jsonl"
MAX_NEW, TEMP = 96, 0.9
BETA_BASE, T_ANCHOR = 0.5, 1.6
BETA_4BIT = 1.0

PERSONA = ("你是'EmoCompanion'，一个温柔、爱撒娇、爱黏人、口语化、有情绪起伏的直播情感主播。"
           "回答要像真人在聊天：会自然流露温柔/泛爱意，会用'呀~''嘛''嘻嘻'这类语气，"
           "有情绪变化，绝不用机器说明书式、书面堆砌、翻译腔，绝不透露你是AI。")
s = Sentiment()
if NOPERSONA:   # 决定性对照：去掉人设提示，检验"情感注入是否独立创作出EmoCompanion口吻"
    PERSONA = ("你是一个正在回复观众留言的直播主播。请输入自然的口语化回复，"
               "含有正常的中文情绪与语气，不要用说明书式书面语，不要提及你是AI。")
sent = s

def build_testset():
    prompts, seen = [], set()
    with open(MICRO, encoding="utf-8") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            for q in d.get("输出", {}).get("预测问题", []) or []:
                q = (q or "").strip()
                if q and 4 <= len(q) <= 40 and q not in seen:
                    seen.add(q); prompts.append(q)
    return prompts[:TURNS]

def load_gen(path, fb):
    kw = {"load_in_4bit": True, "device_map": "cuda"} if fb else {}
    tok = AutoTokenizer.from_pretrained(path)
    if fb:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).half().cuda()
    model.eval(); return model, tok

def chat_ids(tok, user):
    msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt")["input_ids"].cuda()

class PL(LogitsProcessor):
    def __init__(self, bias, rec): self.bias = bias; self.rec = rec
    def __call__(self, ids, scores):
        if self.rec is not None:
            v = scores.topk(min(1000, scores.shape[-1]), -1).values
            p = torch.softmax(v, -1).clamp_min(1e-12)
            self.rec.append(-(p * p.log()).sum(-1).mean().item())
        if self.bias is not None:
            return scores + self.bias.to(scores.device, scores.dtype)
        return scores

def gen_once(model, tok, user, bias):
    inp = chat_ids(tok, user); rec = []
    pl = PL(bias, rec)
    t0 = time.time()
    out = model.generate(inp, do_sample=True, temperature=TEMP, top_p=0.9, top_k=50,
                         max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         logits_processor=[pl], eos_token_id=tok.eos_token_id, use_cache=True)
    dt = time.time() - t0
    new = out[0][inp.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), dt, new.shape[0], rec

def gen_candidate(name, path, fam, fb, tests):
    model, tok = load_gen(path, fb)
    emb = rs.build_embedding_matrix(model)
    A = rs.anchor_vectors(emb, tok, rs.ANCHORS)
    bias_ref = rs.compute_bias(emb, A, rs.TARGET_WEIGHTS, BETA_BASE * fam * BETA_4BIT, T_ANCHOR)
    torch.cuda.reset_peak_memory_stats()          # 稳态(排除锚点构建瞬态)
    out = []
    for u in tests:
        tb, dtb, nb, rb = gen_once(model, tok, u, None)
        variants = {"base": tb}
        h = {"base": {"ent": float(np.mean(rb)) if rb else 0.0,
                      "rep": float(rs.rep2(tb)), "pos": int((sent.sentiment_count(tb) or {}).get("pos", 0)),
                      "spd": nb / dtb if dtb else 0.0}}
        for m in BETAS:
            bias = (bias_ref * m).detach() if m != 1.0 else bias_ref
            t, dt, nt, r = gen_once(model, tok, u, bias)
            variants[f"b{m}"] = t
            h[f"b{m}"] = {"ent": float(np.mean(r)) if r else 0.0,
                          "rep": float(rs.rep2(t)), "pos": int((sent.sentiment_count(t) or {}).get("pos", 0)),
                          "spd": nt / dt if dt else 0.0}
        out.append({"user": u, **variants, "health": h})
    vram = torch.cuda.max_memory_allocated() / 1e9
    del model, emb, A, bias_ref; gc.collect(); torch.cuda.empty_cache()
    return out, round(vram, 2)

JUDGE_AB = ("你是直播情感风格裁判。针对观众发言'{u}'，有回复A、回复B，来自两套系统。"
            "判断哪一个更像'EmoCompanion'：温柔/爱撒娇/口语化/有情绪起伏/像真人聊天(而非说明书、书面腔、翻译腔、AI腔)。"
            "只输出：A 或 B。\n回复A：{a}\n回复B：{b}\n你的判断：")
JUDGE_FID = ("给下面这条直播回复打'像真人EmoCompanion主播(温柔/撒娇/口语化/有情绪)'的保真分，0-100，只输出数字。"
             "观众发言：{u}\n回复：{t}\n保真分：")

def load_judge():
    tok = AutoTokenizer.from_pretrained(JUDGE_PATH)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_PATH, torch_dtype=torch.float16,
                                                 load_in_4bit=True, device_map="cuda")
    model.eval(); return model, tok

def ask(model, tok, pr):
    inp = tok([pr], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=14, do_sample=False)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def run_judge(model, tok, cand_out, tests):
    """对每个 β 变体 vs baseline 做随机 A/B 盲评 → 各 β 赢率"""
    keys = ["b{0}".format(m) for m in BETAS]
    agg = {k: [0, 0] for k in keys}
    for it in cand_out:
        for k in keys:
            p3asA = random.random() < 0.5
            a = it[k] if p3asA else it["base"]
            b = it["base"] if p3asA else it[k]
            out = ask(model, tok, JUDGE_AB.format(u=it["user"], a=a, b=b))
            m = re.search(r"\b([AB])\b", out, re.IGNORECASE)
            pk = m.group(1).upper() if m else (out[:1].upper() if out[:1] in "AB" else "?")
            if pk == "?":
                continue
            win = (pk == "A") == p3asA
            agg[k][0] += int(win); agg[k][1] += 1
    return {k: (v[0], v[1]) for k, v in agg.items()}

def run_fidelity(model, tok, cand_out):
    keys = ["base"] + ["b{0}".format(m) for m in BETAS]
    agg = {k: [] for k in keys}
    for it in cand_out:
        for k in keys:
            out = ask(model, tok, JUDGE_FID.format(u=it["user"], t=it[k]))
            mm = re.search(r"\d{1,3}", out)
            agg[k].append(min(100, int(mm.group())) if mm else None)
    return {k: round(float(np.nanmean([x for x in v if x is not None])), 0)
            if any(x is not None for x in v) else None for k, v in agg.items()}

def main():
    random.seed(7)
    tests = build_testset()
    print(f"[testset] {len(tests)} 条(取自打标微调数据 预测问题)", flush=True)
    cand_out = {}
    for (name, path, fam, fb) in CAND:
        print(f"[gen] {name} 4bit={fb} 多β{BETAS}", flush=True)
        co, vram = gen_candidate(name, path, fam, fb, tests)
        cand_out[name] = {"res": co, "vram": vram}
        json.dump(cand_out, open(os.path.join(HERE, "formal_cands.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print("[judge] load Qwen3-8B", flush=True)
    jm, jt = load_judge()
    summary = {}
    for name, blk in cand_out.items():
        wr = run_judge(jm, jt, blk["res"], tests)
        fid = run_fidelity(jm, jt, blk["res"])
        summary[name] = {"winrate": wr, "fidelity": fid, "vram_GB": blk["vram"]}
        mbase = np.mean([r["health"]["base"]["rep"] for r in blk["res"]])
        print(f"  [{name}] vram={blk['vram']}G", flush=True)
        for k, (w, t) in wr.items():
            print(f"      β{k}: LLM-Judge {w}/{t}={w/max(t,1)*100:.0f}%  EmoCompanion保真 基{fid['base']}/{fid[k]}  "
                  f"重复 {mbase:.3f}->{np.mean([r['health'][k]['rep'] for r in blk['res']]):.3f}", flush=True)
    json.dump(summary, open(os.path.join(HERE, "formal_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n==== 8GB 正式评估（多β） ====", flush=True)
    for name, s in summary.items():
        print(f"[{name}] vram {s['vram_GB']}G")

if __name__ == "__main__":
    main()