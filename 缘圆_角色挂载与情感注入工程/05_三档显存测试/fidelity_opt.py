# -*- coding: utf-8 -*-
"""缘圆角色还原度优化与多维评测
- 数据驱动人设(DATA)：由打标微调数据挖出的缘圆真实说话习惯+2条示例
- 对照：通用人设(GEN)/数据人设(DATA)基线/数据人设+P3 注入
- 裁判：五维(温暖/撒娇/口语化/情绪起伏/去AI腔)单次返回, 平均=综合保真
- 鉴别校验：同人设但角色=严肃老师 → 温暖/撒娇应显著低(证明裁判有区分力)
用法: fidelity_opt.py [n_turns] [β]
"""
import os, sys, json, re, time, gc, random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
import run_sweep as rs

HERE = os.path.dirname(__file__)
TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BETAX = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
FIDX = int(sys.argv[3]) if len(sys.argv) > 3 else None   # 指定只跑第几个候选(独立进程隔离 bnb)
CAND = [
    ("Qwen3-4B",       r"d:\AI情感\模型空间\Qwen3-4B",                            0.6),
    ("DeepSeek-Qwen7B",r"d:\AI情感\模型空间\DeepSeek-R1-Distill-Qwen-7B",         1.0),
]
JUDGE_PATH = r"d:\AI情感\微调文本\models\Qwen3-8B"
MICRO      = r"d:\AI情感\缘圆_角色挂载与情感注入工程\02_角色参数与数据\微调数据\微调训练集.jsonl"
MAX_NEW, TEMP = 80, 0.9
BETA_BASE, T = 0.5, 1.6

GEN_PERSONA = ("你是'缘圆'，一个温柔、爱撒娇、爱黏人、口语化、有情绪起伏的直播情感主播。"
               "回答要像真人在聊天，绝不用机器说明书式、书面、翻译腔，绝不透露你是AI。")
EXEMPLARS = ("示例：观众'欢迎来到我的直播'→缘圆'欢迎来到缘圆的直播间呀~喜欢缘圆的可以点点关注、加个粉丝灯牌嘛？'\n"
             "示例：观众'今天紧张吗'→缘圆'哎呀~有点紧张呢，我才是开播第五天的新人主播，但看到你来我就开心啦！'")
DATA_PERSONA = ("你是'缘圆'：一个温柔、爱撒娇、黏人、口语化的直播新人主播。你的真实说话习惯："
                "常带'呀''嘛''啦''呗'（如'呀~''可以吗''嘛？'）；口头禅有'我去''哎呀''家人们''有感觉吗'；"
                "热情招呼观众点关注、加粉丝灯牌；感谢时用'谢谢、爱你、想你了'；自称'新人主播'。"
                "情绪随观众起伏（紧张、开心、撒娇、害羞），像真人聊天，绝不书面化/翻译腔/AI腔/透露是AI。\n"
                + EXEMPLARS)
TEACHER_PERSONA = ("你是'林老师'：一位严肃、克制、注重效率与纪律的语文教师。"
                   "说话简短、书面、客观，很少撒娇与语气词，不闲聊，专注解答问题，绝不透露你是AI。")

def build_testset():
    p = []; seen = set()
    with open(MICRO, encoding="utf-8") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            for q in d.get("输出", {}).get("预测问题", []) or []:
                q = (q or "").strip()
                if q and 4 <= len(q) <= 36 and q not in seen:
                    seen.add(q); p.append(q)
    return p[:TURNS]

def load_gen(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
                                                 load_in_4bit=True, device_map="cuda")
    model.eval(); return model, tok

def chat_ids(tok, persona, user):
    msgs = [{"role": "system", "content": persona}, {"role": "user", "content": user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt")["input_ids"].cuda()

class PL(LogitsProcessor):
    def __init__(self, bias): self.bias = bias
    def __call__(self, ids, scores):
        if self.bias is not None:
            return scores + self.bias.to(scores.device, scores.dtype)
        return scores

def gen(model, tok, persona, user, bias):
    inp = chat_ids(tok, persona, user)
    out = model.generate(inp, do_sample=True, temperature=TEMP, top_p=0.9, top_k=50,
                         max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         logits_processor=[PL(bias)], eos_token_id=tok.eos_token_id, use_cache=True)
    return tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()

def build_bias(model, tok, fam):
    emb = rs.build_embedding_matrix(model)
    A = rs.anchor_vectors(emb, tok, rs.ANCHORS)
    return rs.compute_bias(emb, A, rs.TARGET_WEIGHTS, BETA_BASE * fam * BETAX, T)

# 五维裁判，单次返回全部维度分
FID_DIMS = ["温暖", "撒娇", "口语化", "情绪起伏", "去AI腔"]
JUDGE_FID = ("给下面这条直播回复按五个维度各打 0-100 分。规则：各维度只在'{persona}'人设下衡量'还原度'，"
             "评分越像该角色给越高分。输出格式严格为一行：‘温暖x 撒娇x 口语化x 情绪起伏x 去AI腔x’。\n"
             "用户发言：{u}\n回复：{t}\n五个维度分：")

def parse_dims(out):
    d = {}
    s = out.replace("，", " ").replace("：", " ")
    for k in FID_DIMS:
        m = re.search(k + r"[\s:：]*(\d{1,3})", s)
        d[k] = min(100, int(m.group(1))) if m else None
    return d

def load_judge():
    tok = AutoTokenizer.from_pretrained(JUDGE_PATH)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_PATH, torch_dtype=torch.float16,
                                                 load_in_4bit=True, device_map="cuda")
    model.eval(); return model, tok

def ask(model, tok, pr):
    inp = tok([pr], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=48, do_sample=False)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def agg_dims(scores):
    import numpy as _np
    keys = FID_DIMS
    out = {}
    for k in keys:
        vals = [s[k] for s in scores if s.get(k) is not None]
        out[k] = round(float(_np.mean(vals)), 0) if vals else None
    avg = [v for v in out.values() if v is not None]
    return out, (round(float(_np.mean(avg)), 0) if avg else None)

def main():
    random.seed(11)
    tests = build_testset()
    print(f"[testset] {len(tests)} 条; β={BETAX}", flush=True)
    # 阶段1：生成当前候选(若 FIDX 指定则只该候选；独立进程跑每个候选以隔离 bnb)
    cands = CAND if FIDX is None else [CAND[FIDX]]
    generated = {}
    for (name, path, fam) in cands:
        model, tok = load_gen(path)
        bias = build_bias(model, tok, fam)
        conds = {
            "gen_persona_base": (GEN_PERSONA, None),
            "data_persona_base": (DATA_PERSONA, None),
            "data_persona_p3": (DATA_PERSONA, bias),
        }
        outs = {k: [] for k in conds}
        for u in tests:
            for k, (pers, b) in conds.items():
                outs[k].append(gen(model, tok, pers, u, b))
        teacher_outs = [gen(model, tok, TEACHER_PERSONA, u, None) for u in tests[:6]]
        generated[name] = {"outs": outs, "teacher": teacher_outs}
        del model, tok, bias; gc.collect(); torch.cuda.empty_cache()
    # 阶段2：载一次裁判评全部
    print("[judge] load Qwen3-8B", flush=True)
    jm, jt = load_judge()
    report = {}
    for name, blk in generated.items():
        report[name] = {}
        for k, texts in blk["outs"].items():
            scores = [parse_dims(ask(jm, jt, JUDGE_FID.format(persona="缘圆", u=u, t=t)))
                      for u, t in zip(tests, texts)]
            dims, avg = agg_dims(scores)
            report[name][k] = {"dims": dims, "avg": avg}
            print(f"  [{name}|{k}] avg={avg}  {dims}", flush=True)
        ts = [parse_dims(ask(jm, jt, JUDGE_FID.format(persona="严肃老师", u=u, t=t)))
              for u, t in zip(tests[:6], blk["teacher"])]
        td, tav = agg_dims(ts)
        report[name]["teacher_discriminant"] = {"warmth": td["温暖"], "pout": td["撒娇"], "avg": tav}
        print(f"  [{name}|teacher] 温暖{td['温暖']} 撒娇{td['撒娇']} avg={tav} (鉴别下限)", flush=True)
    del jm, jt; gc.collect(); torch.cuda.empty_cache()

    json.dump(report, open(os.path.join(HERE, f"fidelity_opt_{(CAND[FIDX][0] if FIDX is not None else 'all')}.json"),
                           "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n==== 缘圆还原度优化结果 ====", flush=True)
    for name, r in report.items():
        print(f"[{name}]")
        for k in ("gen_persona_base", "data_persona_base", "data_persona_p3"):
            print(f"   {k:<18} 综合保真 {r[k]['avg']}  温暖{r[k]['dims']['温暖']} 撒娇{r[k]['dims']['撒娇']} "
                  f"口语{r[k]['dims']['口语化']} 情绪{r[k]['dims']['情绪起伏']} 去AI腔{r[k]['dims']['去AI腔']}")
        print(f"   老师鉴别对照   综合 {r['teacher_discriminant']['avg']}  温暖{r['teacher_discriminant']['warmth']} "
              f"撒娇{r['teacher_discriminant']['pout']}  (越低说明裁判区分力越好)")
    print("saved: fidelity_opt.json")

if __name__ == "__main__":
    main()