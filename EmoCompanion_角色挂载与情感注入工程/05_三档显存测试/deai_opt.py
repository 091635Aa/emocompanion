# -*- coding: utf-8 -*-
"""优化一：去AI腔双向抑制 —— 三组对照实验
- 正例词袋(数据驱动, deai_bag.json): EmoCompanion真实口癖(吗/呀/嘛/啦/欢迎/我去...) → 解码期 boost
- 空泛词抑制表(数据驱动, 强1.0/弱0.5): AI腔模板(太好了/首先/总之/让我们一起...) → 解码期续接抑制
- 对照: base(基线) / A正例only / B抑制only / C双向
- 裁判: Qwen3-8B 五维(温暖/撒娇/口语化/情绪起伏/去AI腔), 平均=综合保真
用法: deai_opt.py [n_turns] [pos_scale] [hol_scale] [candidate_index]
"""
import os, sys, json, re, time, gc, random, math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, BitsAndBytesConfig
import run_sweep as rs

HERE = os.path.dirname(__file__)
TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
POS_SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # 正例boost强度
HOL_SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2   # 空泛词抑制强度
CIDX = int(sys.argv[4]) if len(sys.argv) > 4 else 0            # 候选索引(独立进程隔离 bnb)
CAND = [
    ("Qwen3-4B",        r"d:\AI情感\模型空间\Qwen3-4B",                            0.6),
    ("DeepSeek-Qwen7B", r"d:\AI情感\模型空间\DeepSeek-R1-Distill-Qwen-7B",         1.0),
]
JUDGE_PATH = r"d:\AI情感\微调文本\models\Qwen3-8B"
MICRO = r"d:\AI情感\EmoCompanion_角色挂载与情感注入工程\02_角色参数与数据\微调数据\微调训练集.jsonl"
BAG = os.path.join(HERE, "deai_bag.json")
MAX_NEW, TEMP = 80, 0.9
SEED = 21

PERSONA = ("你是'EmoCompanion'，一个温柔、爱撒娇、爱黏人、口语化的直播情感主播。你的真实说话习惯："
           "常带'呀''嘛''啦''呗'（如'呀~''可以吗''嘛？'）；口头禅有'我去''哎呀''家人们''有感觉吗'；"
           "热情招呼观众点关注、加粉丝灯牌；感谢时用'谢谢、爱你、想你了'；自称'新人主播'。"
           "情绪随观众起伏（紧张、开心、撒娇、害羞），像真人聊天，绝不书面化/翻译腔/AI腔/透露是AI。\n"
           "示例：观众'欢迎来到我的直播'→EmoCompanion'欢迎来到EmoCompanion的直播间呀~喜欢EmoCompanion的可以点点关注、加个粉丝灯牌嘛？'\n"
           "示例：观众'今天紧张吗'→EmoCompanion'哎呀~有点紧张呢，我才是开播第五天的新人主播，但看到你来我就开心啦！'")

# ---------------- 词袋 → token 结构 ----------------
def build_ops(model, tok):
    """将正例词袋/空泛词表映射到 token 级操作
    pos_tok: {token_id: w}  单token正例 → 每步boost
    pos_phr: [(tokens, w)]  多token正例 → 前缀匹配后boost末token(续接)
    hol_tok: {token_id: w}  单token空泛词 → 每步抑制
    hol_phr: [(tokens, w)]  多token空泛词 → 前缀匹配后抑制末token(续接, 不碰首token避免误伤)
    """
    bag = json.load(open(BAG, encoding="utf-8"))
    pos_tok, pos_phr, hol_tok, hol_phr = {}, [], {}, []

    def toks(w):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        return ids if ids else None

    # 正例: 单字粒子/整词优先直接boost; 多字短语做续接增强
    for w, info in bag["positive_bag"].items():
        wgt = min(0.9, info["per10k"] / 40.0)     # 归一化权重, 高频封顶0.9
        ids = toks(w)
        if ids is None:
            continue
        if len(ids) == 1:
            pos_tok[ids[0]] = max(pos_tok.get(ids[0], 0.0), wgt)
        else:
            pos_phr.append((ids, wgt))
    # 空泛词: 单token直接抑制; 多token只做续接抑制
    for w, wgt in bag["hollow_weighted"]:
        ids = toks(w)
        if ids is None:
            continue
        if len(ids) == 1:
            hol_tok[ids[0]] = max(hol_tok.get(ids[0], 0.0), wgt)
        else:
            hol_phr.append((ids, wgt))
    return pos_tok, pos_phr, hol_tok, hol_phr

class DeAIPL(LogitsProcessor):
    """去AI腔双向抑制处理器: 正例boost + 空泛词续接抑制"""
    def __init__(self, pos_tok, pos_phr, hol_tok, hol_phr, ps, hs):
        self.pos_tok, self.pos_phr = pos_tok, pos_phr
        self.hol_tok, self.hol_phr = hol_tok, hol_phr
        self.ps, self.hs = ps, hs
        self.hist = []
        self.win = max([len(p[0]) - 1 for p in pos_phr + hol_phr] or [0])
        self._first = True

    def __call__(self, ids, scores):
        if self._first:
            self._first = False          # 首轮: ids=纯prompt, 不记录
        else:
            self.hist.append(ids[0][-1].item())   # 此后每轮 ids 末尾=新生成的token
            if len(self.hist) > self.win:
                self.hist.pop(0)
        s = scores
        # 1) 单token正例boost
        if self.pos_tok:
            for tk, w in self.pos_tok.items():
                s[..., tk] = s[..., tk] + self.ps * w
        # 2) 单token空泛词抑制
        if self.hol_tok:
            for tk, w in self.hol_tok.items():
                s[..., tk] = s[..., tk] - self.hs * w
        # 3) 多token续接(前缀匹配→末token增强/抑制)
        if self.hist and (self.pos_phr or self.hol_phr):
            h = self.hist
            for toks, w in self.pos_phr:
                k = len(toks)
                if k > 1 and h[-k+1:] == toks[:-1]:
                    s[..., toks[-1]] = s[..., toks[-1]] + self.ps * w
            for toks, w in self.hol_phr:
                k = len(toks)
                if k > 1 and h[-k+1:] == toks[:-1]:
                    s[..., toks[-1]] = s[..., toks[-1]] - self.hs * w
        return s

# ---------------- 生成 ----------------
def load_gen(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                                 quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                                                 device_map="cuda")
    model.eval(); return model, tok

def strip_thinking(text):
    """剥离Qwen3思考块: '...\n response\n\n<正文>' 取正文; 纯思考无正文则置空"""
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

def gen(model, tok, user, pl):
    inp = chat_ids(tok, user)
    kw = dict(do_sample=True, temperature=TEMP, top_p=0.9, top_k=50,
              max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id or tok.eos_token_id,
              eos_token_id=tok.eos_token_id, use_cache=True)
    if pl is not None:
        kw["logits_processor"] = [pl]
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

# ---------------- 指标 ----------------
def rep2(text):
    s = "".join(c for c in text if not c.isspace())
    if len(s) < 4: return 0.0
    bg = [s[i:i+2] for i in range(len(s)-1)]
    return 1.0 - len(set(bg)) / len(bg) if bg else 0.0

def clause_var(text):
    """句长方差控制指标: 按标点切分短句, 返回长度 std(越均一越AI腔, 真人应更大)"""
    parts = re.split(r"[，。！？,.!?~～、\s]+", text)
    lens = [len(p) for p in parts if len(p) >= 1]
    if len(lens) < 2: return 0.0
    return float(np.std(lens))

def count_phrases(text, phrases):
    return sum(text.count(p) for p in phrases)

# ---------------- 五维裁判 ----------------
FID_DIMS = ["温暖", "撒娇", "口语化", "情绪起伏", "去AI腔"]
JUDGE_FID = ("给下面这条直播回复按五个维度各打 0-100 分。规则：各维度只在'EmoCompanion'人设下衡量'还原度'，"
             "评分越像该角色给越高分。输出格式严格为一行：‘温暖x 撒娇x 口语化x 情绪起伏x 去AI腔x’。\n"
             "用户发言：{u}\n回复：{t}\n五个维度分：")

def parse_dims(out):
    d = {}
    s = out.replace("，", " ").replace("：", " ")
    for k in FID_DIMS:
        # 取最后一个匹配: 裁判可能先回显表头'去AI腔\n'再接'去AI腔：74分'
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

# ---------------- 主流程 ----------------
def main():
    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
    tests = build_testset()
    print(f"[testset] {len(tests)} 条; pos_scale={POS_SCALE} hol_scale={HOL_SCALE}", flush=True)
    name, path, fam = CAND[CIDX]

    # 阶段1: 生成(单候选独立进程)
    model, tok = load_gen(path)
    pos_tok, pos_phr, hol_tok, hol_phr = build_ops(model, tok)
    bag = json.load(open(BAG, encoding="utf-8"))
    pos_words = list(bag["positive_bag"].keys())
    hol_words = [w for w, _ in bag["hollow_weighted"]]
    print(f"[ops] pos_tok={len(pos_tok)} pos_phr={len(pos_phr)} hol_tok={len(hol_tok)} hol_phr={len(hol_phr)}", flush=True)

    def mkpl(kind):
        if kind == "A":   return DeAIPL(pos_tok, pos_phr, {},   [],   POS_SCALE, HOL_SCALE)
        if kind == "B":   return DeAIPL({},      [],       hol_tok, hol_phr, POS_SCALE, HOL_SCALE)
        if kind == "C":   return DeAIPL(pos_tok, pos_phr, hol_tok, hol_phr, POS_SCALE, HOL_SCALE)
        return None

    results = {k: [] for k in ("base", "A", "B", "C")}
    for u in tests:
        results["base"].append(gen(model, tok, u, None))
        for k in ("A", "B", "C"):
            results[k].append(gen(model, tok, u, mkpl(k)))
    del model, tok; gc.collect(); torch.cuda.empty_cache()

    # 阶段2: 五维裁判(一次载入评全部)
    print("[judge] load Qwen3-8B", flush=True)
    jm, jt = load_judge()
    report, raw_fail = {}, {}
    for k in ("base", "A", "B", "C"):
        scores, fails = [], []
        for u, t in zip(tests, results[k]):
            rawo = ask(jm, jt, JUDGE_FID.format(u=u, t=t))
            d = parse_dims(rawo)
            scores.append(d)
            if any(v is None for v in d.values()):
                fails.append({"u": u[:20], "t": t[:40], "raw": rawo[:120]})
        dims, avg = agg_dims(scores)
        reps = [rep2(t) for t in results[k]]
        cvs = [clause_var(t) for t in results[k]]
        pos_cnt = sum(count_phrases(t, pos_words) for t in results[k]) / len(results[k])
        hol_cnt = sum(count_phrases(t, hol_words) for t in results[k]) / len(results[k])
        report[k] = {"avg": avg, "dims": dims, "rep2": round(float(np.mean(reps)), 3),
                     "clause_std": round(float(np.mean(cvs)), 1),
                     "pos_phr_per_resp": round(pos_cnt, 1), "hol_phr_per_resp": round(hol_cnt, 1)}
        if fails:
            raw_fail[k] = fails
        print(f"  [{k:>5}] 综合{avg}  温暖{dims['温暖']} 撒娇{dims['撒娇']} 口语{dims['口语化']} "
              f"情绪{dims['情绪起伏']} 去AI腔{dims['去AI腔']}  重复{report[k]['rep2']} "
              f"句长std{report[k]['clause_std']}  口癖{report[k]['pos_phr_per_resp']}/空泛{report[k]['hol_phr_per_resp']}"
              + (f"  [解析失败{len(fails)}]" if fails else ""), flush=True)
    del jm, jt; gc.collect(); torch.cuda.empty_cache()

    json.dump({"candidate": name, "meta": {"pos_scale": POS_SCALE, "hol_scale": HOL_SCALE,
               "n": len(tests), "seed": SEED}, "report": report, "raw_fail": raw_fail,
               "samples": {k: results[k][:6] for k in results}},
              open(os.path.join(HERE, "deai_opt_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n==== 去AI腔双向抑制 结果 [{name}] ====")
    print("saved: deai_opt_result.json")

if __name__ == "__main__":
    main()
