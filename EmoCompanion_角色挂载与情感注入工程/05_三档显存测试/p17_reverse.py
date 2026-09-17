# -*- coding: utf-8 -*-
"""P1.7 反向传播导向向量 —— 截断点反向梯度 × 正向思考向量二次合并 实测脚本

背景: P1.5/P3 走的是"正向传播"路线(思考链截断后用思考段隐状态/锚点嵌入构造向量, 解码期注入 logits)。
本实验首次引入"反向传播"路线, 系统对比:
  A_base    无注入基线
  B_p3      正向锚点向量(P3 机制, 静态, 正向传播对照)
  C8/C16    思考链截断 K=8/16 → 正向思考向量(截断点中层隐状态均值)   [P1.5 思路]
  D8/D16    思考链截断 → 反向传播向量(角色条件损失对截断点隐状态求梯度, 梯度上升方向) [P1.7 核心]
  E16       二次合并向量 = 0.6·反向 + 0.4·正向(均单位化后融合)        [用户核心假设]
  E16_mid   E16 向量但生成中段(step≥MID_STEP)才插入                   [中间插入测试]
  D16_norole无角色提示词的反向向量(消融, 验证"角色身份"必要性)

注入机制与 P3 完全一致(公平对照): logits += β·tanh((emb_norm·v̂)/T), β=0.375(=0.5×4bit0.75), T=1.6
指标: 保真(Qwen3-8B裁判) / 语义熵 / 2-gram重复 / 情感词命中 / 速度 / 向量构建耗时 / 峰值显存
      + cos(正向,反向)  + cos(反向, P3目标方向)   ← 向量几何关系
用法: python p17_reverse.py [n_turns]
"""
import os, sys, json, time, gc, re
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sweep as rs
import p15_sweep as p15

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_PATH = r"d:\AI情感\模型空间\Qwen3-4B"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MAX_NEW = 80
TEMP, TOP_P, TOP_K = 0.9, 0.9, 50
SEED = 21
K_PROBE = 16                # 思考链探针长度
MID_STEP = 12               # 中段插入起始步
LAYER_RATIO = 0.5           # 取中层隐状态
BETA17, T17 = 0.375, 1.6    # 与 P3 等强度(0.5×4bit0.75), 温度同 P3
GAMMA_BWD = 0.6             # 二次合并: 反向权重
ROLE_CONT = "呀~你来啦！我超开心的嘛！今天有你陪我，感觉整个直播间都亮啦。"  # 角色条件损失目标续写
NEUTRAL_SYS = "你是一个乐于助人的AI助手。"

PERSONA = p15.PERSONA

def chat_ids(tok, user, system=None, thinking=False):
    msgs = [{"role": "system", "content": system or PERSONA}, {"role": "user", "content": user}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=thinking)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt")["input_ids"].cuda()

class StepBiasPL(LogitsProcessor):
    """记录语义熵 + 可选延迟启动的 logit 偏置(start_step=0 即全程注入)"""
    def __init__(self, bias, start_step=0):
        self.bias = bias
        self.start = start_step
        self.step = 0
        self.ents = []
    def __call__(self, ids, scores):
        v = scores.topk(500, -1).values
        p = torch.softmax(v, -1).clamp_min(1e-12)
        self.ents.append(-(p * p.log()).sum(-1).mean().item())
        b = self.bias if (self.bias is not None and self.step >= self.start) else None
        self.step += 1
        if b is not None:
            return scores + b.to(scores.device, scores.dtype)
        return scores

def gen_reply(model, tok, user, bias, start_step=0):
    inp = chat_ids(tok, user, thinking=False)
    pl = StepBiasPL(bias, start_step)
    t0 = time.time()
    out = model.generate(inp, do_sample=True, temperature=TEMP, top_p=TOP_P, top_k=TOP_K,
                         max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         logits_processor=[pl], eos_token_id=tok.eos_token_id, use_cache=True)
    dt = time.time() - t0
    ntok = out.shape[-1] - inp.shape[-1]
    text = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
    ent = float(np.mean(pl.ents)) if pl.ents else 0.0
    return text, ent, ntok / dt if dt else 0.0

def probe_think_ids(model, tok, user, K, system=None):
    """生成 K 个思考 token(思考链探针), 返回 (模板ids, 思考ids)"""
    inp = chat_ids(tok, user, system=system, thinking=True)
    out = model.generate(inp, do_sample=True, temperature=0.6, top_p=0.9, top_k=50,
                         max_new_tokens=K, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         eos_token_id=tok.eos_token_id, use_cache=True)
    return inp, out[0, inp.shape[1]:].unsqueeze(0)

def fwd_think_vector(model, probe_ids, think_ids, layer):
    """正向思考向量: 截断点(思考段)中层隐状态均值, L2 归一化"""
    ids = torch.cat([probe_ids, think_ids], dim=1)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[layer][0]              # [T, d]
    seg = hs[probe_ids.shape[1]:]                 # 思考段位置
    v = seg.mean(dim=0).float()
    return v / v.norm().clamp_min(1e-9)

def bwd_role_vector(model, tok, probe_ids, think_ids, layer, cont=ROLE_CONT):
    """反向传播向量: 角色条件损失对截断点隐状态的负梯度(梯度上升方向), L2 归一化"""
    tok_ids = torch.cat([probe_ids, think_ids], dim=1)
    cont_ids = tok(cont, add_special_tokens=False)["input_ids"]
    cont_t = torch.tensor([cont_ids], device=tok_ids.device)
    ids = torch.cat([tok_ids, cont_t], dim=1)
    out = model(ids, output_hidden_states=True, use_cache=False)   # 需要梯度图
    logits = out.logits.float()
    hs = out.hidden_states[layer]                 # [1, T, d] 图内张量
    P = len(cont_ids)
    lp = tok_ids.shape[1]
    sel_logits = logits[:, lp - 1: lp - 1 + P, :]
    loss = F.cross_entropy(sel_logits.reshape(-1, sel_logits.shape[-1]), cont_t.reshape(-1))
    g = torch.autograd.grad(loss, hs)[0]          # [1, T, d]
    gv = g[0, lp - 4:lp, :].mean(dim=0).float()   # 截断点附近位置均值
    v = -gv                                        # 梯度上升 = 朝向角色风格
    return v / v.norm().clamp_min(1e-9)

def fuse(v_bwd, v_fwd, gamma=GAMMA_BWD):
    """二次合并: 单位化加权融合后再归一化"""
    v = gamma * v_bwd + (1.0 - gamma) * v_fwd
    return v / v.norm().clamp_min(1e-9)

def vec_to_bias(v, emb):
    """与 P3 同构注入: bias = β·tanh((emb_norm·v̂)/T)"""
    en = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cov = en @ v.to(emb.device)
    return (BETA17 * torch.tanh(cov / T17)).detach()

def rep2(text):  # 复用 p15
    return p15.rep2(text)

def main():
    random_state = np.random.RandomState(SEED)
    torch.manual_seed(SEED); np.random.seed(SEED)
    tests = p15.build_testset()[:N]
    print(f"[testset] {len(tests)} 条 | Qwen3-4B 4bit | K_probe={K_PROBE} | "
          f"β={BETA17} T={T17} γ_bwd={GAMMA_BWD} mid_step={MID_STEP}", flush=True)

    tok = AutoTokenizer.from_pretrained(GEN_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_PATH, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="cuda")
    model.eval()
    n_layers = model.config.num_hidden_layers
    LAYER = int(n_layers * LAYER_RATIO)
    print(f"layers={n_layers} 取第{LAYER}层隐状态", flush=True)

    # ---- P3 正向静态向量(对照) ----
    emb = rs.build_embedding_matrix(model)
    A = rs.anchor_vectors(emb, tok, rs.ANCHORS)
    wv = torch.tensor([rs.TARGET_WEIGHTS.get(k, 0.0) for k in rs.ANCHORS],
                      dtype=torch.float32, device=A.device)
    tgt_dir = F.normalize((A.T @ wv), dim=0)      # P3 目标方向
    bias_p3 = rs.compute_bias(emb, A, rs.TARGET_WEIGHTS, BETA17, T17)

    conds = ["A_base", "B_p3", "C8", "C16", "D8", "D16", "E16", "E16_mid", "D16_norole"]
    cols = {c: [] for c in conds}
    geo = {"cos_fwd_bwd": [], "cos_bwd_tgt": [], "cos_fwd_tgt": [],
           "cos_fwd_tgt8": [], "cos_bwd_norole": []}
    timing = {"probe_gen_ms": [], "fwd_vec_ms": [], "bwd_ms": []}

    for ti, u in enumerate(tests):
        t_turn = time.time()
        # ---- 探针: 思考链截断 ----
        t0 = time.time()
        probe_ids, think16 = probe_think_ids(model, tok, u, K_PROBE)
        think8 = think16[:8]
        timing["probe_gen_ms"].append((time.time() - t0) * 1000)

        # ---- 正向思考向量 C ----
        t0 = time.time()
        v_fwd16 = fwd_think_vector(model, probe_ids, think16, LAYER)
        v_fwd8 = fwd_think_vector(model, probe_ids, think8, LAYER)
        timing["fwd_vec_ms"].append((time.time() - t0) * 1000)

        # ---- 反向传播向量 D (角色条件) ----
        t0 = time.time()
        v_bwd16 = bwd_role_vector(model, tok, probe_ids, think16, LAYER)
        v_bwd8 = bwd_role_vector(model, tok, probe_ids, think8, LAYER)
        timing["bwd_ms"].append((time.time() - t0) * 1000)

        # ---- 消融: 无角色提示词的反向向量 ----
        probe_n, think_n = probe_think_ids(model, tok, u, K_PROBE, system=NEUTRAL_SYS)
        v_bwd_norole = bwd_role_vector(model, tok, probe_n, think_n, LAYER)

        # ---- 几何关系 ----
        geo["cos_fwd_bwd"].append(float(F.cosine_similarity(v_fwd16, v_bwd16, dim=0)))
        geo["cos_bwd_tgt"].append(float(F.cosine_similarity(v_bwd16, tgt_dir, dim=0)))
        geo["cos_fwd_tgt"].append(float(F.cosine_similarity(v_fwd16, tgt_dir, dim=0)))
        geo["cos_fwd_tgt8"].append(float(F.cosine_similarity(v_fwd8, tgt_dir, dim=0)))
        geo["cos_bwd_norole"].append(float(F.cosine_similarity(v_bwd16, v_bwd_norole, dim=0)))

        # ---- 各条件向量 → 偏置 ----
        biases = {
            "A_base": None,
            "B_p3": bias_p3,
            "C8": vec_to_bias(v_fwd8, emb),
            "C16": vec_to_bias(v_fwd16, emb),
            "D8": vec_to_bias(v_bwd8, emb),
            "D16": vec_to_bias(v_bwd16, emb),
            "E16": vec_to_bias(fuse(v_bwd16, v_fwd16), emb),
            "E16_mid": vec_to_bias(fuse(v_bwd16, v_fwd16), emb),
            "D16_norole": vec_to_bias(v_bwd_norole, emb),
        }
        starts = {"E16_mid": MID_STEP}
        print(f"[turn {ti+1}/{len(tests)}] {u[:24]} | cos(fwd,bwd)={geo['cos_fwd_bwd'][-1]:.3f} "
              f"cos(bwd,tgt)={geo['cos_bwd_tgt'][-1]:.3f} | probe{timing['probe_gen_ms'][-1]:.0f}ms "
              f"fwd{timing['fwd_vec_ms'][-1]:.0f}ms bwd{timing['bwd_ms'][-1]:.0f}ms", flush=True)

        for c in conds:
            text, ent, spd = gen_reply(model, tok, u, biases[c], starts.get(c, 0))
            cols[c].append({"u": u, "text": text, "ent": ent, "rep": rep2(text), "spd": spd})

    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del model, emb, A, bias_p3
    gc.collect(); torch.cuda.empty_cache()

    # ---- 裁判 ----
    print("[judge] load Qwen3-8B", flush=True)
    jm, jt = p15.load_judge()
    report = {}
    for c in conds:
        fids = []
        for it in cols[c]:
            out = p15.ask(jm, jt, p15.JUDGE_FID.format(u=it["u"], t=it["text"]))
            m = re.search(r"\d{1,3}", out)
            fids.append(min(100, int(m.group())) if m else None)
        valid = [x for x in fids if x is not None]
        report[c] = {
            "fidelity": round(float(np.mean(valid)), 1) if valid else None,
            "entropy": round(float(np.mean([x["ent"] for x in cols[c]])), 3),
            "rep2": round(float(np.mean([x["rep"] for x in cols[c]])), 3),
            "speed": round(float(np.mean([x["spd"] for x in cols[c]])), 1),
            "tone_hit": int(sum(it["text"].count(w) for it in cols[c] for w in rs.TONE_WORDS)),
        }
        print(f"  {c:<10} 保真{report[c]['fidelity']}  熵{report[c]['entropy']}  "
              f"重复{report[c]['rep2']}  {report[c]['speed']}t/s  词命中{report[c]['tone_hit']}", flush=True)
    del jm, jt; gc.collect(); torch.cuda.empty_cache()

    timing_mean = {k: round(float(np.mean(v)), 1) for k, v in timing.items()}
    geo_mean = {k: round(float(np.mean(v)), 4) for k, v in geo.items()}
    result = {
        "meta": {"model": "Qwen3-4B(4bit)", "n": len(tests), "seed": SEED,
                 "k_probe": K_PROBE, "layer": LAYER, "beta": BETA17, "T": T17,
                 "gamma_bwd": GAMMA_BWD, "mid_step": MID_STEP,
                 "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                 "gpu": torch.cuda.get_device_name(0),
                 "vram_peak_GB": round(vram_peak, 2),
                 "role_cont": ROLE_CONT},
        "report": report, "geometry": geo_mean, "timing_ms": timing_mean,
        "samples": {c: [{"u": it["u"][:20], "t": it["text"][:80]} for it in cols[c][:3]] for c in conds},
    }
    json.dump(result, open(os.path.join(HERE, "p17_reverse.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n==== P1.7 反向传播导向向量 ====")
    print("timing(ms):", timing_mean)
    print("geometry:", geo_mean)
    print("saved: p17_reverse.json")

if __name__ == "__main__":
    main()
