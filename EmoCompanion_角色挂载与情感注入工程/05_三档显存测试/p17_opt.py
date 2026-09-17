# -*- coding: utf-8 -*-
"""P1.7 优化轮 —— 正反向传播同时开启, 系统扫描可优化维度

在 p17_reverse.py 结论(E16=0.6反向+0.4正向 二次合并, 保真88.0全场最优)基础上:
  A_base      无注入基线(同轮复测)
  F040/F060/F080  二次合并配比扫描 γ_bwd∈{0.4,0.6,0.8} (F060=上轮最优配方, 同轮参照)
  FUSE_BI     正反向双向推理增强: 反向=3条不同情感角色续写的梯度求和(一次batch反向)
              + 多层(12/18/24, 权重0.3/0.4/0.3)加权; 正向=多层思考向量
  FUSE_DECAY  中途插入优化: 前8步全强→线性衰减至0.3(F060向量), 替代上轮失败的硬中段启动
  FUSE_POPT   提示词优化: 探针系统词追加"先在心里想情绪和口吻"引导 + 双向增强
指标与注入机制同 P3/P1.7: logits += β·tanh((emb_norm·v̂)/T), β=0.375, T=1.6
用法: python p17_opt.py [n_turns]
"""
import os, sys, json, time, gc, re
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sweep as rs
import p15_sweep as p15
import p17_reverse as p17

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_PATH = p17.GEN_PATH
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LAYERS, LAYER_W = [12, 18, 24], [0.3, 0.4, 0.3]

# 3 条不同情感的角色续写(反向传播的角色条件损失, batch 一次求和)
CONT_LIST = [
    "呀~你来啦！我超开心的嘛！今天有你陪我，感觉整个直播间都亮啦。",   # 开心
    "哎呀，别难过嘛，EmoCompanion给你抱抱，会温柔地一直陪着你哦。",             # 温柔/撒娇
    "嘿嘿，家人们今天也超有感觉的，就是有点小紧张呢，包涵一下嘛。",     # 俏皮/紧张
]
PERSONA_OPT = p17.PERSONA + "\n（在心里先用一两句话想想：EmoCompanion会带着什么情绪、用什么口吻回应这位观众，再组织语言。）"

def fwd_think_vector_multi(model, probe_ids, think_ids, layers=LAYERS, weights=LAYER_W):
    """多层正向思考向量: 各层思考段隐状态均值→归一化→加权融合→归一化"""
    ids = torch.cat([probe_ids, think_ids], dim=1)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    Lp = probe_ids.shape[1]
    vs = []
    for l in layers:
        seg = out.hidden_states[l][0][Lp:]
        v = seg.mean(dim=0).float()
        vs.append(v / v.norm().clamp_min(1e-9))
    v = sum(w * vv for w, vv in zip(weights, vs))
    return v / v.norm().clamp_min(1e-9)

def bwd_role_vector_multi(model, tok, probe_ids, think_ids, conts=CONT_LIST,
                          layers=LAYERS, weights=LAYER_W):
    """多层多续写反向向量: 3条角色续写 batch 成一批, 损失求和, 一次反向取多层梯度"""
    prefix = torch.cat([probe_ids, think_ids], dim=1)          # [1, Lp]
    cont_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in conts]
    P = max(len(c) for c in cont_ids)
    pad_id = tok.pad_token_id or tok.eos_token_id or 0
    dev = prefix.device
    rows, labs = [], []
    for c in cont_ids:
        pad = [pad_id] * (P - len(c))
        rows.append(torch.cat([prefix[0], torch.tensor(c + pad, device=dev)]))
        labs.append(torch.tensor([-100] * prefix.shape[1] + c + [-100] * (P - len(c)), device=dev))
    ids = torch.stack(rows)                                     # [3, Lp+P]
    labels = torch.stack(labs)
    out = model(ids, output_hidden_states=True, use_cache=False)
    logits = out.logits.float()
    sl = logits[:, :-1, :]
    loss = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), labels[:, 1:].reshape(-1),
                           ignore_index=-100)
    hs_list = [out.hidden_states[l] for l in layers]
    grads = torch.autograd.grad(loss, hs_list)
    Lp = prefix.shape[1]
    vs = []
    for g in grads:                                             # [3, T, d]
        gv = -g[:, Lp - 4:Lp, :].mean(dim=(0, 1)).float()       # 行均值×截断点位置均值, 负梯度=上升方向
        vs.append(gv / gv.norm().clamp_min(1e-9))
    v = sum(w * vv for w, vv in zip(weights, vs))
    return v / v.norm().clamp_min(1e-9)

class DecayPL(LogitsProcessor):
    """衰减调度注入: 前8步全强, 之后线性衰减至 0.3 (中途插入的软着陆版)"""
    def __init__(self, bias):
        self.bias = bias
        self.step = 0
        self.ents = []
    def _ent(self, scores):
        v = scores.topk(500, -1).values
        p = torch.softmax(v, -1).clamp_min(1e-12)
        return -(p * p.log()).sum(-1).mean().item()
    def scale(self, step):
        if step < 8:
            return 1.0
        return max(0.3, 1.0 - 0.7 * (step - 8) / 32.0)
    def __call__(self, ids, scores):
        self.ents.append(self._ent(scores))
        if self.bias is not None:
            s = self.scale(self.step)
            self.step += 1
            return scores + (self.bias * s).to(scores.device, scores.dtype)
        self.step += 1
        return scores

def gen_with(model, tok, user, bias, pl_cls, start_step=0):
    inp = p17.chat_ids(tok, user, thinking=False)
    if pl_cls is p17.StepBiasPL:
        pl = pl_cls(bias, start_step)
    else:
        pl = pl_cls(bias)
    t0 = time.time()
    out = model.generate(inp, do_sample=True, temperature=0.9, top_p=0.9, top_k=50,
                         max_new_tokens=80, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         logits_processor=[pl], eos_token_id=tok.eos_token_id, use_cache=True)
    dt = time.time() - t0
    ntok = out.shape[-1] - inp.shape[-1]
    text = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
    ent = float(np.mean(pl.ents)) if pl.ents else 0.0
    return text, ent, ntok / dt if dt else 0.0

def main():
    torch.manual_seed(21); np.random.seed(21)
    tests = p15.build_testset()[:N]
    print(f"[testset] {len(tests)} 条 | 双向融合 | layers={LAYERS} w={LAYER_W} | "
          f"conts={len(CONT_LIST)} | β={p17.BETA17} T={p17.T17}", flush=True)

    tok = AutoTokenizer.from_pretrained(GEN_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_PATH, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="cuda")
    model.eval()
    L18 = int(model.config.num_hidden_layers * 0.5)

    emb = rs.build_embedding_matrix(model)
    A = rs.anchor_vectors(emb, tok, rs.ANCHORS)
    wv = torch.tensor([rs.TARGET_WEIGHTS.get(k, 0.0) for k in rs.ANCHORS],
                      dtype=torch.float32, device=A.device)
    tgt_dir = F.normalize(A.T @ wv, dim=0)

    conds = ["A_base", "F040", "F060", "F080", "FUSE_BI", "FUSE_DECAY", "FUSE_POPT"]
    cols = {c: [] for c in conds}
    geo = {"cos18": [], "cos_ml": [], "cos_bwdml_tgt": [], "cos_b18_bml": []}
    timing = {"probe_ms": [], "fwd18_ms": [], "bwd18_ms": [], "fwd_ml_ms": [], "bwd_ml_ms": [],
              "popt_probe_ms": [], "popt_vec_ms": []}

    for ti, u in enumerate(tests):
        # 标准探针(角色提示词)
        t0 = time.time()
        probe_ids, think16 = p17.probe_think_ids(model, tok, u, p17.K_PROBE)
        timing["probe_ms"].append((time.time() - t0) * 1000)
        t0 = time.time()
        v_fwd18 = p17.fwd_think_vector(model, probe_ids, think16, L18)
        timing["fwd18_ms"].append((time.time() - t0) * 1000)
        t0 = time.time()
        v_bwd18 = p17.bwd_role_vector(model, tok, probe_ids, think16, L18)
        timing["bwd18_ms"].append((time.time() - t0) * 1000)
        # 双向增强(多层+多续写)
        t0 = time.time()
        v_fwd_ml = fwd_think_vector_multi(model, probe_ids, think16)
        timing["fwd_ml_ms"].append((time.time() - t0) * 1000)
        t0 = time.time()
        v_bwd_ml = bwd_role_vector_multi(model, tok, probe_ids, think16)
        timing["bwd_ml_ms"].append((time.time() - t0) * 1000)
        # 提示词优化探针
        t0 = time.time()
        probe_p, think_p = p17.probe_think_ids(model, tok, u, p17.K_PROBE, system=PERSONA_OPT)
        v_fwd_p = fwd_think_vector_multi(model, probe_p, think_p)
        v_bwd_p = bwd_role_vector_multi(model, tok, probe_p, think_p)
        timing["popt_probe_ms"].append((time.time() - t0) * 1000)
        timing["popt_vec_ms"].append(timing["fwd_ml_ms"][-1] + timing["bwd_ml_ms"][-1])

        geo["cos18"].append(float(F.cosine_similarity(v_fwd18, v_bwd18, dim=0)))
        geo["cos_ml"].append(float(F.cosine_similarity(v_fwd_ml, v_bwd_ml, dim=0)))
        geo["cos_bwdml_tgt"].append(float(F.cosine_similarity(v_bwd_ml, tgt_dir, dim=0)))
        geo["cos_b18_bml"].append(float(F.cosine_similarity(v_bwd18, v_bwd_ml, dim=0)))

        biases = {
            "A_base": None,
            "F040": p17.vec_to_bias(p17.fuse(v_bwd18, v_fwd18, 0.4), emb),
            "F060": p17.vec_to_bias(p17.fuse(v_bwd18, v_fwd18, 0.6), emb),
            "F080": p17.vec_to_bias(p17.fuse(v_bwd18, v_fwd18, 0.8), emb),
            "FUSE_BI": p17.vec_to_bias(p17.fuse(v_bwd_ml, v_fwd_ml, 0.6), emb),
            "FUSE_DECAY": p17.vec_to_bias(p17.fuse(v_bwd18, v_fwd18, 0.6), emb),
            "FUSE_POPT": p17.vec_to_bias(p17.fuse(v_bwd_p, v_fwd_p, 0.6), emb),
        }
        print(f"[turn {ti+1}/{len(tests)}] {u[:20]} | cos18={geo['cos18'][-1]:.3f} "
              f"cos_ml={geo['cos_ml'][-1]:.3f} cos(b18,bml)={geo['cos_b18_bml'][-1]:.3f}", flush=True)

        for c in conds:
            pl_cls = DecayPL if c == "FUSE_DECAY" else p17.StepBiasPL
            text, ent, spd = gen_with(model, tok, u, biases[c], pl_cls)
            cols[c].append({"u": u, "text": text, "ent": ent, "rep": p15.rep2(text), "spd": spd})

    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del model, emb, A
    gc.collect(); torch.cuda.empty_cache()

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
        print(f"  {c:<11} 保真{report[c]['fidelity']}  熵{report[c]['entropy']}  "
              f"重复{report[c]['rep2']}  {report[c]['speed']}t/s  词命中{report[c]['tone_hit']}", flush=True)
    del jm, jt; gc.collect(); torch.cuda.empty_cache()

    timing_mean = {k: round(float(np.mean(v)), 1) for k, v in timing.items()}
    geo_mean = {k: round(float(np.mean(v)), 4) for k, v in geo.items()}
    result = {
        "meta": {"model": "Qwen3-4B(4bit)", "n": len(tests), "seed": 21,
                 "layers": LAYERS, "layer_w": LAYER_W, "n_conts": len(CONT_LIST),
                 "beta": p17.BETA17, "T": p17.T17,
                 "torch": torch.__version__,
                 "gpu": torch.cuda.get_device_name(0),
                 "vram_peak_GB": round(vram_peak, 2)},
        "report": report, "geometry": geo_mean, "timing_ms": timing_mean,
        "samples": {c: [{"u": it["u"][:20], "t": it["text"][:80]} for it in cols[c][:3]] for c in conds},
    }
    json.dump(result, open(os.path.join(HERE, "p17_opt.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n==== P1.7 优化轮 ====")
    print("timing(ms):", timing_mean)
    print("geometry:", geo_mean)
    print("saved: p17_opt.json")

if __name__ == "__main__":
    main()
