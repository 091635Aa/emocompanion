# -*- coding: utf-8 -*-
"""P1.7 验证轮 —— n=30 复验最终配置 (A_base / F060 / FUSE_DECAY)
用法: python p17_valid.py [n]
"""
import os, sys, json, re, gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sweep as rs
import p15_sweep as p15
import p17_reverse as p17
import p17_opt as po

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

def main():
    torch.manual_seed(21); np.random.seed(21)
    tests = p15.build_testset()[:N]
    print(f"[valid] n={len(tests)} | conds=A_base/F060/FUSE_DECAY", flush=True)
    tok = AutoTokenizer.from_pretrained(p17.GEN_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        p17.GEN_PATH, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="cuda")
    model.eval()
    L18 = int(model.config.num_hidden_layers * 0.5)
    emb = rs.build_embedding_matrix(model)

    conds = ["A_base", "F060", "FUSE_DECAY"]
    part_path = os.path.join(HERE, "p17_valid_partial.json")
    cols = {c: [] for c in conds}
    start_i = 0
    if os.path.exists(part_path):
        try:
            pj = json.load(open(part_path, encoding="utf-8"))
            cols = {c: pj.get(c, []) for c in conds}
            start_i = min(len(cols[c]) for c in conds)
            print(f"[resume] 从第 {start_i+1} 轮续跑", flush=True)
        except Exception:
            pass
    for ti, u in enumerate(tests[start_i:], start=start_i):
        probe_ids, think16 = p17.probe_think_ids(model, tok, u, p17.K_PROBE)
        v_fwd18 = p17.fwd_think_vector(model, probe_ids, think16, L18)
        v_bwd18 = p17.bwd_role_vector(model, tok, probe_ids, think16, L18)
        bias = p17.vec_to_bias(p17.fuse(v_bwd18, v_fwd18, 0.6), emb)
        for c in conds:
            pl_cls = po.DecayPL if c == "FUSE_DECAY" else p17.StepBiasPL
            b = bias if c != "A_base" else None
            text, ent, spd = po.gen_with(model, tok, u, b, pl_cls)
            cols[c].append({"u": u, "text": text, "ent": ent, "rep": p15.rep2(text), "spd": spd})
        json.dump({c: cols[c] for c in conds}, open(part_path, "w", encoding="utf-8"),
                  ensure_ascii=False)
        if (ti + 1) % 5 == 0:
            print(f"  ...{ti+1}/{len(tests)}", flush=True)

    vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del model, emb; gc.collect(); torch.cuda.empty_cache()

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
            "fidelity_std": round(float(np.std(valid)), 1) if valid else None,
            "entropy": round(float(np.mean([x["ent"] for x in cols[c]])), 3),
            "rep2": round(float(np.mean([x["rep"] for x in cols[c]])), 3),
            "speed": round(float(np.mean([x["spd"] for x in cols[c]])), 1),
            "tone_hit": int(sum(it["text"].count(w) for it in cols[c] for w in rs.TONE_WORDS)),
        }
        print(f"  {c:<11} 保真{report[c]['fidelity']}±{report[c]['fidelity_std']}  "
              f"熵{report[c]['entropy']}  重复{report[c]['rep2']}  词命中{report[c]['tone_hit']}", flush=True)
    del jm, jt; gc.collect(); torch.cuda.empty_cache()

    json.dump({"meta": {"n": len(tests), "seed": 21, "vram_peak_GB": round(vram, 2)},
               "report": report,
               "samples": {c: [{"u": it["u"][:20], "t": it["text"][:70]} for it in cols[c][:3]]
                           for c in conds}},
              open(os.path.join(HERE, "p17_valid.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if os.path.exists(part_path):
        os.remove(part_path)
    print("saved: p17_valid.json")

if __name__ == "__main__":
    main()
