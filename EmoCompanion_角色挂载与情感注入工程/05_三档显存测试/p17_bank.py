# -*- coding: utf-8 -*-
"""P1.7 离线向量库 —— 用真实情感标注回复做续写损失, 预计算 情感×向量 查表库 + 问题嵌入质心路由

数据源: 02_角色参数与数据/微调数据/微调训练集_latest.jsonl
  每条记录取: discrete_emotion_primary(主情感) / 预测问题[0](探针问题) / 最长分段transcript(续写损失, 截60字)
每情感最多 10 条 → 每条: probe(persona)+think16 → 正向向量 + 反向向量(真实回复为续写) → fuse(0.6)
产出:
  p17_bank.npz   emo_vecs[E,d] / emo_centroids[E,d] / emo_names / resid_norm
  p17_bank.json  元数据 + cos(正向,反向) 统计
评估: 10 条测试问题, 三条件 A_base / BankRoute(路由查表, 零探针) / F060(在线探针) 对比
用法: python p17_bank.py [per_emotion] [n_eval]
"""
import os, sys, json, re, gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sweep as rs
import p15_sweep as p15
import p17_reverse as p17

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_PATH = r"d:\AI情感\EmoCompanion_角色挂载与情感注入工程\02_角色参数与数据\微调数据\微调训练集_latest.jsonl"
EMOS = ["开心", "俏皮", "撒娇", "温柔", "平静", "兴奋", "激动", "悲伤"]
PER_EMO = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_EVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 10
CAP = 60

def load_bank_entries():
    """扫描训练集 → [(emotion, question, reply), ...]"""
    per = {e: [] for e in EMOS}
    with open(BANK_PATH, encoding="utf-8-sig") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            out = d.get("输出", {})
            emo_raw = (out.get("discrete_emotion_primary") or "").strip()
            emo = next((e for e in EMOS if e in emo_raw), None)
            if emo is None or len(per[emo]) >= PER_EMO:
                continue
            segs = out.get("时间轴分段") or []
            if not segs:
                continue
            reply = max((s.get("transcript") or "" for s in segs), key=len)[:CAP].strip()
            qs = out.get("预测问题") or []
            if not reply or not qs:
                continue
            per[emo].append((emo, qs[0].strip(), reply))
    entries = [e for lst in per.values() for e in lst]
    return per, entries

def main():
    torch.manual_seed(21); np.random.seed(21)
    per, entries = load_bank_entries()
    print(f"[bank] 每情感配额={PER_EMO} 实际条目={len(entries)} "
          f"分布={{e: len(v) for e, v in per.items() if v}}", flush=True)

    tok = AutoTokenizer.from_pretrained(p17.GEN_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        p17.GEN_PATH, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="cuda")
    model.eval()
    L18 = int(model.config.num_hidden_layers * 0.5)
    emb = rs.build_embedding_matrix(model)          # V×d float32

    vecs, qembs, cos_fb, resid_norms = [], [], [], []
    for i, (emo, q, reply) in enumerate(entries):
        probe_ids, think16 = p17.probe_think_ids(model, tok, q, p17.K_PROBE)
        v_fwd = p17.fwd_think_vector(model, probe_ids, think16, L18)
        v_bwd = p17.bwd_role_vector(model, tok, probe_ids, think16, L18, cont=reply)
        v = p17.fuse(v_bwd, v_fwd, 0.6)
        cos_fb.append(float(F.cosine_similarity(v_fwd, v_bwd, dim=0)))
        with torch.no_grad():
            resid_norm = float(model(probe_ids, output_hidden_states=True, use_cache=False)
                               .hidden_states[L18][0][-1].float().norm())
        resid_norms.append(resid_norm)
        qids = tok(q, add_special_tokens=False)["input_ids"]
        qemb = F.normalize(emb[qids].mean(0), dim=0)
        vecs.append(v.cpu().numpy()); qembs.append(qemb.cpu().numpy())
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(entries)}", flush=True)

    vecs = np.stack(vecs); qembs = np.stack(qembs)
    emo_vecs, emo_cent = [], []
    for e in EMOS:
        idx = [i for i, x in enumerate(entries) if x[0] == e]
        if idx:
            m = vecs[idx].mean(0); m /= (np.linalg.norm(m) + 1e-9)
            c = qembs[idx].mean(0); c /= (np.linalg.norm(c) + 1e-9)
        else:
            m = np.zeros(vecs.shape[1], dtype="float32"); c = m
        emo_vecs.append(m); emo_cent.append(c)
    emo_vecs = np.stack(emo_vecs); emo_cent = np.stack(emo_cent)
    resid_norm_mean = float(np.mean(resid_norms))
    np.savez(os.path.join(HERE, "p17_bank.npz"), emo_vecs=emo_vecs, emo_cent=emo_cent,
             emo_names=np.array(EMOS), resid_norm=np.float32(resid_norm_mean))
    bank_meta = {"per_emo": {e: len(v) for e, v in per.items()}, "n_entries": len(entries),
                 "cos_fwd_bwd_mean": round(float(np.mean(cos_fb)), 4),
                 "resid_norm_l18": round(resid_norm_mean, 2)}
    json.dump(bank_meta, open(os.path.join(HERE, "p17_bank.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[bank] saved p17_bank.npz | {bank_meta}", flush=True)

    # ---- 评估: 路由查表 vs 在线探针 vs 基线 ----
    tests = p15.build_testset()[:N_EVAL]
    conds = ["A_base", "BankRoute", "F060"]
    cols = {c: [] for c in conds}
    routed, cos_bank_probe = [], []
    for u in tests:
        qids = tok(u, add_special_tokens=False)["input_ids"]
        qe = F.normalize(emb[qids].mean(0), dim=0)
        sims = emo_cent @ qe.cpu().numpy()
        ei = int(np.argmax(sims))
        routed.append(EMOS[ei])
        bank_v = torch.tensor(emo_vecs[ei], dtype=torch.float32, device=emb.device)
        bank_bias = p17.vec_to_bias(bank_v, emb)
        probe_ids, think16 = p17.probe_think_ids(model, tok, u, p17.K_PROBE)
        v_fwd = p17.fwd_think_vector(model, probe_ids, think16, L18)
        v_bwd = p17.bwd_role_vector(model, tok, probe_ids, think16, L18)
        cos_bank_probe.append(float(F.cosine_similarity(bank_v, p17.fuse(v_bwd, v_fwd, 0.6), dim=0)))
        f060_bias = p17.vec_to_bias(p17.fuse(v_bwd, v_fwd, 0.6), emb)
        for c in conds:
            b = {"A_base": None, "BankRoute": bank_bias, "F060": f060_bias}[c]
            text, ent, spd = p17.gen_reply(model, tok, u, b)
            cols[c].append({"u": u, "text": text, "ent": ent, "rep": p15.rep2(text), "spd": spd})
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
        report[c] = {"fidelity": round(float(np.mean(valid)), 1) if valid else None,
                     "entropy": round(float(np.mean([x["ent"] for x in cols[c]])), 3),
                     "rep2": round(float(np.mean([x["rep"] for x in cols[c]])), 3),
                     "speed": round(float(np.mean([x["spd"] for x in cols[c]])), 1),
                     "tone_hit": int(sum(it["text"].count(w) for it in cols[c] for w in rs.TONE_WORDS))}
        print(f"  {c:<10} 保真{report[c]['fidelity']}  熵{report[c]['entropy']}  "
              f"重复{report[c]['rep2']}  词命中{report[c]['tone_hit']}", flush=True)
    del jm, jt; gc.collect(); torch.cuda.empty_cache()

    report["_route_dist"] = {e: routed.count(e) for e in EMOS if routed.count(e)}
    report["_cos_bank_probe_mean"] = round(float(np.mean(cos_bank_probe)), 4)
    bank_meta["eval"] = report
    bank_meta["vram_peak_GB"] = round(vram, 2)
    json.dump(bank_meta, open(os.path.join(HERE, "p17_bank.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n==== P1.7 向量库 ====")
    print("路由分布:", report["_route_dist"])
    print("cos(库向量,探针向量)均值:", report["_cos_bank_probe_mean"])
    print("saved: p17_bank.json")

if __name__ == "__main__":
    main()
