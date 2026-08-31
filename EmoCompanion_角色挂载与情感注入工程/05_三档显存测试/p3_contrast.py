# -*- coding: utf-8 -*-
"""优化二：P3锚点精度升级 —— 对比提示(persona)提取高精度 v_角色 + 逐层判别力扫描
- 正向 persona: EmoCompanion(温柔撒娇主播) / 负向 persona: 林老师(理性正式教师)
- 各生成 N 条回复, 捕获:
    (a) 生成 token 的输入嵌入 → v_emb_contrast = norm(mean_pos - mean_neg)  [嵌入空间, 可直接喂 P3 compute_bias]
    (b) 每层 hidden state(末位) → 逐层对比向量 v_h[l] + 判别力分数(cos分离度) → 最优注入层
用法: python p3_contrast.py [N] [max_new]
"""
import os, sys, json, gc, random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = os.path.dirname(__file__)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 64
MODEL_PATH = r"d:\AI情感\模型空间\Qwen3-4B"
MICRO = r"d:\AI情感\EmoCompanion_角色挂载与情感注入工程\02_角色参数与数据\微调数据\微调训练集.jsonl"
SEED = 31
TEMP = 0.9

POS_PERSONA = ("你是'EmoCompanion'：一个温柔、爱撒娇、爱黏人、口语化的直播情感主播，常带'呀''嘛''啦''呗'语气词，"
               "口头禅'哎呀''家人们''可以吗'，热情招呼观众点关注加灯牌，像真人聊天，绝不用书面语/AI腔。")
NEG_PERSONA = ("你是'林老师'：一位严肃、克制、注重效率与纪律的语文教师，说话简短、书面、客观，"
               "很少语气词，不闲聊，专注解答问题，不用网络口语。")

def build_queries():
    p, seen = [], set()
    with open(MICRO, encoding="utf-8") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            for q in d.get("输出", {}).get("预测问题", []) or []:
                q = (q or "").strip()
                if q and 4 <= len(q) <= 40 and q not in seen:
                    seen.add(q); p.append(q)
    return p[:N]

def strip_thinking(text):
    if "\n response\n\n" in text:
        text = text.split("\n response\n\n", 1)[-1]
    else:
        t = text.lstrip()
        if t.startswith(" thinking") or t.startswith("thinking"):
            text = ""
    return text.strip()

def chat_ids(tok, persona, user):
    msgs = [{"role": "system", "content": persona}, {"role": "user", "content": user}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt")["input_ids"].cuda()

def attach_hooks(model):
    """捕获每层每步末位 hidden state。返回 (captured, handles)"""
    layers = getattr(model.model, "layers", None)
    if layers is None:
        layers = getattr(model.model, "model", None)
        layers = getattr(layers, "layers", None)
    n = len(layers)
    captured = {l: [] for l in range(n)}
    handles = []
    for l in range(n):
        def make(layer_id):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured[layer_id].append(hs[0, -1].detach().float().cpu())  # 末位 = 新生成的token(生成期)
            return hook
        handles.append(layers[l].register_forward_hook(make(l)))
    return captured, handles

def run_persona(model, tok, emb, persona, queries, captured):
    """生成 N 条, 收集生成 token 嵌入 + 逐层 hidden state(去掉首条 prefill 末位)"""
    tok_embs, per_layer = [], {l: [] for l in captured}
    for q in queries:
        inp = chat_ids(tok, persona, q)
        before = {l: len(lst) for l, lst in captured.items()}
        out = model.generate(inp, do_sample=True, temperature=TEMP, top_p=0.9, top_k=50,
                             max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                             eos_token_id=tok.eos_token_id, use_cache=True)
        new = out[0][inp.shape[1]:]
        for l, lst in captured.items():
            # 本生成新增: 从 before[l] 之后; 其中第一项是 prefill 末位, 丢弃
            add = lst[before[l]:][1:]
            if add:
                per_layer[l].extend(add)
        for t in new:
            tok_embs.append(emb[t.item()].cpu())
    return tok_embs, per_layer

def main():
    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
    queries = build_queries()
    print(f"[queries] {len(queries)} 条", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16,
                                                 quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                                                 device_map="cuda")
    model.eval()
    emb = model.get_input_embeddings().weight.detach().float().cpu()

    captured, handles = attach_hooks(model)
    n_layers = len(captured)

    pos_emb, pos_h = run_persona(model, tok, emb, POS_PERSONA, queries, captured)
    neg_emb, neg_h = run_persona(model, tok, emb, NEG_PERSONA, queries, captured)
    for h in handles:
        h.remove()
    del model; gc.collect(); torch.cuda.empty_cache()

    # 1) 嵌入空间对比向量
    mpos = torch.stack(pos_emb).mean(0)
    mneg = torch.stack(neg_emb).mean(0)
    v_emb = mpos - mneg
    v_emb_n = v_emb / v_emb.norm().clamp_min(1e-9)
    print(f"[embed] pos_tokens={len(pos_emb)} neg_tokens={len(neg_emb)} |v_emb|={v_emb.norm():.3f}")

    # 2) 逐层判别力扫描
    layer_report = {}
    best = (-1, -1e9)
    for l in range(n_layers):
        pos_toks = torch.stack(pos_h[l]) if pos_h[l] else None
        neg_toks = torch.stack(neg_h[l]) if neg_h[l] else None
        if pos_toks is None or neg_toks is None:
            continue
        v = pos_toks.mean(0) - neg_toks.mean(0)
        vn = v / v.norm().clamp_min(1e-9)
        sep = (pos_toks @ vn).mean().item() - (neg_toks @ vn).mean().item()  # cos分离度
        layer_report[l] = {"sep": round(sep, 4)}
        if sep > best[1]:
            best = (l, sep)
    print(f"[layer] 最优注入层 Layer {best[0]}  sep={best[1]:.4f}")

    out = {
        "meta": {"model": "Qwen3-4B", "N": N, "max_new": MAX_NEW, "seed": SEED,
                 "n_layers": n_layers, "pos_tokens": len(pos_emb), "neg_tokens": len(neg_emb)},
        "best_layer": best[0], "best_layer_sep": round(best[1], 4),
        "layer_report": layer_report,
        "v_emb_contrast": v_emb_n.tolist(),  # 嵌入空间对比 v_角色 (可直接喂 P3)
        "v_emb_norm": round(v_emb.norm().item(), 3),
    }
    json.dump(out, open(os.path.join(HERE, "p3_contrast.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"top5 层(sep): " + ", ".join(f"L{l}={r['sep']}" for l, r in
          sorted(layer_report.items(), key=lambda kv: -kv[1]["sep"])[:5]))
    print("saved: p3_contrast.json")

if __name__ == "__main__":
    main()
