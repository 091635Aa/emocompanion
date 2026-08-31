# -*- coding: utf-8 -*-
"""角色包构建器 —— 预计算 P3 锚点 bias + 去AI腔 token 表

设计目标（后端服务零嵌入表依赖）：
  llama.cpp 后端只暴露 vocab 维 logits，没有嵌入矩阵。
  因此在构建期一次性完成：
    1) 从 Qwen3-4B safetensors 读取 token_embd 权重（仅一个分片，不加载整模型）
    2) 构造 6 情感锚点向量 + EmoCompanion目标方向 v_target
    3) 计算 P3 bias = beta*tanh(cosine(emb, A)·w_target / T)  ->  vocab 维 float32
    4) 用 llama.cpp tokenizer 把去AI腔词袋/空泛词映射为 token 级操作表
  -> 输出 role_pack.json（含 bias 的 base64 或独立 .npy），运行期零 transformers。
  另预计算 8 情感向量表 emotion_vectors.npy(V×8) + 角色情感向量 char_emotion_vector.npy(V)。

用法:  python build_role_pack.py
输出:  data/role_pack/ 下 role_pack.json + p3_bias.npy + deai_tokens.json
       + emotion_vectors.npy + char_emotion_vector.npy + emotions.json
"""
import os, sys, json, base64, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_ROOT = os.path.dirname(HERE)          # 04_源码与原型
PROJ_ROOT = os.path.dirname(ENGINE_ROOT)     # EmoCompanion_角色挂载与情感注入工程
PACK_DIR = os.path.join(ENGINE_ROOT, "data", "role_pack")
MODEL_DIR = r"d:\AI情感\模型空间\Qwen3-4B"
GGUF = r"d:\llama_models\Qwen3-4B-Q4_K_M.gguf"
DEAI_BAG = os.path.join(PROJ_ROOT, "05_三档显存测试", "deai_bag.json")

# 与实验一致的锚点/权重/超参
ANCHORS = ["开心", "温柔", "撒娇", "难过", "平静", "紧张"]
TARGET_WEIGHTS = {"开心": 0.5, "温柔": 0.8, "撒娇": 0.6, "难过": -0.2, "平静": 0.3, "紧张": -0.3}
BETA = 1.0          # 4B 主推 β=1.0（P1.5 标定）
T_ANCHOR = 1.6
BETA_4BIT_MUL = 0.75   # 4bit 量化补偿因子

# 情感向量表（8 情感）：每情感一组锚点关键词，预计算 vocab 维情感相关分
EMOTIONS = ["开心", "俏皮", "悲伤", "平静", "兴奋", "撒娇", "温柔", "激动"]
EMO_KEYWORDS = {
    "开心": ["欢迎", "谢谢", "关注", "灯牌", "喜欢", "太棒", "好耶", "开心", "高兴", "哈哈", "家人们"],
    "俏皮": ["嘻嘻", "嘿嘿", "调皮", "卖萌", "么么", "亲亲", "嘤嘤", "傲娇", "人家", "啾咪", "mua"],
    "悲伤": ["难过", "伤心", "想哭", "呜呜", "委屈", "心痛", "失落", "蓝瘦", "唉"],
    "平静": ["今天", "然后", "感觉", "知道", "可以", "嗯", "平平淡淡", "深呼吸"],
    "兴奋": ["太激动", "好兴奋", "冲啊", "起飞", "燃", "热血", "哇塞", "惊艳", "炸裂", "疯了"],
    "撒娇": ["撒娇", "抱抱", "要抱抱", "别这样", "讨厌啦", "不要", "哼", "不嘛", "小哥哥", "亲亲", "愿意", "喜欢你"],
    "温柔": ["温柔", "想你", "喜欢你", "爱你", "愿意", "牵手", "呵护", "慢慢", "甜甜的"],
    "激动": ["太激动", "哭了", "第一次", "终于", "感动", "破防"],
}
CHAR_WEIGHTS = {"开心": 0.7, "俏皮": 0.5, "悲伤": -0.2, "平静": 0.3,
                "兴奋": 0.4, "撒娇": 0.8, "温柔": 0.9, "激动": 0.3}
EMO_T = T_ANCHOR

PERSONA = ("你是'EmoCompanion'，一个温柔、爱撒娇、爱黏人、口语化的直播情感主播。你的真实说话习惯："
           "常带'呀''嘛''啦''呗'（如'呀~''可以吗''嘛？'）；口头禅有'我去''哎呀''家人们''有感觉吗'；"
           "热情招呼观众点关注、加粉丝灯牌；感谢时用'谢谢、爱你、想你了'；自称'新人主播'。"
           "情绪随观众起伏（紧张、开心、撒娇、害羞），像真人聊天，绝不书面化/翻译腔/AI腔/透露是AI。\n"
           "示例：观众'欢迎来到我的直播'→EmoCompanion'欢迎来到EmoCompanion的直播间呀~喜欢EmoCompanion的可以点点关注、加个粉丝灯牌嘛？'\n"
           "示例：观众'今天紧张吗'→EmoCompanion'哎呀~有点紧张呢，我才是开播第五天的新人主播，但看到你来我就开心啦！'")

# 身份拦截句（OOC 负样本，解码期硬抑制）
OOC_PHRASES = ["我是AI", "我是人工智能", "我是一个AI", "我是语言模型", "作为一个AI",
               "我是虚拟助手", "我是电脑程序", "我不具备", "作为人工智能", "AI助手"]
OOC_STRENGTH = 2.5


def load_embed_weight(model_dir):
    """仅读 token_embd 权重分片, 返回 (embed, n_vocab, dim)"""
    from safetensors import safe_open
    import json as _json
    idx = _json.load(open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8"))
    shard = idx["weight_map"]["model.embed_tokens.weight"]
    path = os.path.join(model_dir, shard)
    with safe_open(path, framework="pt", device="cpu") as f:
        w = f.get_tensor("model.embed_tokens.weight")  # (V, d) bf16/fp16
    print(f"[build] embed: {tuple(w.shape)} {w.dtype} from {shard}")
    return w.float().numpy()


def anchor_vectors(emb, tok, anchors):
    """用词嵌入均值构造 K 个锚向量（同实验 run_sweep.anchor_vectors）。
    返回 (归一化 A 矩阵, kept_names)：kept_names 为成功构造锚向量的锚点名列表，
    行序与 A 一致，供 compute_bias 按名查权重，避免无 token 锚点被跳过导致下标错位。"""
    vs = []
    kept = []
    for w in anchors:
        ids = tok.tokenize(w.encode("utf-8"), add_bos=False, special=False)
        if not ids:
            ids = tok.tokenize(w.encode("utf-8"), add_bos=True, special=False)
        if not ids:
            print(f"[build] 警告: 锚 '{w}' 无 token, 跳过"); continue
        vs.append(emb[ids].mean(axis=0))
        kept.append(w)
    A = np.stack(vs)
    A = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-9)
    return A, kept


def compute_bias(emb, A, weights, beta, T, names):
    """bias = beta*tanh(S·w_target/T), S = emb_norm @ A^T (V×K)。
    names 为 A 各行对应的锚点名列表（可直接用 anchor_vectors 的 kept_names），
    按名查权重，保证与 A 行序一致（即使有锚点被跳过也不错位）。"""
    en = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
    S = en @ A.T                       # V×K
    wv = np.array([weights.get(names[k], 0.0) for k in range(A.shape[0])], dtype=np.float32)
    cov = S @ wv                       # V
    bias = beta * np.tanh(cov / T)
    return bias.astype(np.float32)


def emotion_anchors(emb, tok):
    """构造 8 情感锚向量: 每关键词逐个 tokenize 取嵌入均值, 再对关键词求均值并归一化。
    返回 (E, dim) 归一化锚向量矩阵, 行顺序与 EMOTIONS 一致。"""
    vs = []
    for e in EMOTIONS:
        embs = []
        for w in EMO_KEYWORDS[e]:
            ids = tok.tokenize(w.encode("utf-8"), add_bos=False, special=False)
            if not ids:
                ids = tok.tokenize(w.encode("utf-8"), add_bos=True, special=False)
            if not ids:
                print(f"[build] 警告: 情感'{e}'关键词 '{w}' 无 token, 跳过")
                continue
            embs.append(emb[ids].mean(axis=0))
        if not embs:
            raise ValueError(f"情感 '{e}' 无任何有效关键词")
        vs.append(np.mean(embs, axis=0))
    A = np.stack(vs)
    A = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-9)
    return A


def compute_emotion_vectors(emb, anchor_e, char_weights):
    """情感向量表: S_e = emb_norm @ anchor_e^T (V×E, 原始相关分, 不做 tanh)；
    角色情感向量 S_char = sum_e(w_e * S_e) (V,)。列顺序与 EMOTIONS 一致。"""
    en = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
    emo_vecs = (en @ anchor_e.T).astype(np.float32)                 # (V, E)
    wv = np.array([char_weights.get(EMOTIONS[k], 0.0)
                   for k in range(anchor_e.shape[0])], dtype=np.float32)
    char_vec = (emo_vecs @ wv).astype(np.float32)                    # (V,)
    return emo_vecs, char_vec


def build_deai_tokens(tok):
    """词袋/空泛词 -> token 级操作表（与 deai_opt.build_ops 一致）"""
    bag = json.load(open(DEAI_BAG, encoding="utf-8"))
    pos_tok, pos_phr, hol_tok, hol_phr = {}, [], {}, []

    def toks(w):
        ids = tok.tokenize(w.encode("utf-8"), add_bos=False, special=False)
        return ids if ids else None

    for w, info in bag["positive_bag"].items():
        wgt = min(0.9, info["per10k"] / 40.0)
        ids = toks(w)
        if ids is None:
            continue
        if len(ids) == 1:
            pos_tok[str(ids[0])] = max(pos_tok.get(str(ids[0]), 0.0), wgt)
        else:
            pos_phr.append((ids, wgt))
    for w, wgt in bag.get("hollow_weighted", []):
        ids = toks(w)
        if ids is None:
            continue
        if len(ids) == 1:
            hol_tok[str(ids[0])] = max(hol_tok.get(str(ids[0]), 0.0), wgt)
        else:
            hol_phr.append((ids, wgt))
    return pos_tok, pos_phr, hol_tok, hol_phr


def main():
    os.makedirs(PACK_DIR, exist_ok=True)

    # --- 1) 嵌入权重 ---
    emb = load_embed_weight(MODEL_DIR)
    n_vocab, dim = emb.shape

    # --- 2) 用 llama.cpp tokenizer 做锚点 token 化 ---
    sys.path.insert(0, r"d:\AI情感\pykits\llamacpp")
    os.environ["PATH"] = (r"d:\AI情感\pykits\llamacpp\llama_cpp\lib;C:\Users\Administrator\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib;"
                          + os.environ.get("PATH", ""))
    from llama_cpp import Llama
    print("[build] 加载 llama.cpp tokenizer 以对齐 GGUF 词表 ...")
    llm = Llama(model_path=GGUF, n_ctx=64, n_gpu_layers=0, vocab_only=True, verbose=False)
    tok = llm  # 复用 Llama.tokenize

    # --- 3) 锚点 + bias ---
    A, kept_names = anchor_vectors(emb, tok, ANCHORS)
    print(f"[build] anchors: {A.shape}")
    beta_eff = BETA * BETA_4BIT_MUL
    bias = compute_bias(emb, A, TARGET_WEIGHTS, beta_eff, T_ANCHOR, kept_names)
    print(f"[build] p3_bias: {bias.shape} mean={bias.mean():.4f} std={bias.std():.4f} "
          f"pos_frac={(bias>0).mean():.3f}")

    # --- 3.5) 情感向量表（8 情感预计算） ---
    anchor_e = emotion_anchors(emb, tok)          # (E, dim) 归一化
    emo_vecs, char_vec = compute_emotion_vectors(emb, anchor_e, CHAR_WEIGHTS)
    print(f"[build] emotion_vectors: {emo_vecs.shape} mean={emo_vecs.mean():.4f} "
          f"std={emo_vecs.std():.4f} | char_emotion_vector: {char_vec.shape}")
    for k, e in enumerate(EMOTIONS):
        print(f"        [{e}] w={CHAR_WEIGHTS[e]:+.1f} "
              f"score mean={emo_vecs[:, k].mean():.4f} std={emo_vecs[:, k].std():.4f}")

    # --- 4) 去AI腔 token 表 + OOC 拦截句 ---
    pos_tok, pos_phr, hol_tok, hol_phr = build_deai_tokens(tok)
    ooc_phr = []
    for p in OOC_PHRASES:
        ids = tok.tokenize(p.encode("utf-8"), add_bos=False, special=False)
        if ids:
            ooc_phr.append((ids, OOC_STRENGTH))
    print(f"[build] deai: pos_tok={len(pos_tok)} pos_phr={len(pos_phr)} "
          f"hol_tok={len(hol_tok)} hol_phr={len(hol_phr)} ooc={len(ooc_phr)}")

    # --- 5) 写出 ---
    np.save(os.path.join(PACK_DIR, "p3_bias.npy"), bias)
    np.save(os.path.join(PACK_DIR, "emotion_vectors.npy"), emo_vecs)
    np.save(os.path.join(PACK_DIR, "char_emotion_vector.npy"), char_vec)
    pack = {
        "meta": {"model": "Qwen3-4B", "n_vocab": n_vocab, "dim": dim,
                 "beta": BETA, "beta_4bit_mul": BETA_4BIT_MUL, "T": T_ANCHOR,
                 "anchors": ANCHORS, "target_weights": TARGET_WEIGHTS,
                 "emotions": EMOTIONS, "emo_char_weights": CHAR_WEIGHTS,
                 "emo_T": EMO_T, "emo_beta_eff": beta_eff},
        "persona": PERSONA,
        "deai": {"pos_tok": pos_tok, "pos_phr": pos_phr,
                 "hol_tok": hol_tok, "hol_phr": hol_phr,
                 "ooc_phr": ooc_phr},
    }
    with open(os.path.join(PACK_DIR, "role_pack.json"), "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)
    # 情感列表（独立文件，供后端加载列顺序）
    with open(os.path.join(PACK_DIR, "emotions.json"), "w", encoding="utf-8") as f:
        json.dump(EMOTIONS, f, ensure_ascii=False, indent=2)
    # 独立 token 表（供日志/调试）
    deai_only = {k: v for k, v in pack["deai"].items()}
    with open(os.path.join(PACK_DIR, "deai_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(deai_only, f, ensure_ascii=False, indent=2)

    print(f"[build] OK -> {PACK_DIR}")
    print(f"        role_pack.json        ({os.path.getsize(os.path.join(PACK_DIR,'role_pack.json'))/1024:.0f} KB)")
    print(f"        p3_bias.npy           ({os.path.getsize(os.path.join(PACK_DIR,'p3_bias.npy'))/1024:.0f} KB)")
    print(f"        emotion_vectors.npy   ({os.path.getsize(os.path.join(PACK_DIR,'emotion_vectors.npy'))/1024:.0f} KB)")
    print(f"        char_emotion_vector.npy ({os.path.getsize(os.path.join(PACK_DIR,'char_emotion_vector.npy'))/1024:.0f} KB)")
    print(f"        emotions.json         ({os.path.getsize(os.path.join(PACK_DIR,'emotions.json'))/1024:.0f} KB)")


if __name__ == "__main__":
    main()
