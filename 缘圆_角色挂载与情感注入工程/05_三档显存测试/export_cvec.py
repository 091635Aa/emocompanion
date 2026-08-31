# -*- coding: utf-8 -*-
"""P1.7 → GGUF Control Vector 导出器 —— 把向量库蒸馏为 llama.cpp --control-vector 可用的 GGUF

llama.cpp 控制向量格式(对齐 repeng export_gguf): 架构 "controlvector", 张量名 "direction.<layer>",
对每个层 i: l_out[i] += scale × direction[i] (未导出的层视为零向量)。
层加权沿用 12/18/24 → 0.3/0.4/0.3; 强度 = alpha × resid_norm(与残差流尺度对齐, alpha 可调)。
用法: python export_cvec.py [alpha]
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PKG = r"d:\AI情感\缘圆_角色挂载与情感注入工程\04_源码与原型\.venv\Lib\site-packages"
if VENV_PKG not in sys.path:
    sys.path.insert(0, VENV_PKG)
import gguf  # noqa: E402

ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
LAYERS, LW = [12, 18, 24], [0.3, 0.4, 0.3]
EMO_W = {"开心": 0.5, "俏皮": 0.4, "撒娇": 0.6, "温柔": 0.8, "平静": 0.2, "兴奋": 0.3, "激动": 0.3, "悲伤": -0.2}

def main():
    z = np.load(os.path.join(HERE, "p17_bank.npz"), allow_pickle=True)
    emo_vecs, names = z["emo_vecs"], list(z["emo_names"])
    resid_norm = float(z["resid_norm"])
    d = emo_vecs.shape[1]
    w = np.array([EMO_W.get(n, 0.0) for n in names], dtype="float32")
    v = (emo_vecs.T @ w)                      # 加权融合全局"缘圆方向"
    nv = np.linalg.norm(v)
    if nv > 1e-9:
        v /= nv
    lw = np.array(LW, dtype="float32"); lw /= lw.sum()
    strength = ALPHA * resid_norm
    out_path = os.path.join(HERE, f"cvec_yuanyuan_p17.gguf")
    writer = gguf.GGUFWriter(out_path, "controlvector")
    writer.add_string("general.name", "yuanyuan-p17-fused-direction")
    writer.add_string("general.description",
                      f"P1.7 fused fwd+bwd steering (alpha={ALPHA}, resid_norm={resid_norm:.1f})")
    for L, wl in zip(LAYERS, lw):
        writer.add_tensor(f"direction.{L}", (v * wl * strength).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    meta = {"alpha": ALPHA, "resid_norm": resid_norm, "strength": strength,
            "layers": LAYERS, "layer_w": lw.tolist(), "n_embd": int(d),
            "emo_w": EMO_W, "out": out_path}
    json.dump(meta, open(os.path.join(HERE, "cvec_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("exported:", meta)

if __name__ == "__main__":
    main()
