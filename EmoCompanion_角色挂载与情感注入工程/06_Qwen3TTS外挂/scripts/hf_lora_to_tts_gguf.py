# -*- coding: utf-8 -*-
"""
HF PEFT LoRA (safetensors) -> llama-tts GGUF LoRA 转换器（Qwen3-TTS 专用）
===========================================================================
llama-tts.exe 的 --lora 只接受 GGUF 格式 LoRA。官方 convert_lora_to_gguf.py
仅支持标准 LLM 架构；Qwen3-TTS 的 talker 主干在 GGUF 用标准 GGML 命名
(blk.N.attn_q ...)，训练 epoch 命名 是 HF (talker.model.layers.N.self_attn.q_proj ...)，
二者一一对应，直接映射：

  self_attn.q_proj -> attn_q   mlp.gate_proj -> ffn_gate
  self_attn.k_proj -> attn_k   mlp.up_proj   -> ffn_up
  self_attn.v_proj -> attn_v   mlp.down_proj -> ffn_down
  self_attn.o_proj -> attn_output

code_predictor 分支在 GGUF 走 mmproj，llama-tts 无法加载其 LoRA，转换时跳过。

GGUF LoRA 规范：tensor 名 "<dest>.lora_a" / "<dest>.lora_b"，shape
(n_rank, in_features)/(out_features, n_rank)；metadata 需 general.type=ADAPTER、
adapter.type=lora、adapter.lora.alpha、general.architecture=qwen3tts。

用法:
  python hf_lora_to_tts_gguf.py --lora-dir <HF adapter 目录> --out <out.gguf> [--f16]
"""
import argparse
import json
import os
import sys

import numpy as np
import gguf
from safetensors import safe_open


def load_file(st_path):
    """用 safe_open(framework=npy) 读取 safetensors，返回 name->numpy ndarray。"""
    store = {}
    with safe_open(st_path, framework="numpy") as f:
        for k in f.keys():
            store[k] = f.get_tensor(k)
    return store

ATTR_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
LAYER_PREFIX = "base_model.model.talker.model.layers."
SKIP_PREFIX = "base_model.model.talker.code_predictor."


def hf_to_dest(name):
    """HF lora tensor 名 -> (gguf_dest, branch) 或 None。"""
    if name.startswith(SKIP_PREFIX):
        return None
    if not name.startswith(LAYER_PREFIX):
        return None
    rest = name[len(LAYER_PREFIX):]
    parts = rest.split(".")
    if len(parts) < 3 or not parts[0].isdigit():
        return None
    branch = parts[-2]
    if branch not in ("lora_A", "lora_B"):
        return None
    attr = ".".join(parts[1:-2])
    ggml = ATTR_MAP.get(attr)
    if ggml is None:
        return None
    # base GGUF 的 tensor 名带 .weight 后缀(blk.N.attn_k.weight)，llama-tts 加载
    # LoRA 时剥离 .lora_a/.lora_b 后用该名匹配 base，故必须一致。
    return f"blk.{parts[0]}.{ggml}.weight", branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--f16", action="store_true", help="f16 输出(默认 f32)")
    args = ap.parse_args()

    st = os.path.join(args.lora_dir, "adapter_model.safetensors")
    if not os.path.isfile(st):
        sys.exit(f"缺少 {st}")
    cfg = json.load(open(os.path.join(args.lora_dir, "adapter_config.json"),
                         encoding="utf-8")) \
        if os.path.isfile(os.path.join(args.lora_dir, "adapter_config.json")) else {}
    alpha = float(cfg.get("lora_alpha", 32))
    rank = int(cfg.get("r", 16))

    ftype = gguf.LlamaFileType.ALL_F32
    dtype = np.float32
    if args.f16:
        ftype = gguf.LlamaFileType.MOSTLY_F16
        dtype = np.float16

    sd = load_file(st)
    pairs = {}
    skipped = 0
    for name, tensor in sd.items():
        mapped = hf_to_dest(name)
        if mapped is None:
            skipped += 1
            continue
        dest, branch = mapped
        pairs.setdefault(dest, {})[branch] = tensor
    print(f"[lora] 映射目标 {len(pairs)} 个，跳过 {skipped} 个(含 code_predictor)", flush=True)

    good = []
    for dest in sorted(pairs.keys()):
        pair = pairs[dest]
        A, B = pair.get("lora_A"), pair.get("lora_B")
        if A is None or B is None:
            continue
        good.append((dest, A, B))

    w = gguf.GGUFWriter(args.out, "qwen3tts")
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_file_type(ftype)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)
    w.add_name("qwen3tts_lora")
    w.add_architecture()
    for dest, A, B in good:
        a = np.asarray(A, dtype=dtype)
        b = np.asarray(B, dtype=dtype)
        w.add_tensor(dest + ".lora_a", a)
        w.add_tensor(dest + ".lora_b", b)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[ok] 写出 {len(good)} 组 LoRA block -> {args.out}, alpha={alpha}")


if __name__ == "__main__":
    main()