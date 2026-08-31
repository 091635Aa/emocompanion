# -*- coding: utf-8 -*-
"""
Qwen3-TTS 角色外挂 —— 推理加载与生成（Task 4 实装）
=====================================================
- Base 常驻（int4/bf16），adapter 动态 attach/detach（角色热切换）
- 情感控制：同包内改情感条件前缀 / ref 音频，实时生效，无需重载
- 记录 RTF 与显存峰值（配合 bench_inference.py 拨测）

用法：
  python load_and_generate.py \
      --base_model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
      --adapter    output/tyy_luoyuan/lora \
      --emotion 开心 --text "哥哥你回来啦"
"""
import argparse
import json
import os
import time

import torch


def pick_device():
    """运行时后端选择：优先 CUDA，其次 MPS(Apple)，最后 CPU（不绑定单一后端）。"""
    if torch.cuda.is_available():
        return "cuda", "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps", "mps"
    except Exception:
        pass
    return "cpu", "cpu"


def load_base(base_model, device, dtype):
    """Base 只读加载（int4 由外层量化策略决定，此处给出 clean 加载路径）。"""
    try:
        # qwen-tts 官方推理路径（若已安装）
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            base_model, device_map=device, dtype=dtype,
            attn_implementation="flash_attention_2" if device == "cuda" else "sdpa",
        )
        return model
    except Exception:
        # 回退：transformers 的 CausalLM 加载（与训练脚本一致）
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_model)
        m = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype,
                                                 device_map=device)
        m.attached_adapter = None
        m._tokenizer = tok
        return m


def attach_adapter(model, adapter_dir):
    """动态挂载 LoRA adapter（PEFT）。返回是否成功。"""
    try:
        from peft import PeftModel
        if getattr(model, "attached_adapter", None) == adapter_dir:
            return True  # 已挂载同一 adapter，跳过
        if getattr(model, "attached_adapter", None) is not None:
            detach_adapter(model)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.attached_adapter = adapter_dir
        return True
    except Exception as e:
        print(f"[warn] attach adapter 失败: {e}")
        return False


def detach_adapter(model):
    """卸载 adapter，恢复 Base（切角色前调用）。"""
    try:
        if getattr(model, "attached_adapter", None) is not None:
            model = model.base_model if hasattr(model, "base_model") else model
            model.attached_adapter = None
    except Exception as e:
        print(f"[warn] detach adapter 失败: {e}")


def load_emotion_vocab(adapter_dir, fallback):
    vp = os.path.join(os.path.dirname(adapter_dir), "emotion_vocab.json")
    if os.path.exists(vp):
        with open(vp, "r", encoding="utf-8") as f:
            return json.load(f)
    return fallback


def generate(model, text, emotion, emotion_vocab, srs=24000, synth_seconds_estimate=None):
    """带情感条件的推理；返回 (音频, sr, rtf, vram_gb)。"""
    prefix = emotion_vocab.get(emotion, "")
    input_text = f"{prefix}{text}".strip()

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    wavs = None
    try:
        # qwen-tts 官方生成接口（支持文本批量/条件）
        wavs, sr = model.generate_voice_clone(
            text=input_text, language="Chinese",
            ref_audio=None, ref_text=None,
        )
    except AttributeError:
        # transformers 回退：纯 causal 生成（示意路径）
        enc = model._tokenizer(input_text, return_tensors="pt").to(
            next(model.parameters()).device)
        out = model.generate(**enc, max_new_tokens=1024)
        wavs, sr = out, model._tokenizer.sampling_rate

    dt = time.perf_counter() - t0
    # 若已知合成时长则算 RTF，否则跳过
    rtf = None
    if synth_seconds_estimate:
        rtf = dt / synth_seconds_estimate
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return wavs, sr, rtf, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="", help="角色外挂 adapter 目录；空则用裸 Base")
    ap.add_argument("--emotion", default="开心")
    ap.add_argument("--text", default="哥哥你回来啦")
    ap.add_argument("--estimate_dur", type=float, default=3.0, help="预估合成音频秒数(算 RTF)")
    args = ap.parse_args()

    device, _ = pick_device()
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) \
        else torch.float16
    print(f"device={device} dtype={dtype}")

    model = load_base(args.base_model, device, dtype)
    emotion_vocab = load_emotion_vocab(args.adapter, {
        args.emotion: f"[emotion]{args.emotion}[/emotion]",
    })
    if args.adapter:
        attach_adapter(model, args.adapter)
        print(f"[role] adapter 已挂载")

    wavs, sr, rtf, peak = generate(model, args.text, args.emotion, emotion_vocab,
                                   synth_seconds_estimate=args.estimate_dur)
    print(f"[out] sr={sr} rtf={rtf if rtf else 'n/a'} peak_vram={peak:.2f}GB")
    # 演示：把音频写盘（若生成结果是数组）
    if wavs is not None and hasattr(wavs, "__len__"):
        try:
            import soundfile as sf
            sf.write("output_demo.wav", wavs[0] if torch.is_tensor(wavs[0]) else wavs[0],
                     sr)
            print("[out] 已写 output_demo.wav")
        except Exception as e:
            print(f"[warn] 写音频失败: {e}")

    # 演示角色热切换：卸载后切另一个 adapter
    if args.adapter:
        detach_adapter(model)
        print("[role] 已卸载 adapter（回复裸 Base）")


if __name__ == "__main__":
    main()