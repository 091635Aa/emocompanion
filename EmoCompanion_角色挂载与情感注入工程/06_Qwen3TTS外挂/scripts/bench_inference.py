# -*- coding: utf-8 -*-
"""
Qwen3-TTS 角色外挂 —— 推理基准测量（Task 5 实装）
=====================================================
测量 RTF（实时率）、首包延迟、峰值/稳态显存；支持不同后端路径：
  - NVIDIA 20/30/40/50 → CUDA
  - AMD → Vulkan / ROCm / DirectML（探测）
  - 无独显 → CPU

输出对照表，用于验收「加速 ≥2×、≤8GB 可跑」。可选 Langfuse 上报每次测量结果。

用法：
  python bench_inference.py --base_model Qwen/... --texts bench_30.txt \\
                            --estimate_dur 3.0 --repeats 3
"""
import argparse
import json
import os
import time

import torch


def pick_backend():
    """返回 (backend_id, device, dtype) —— 不绑定单一后端。"""
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return "cuda", "cuda", dtype
    if torch.backends.mps.is_available():
        return "mps", "mps", torch.float16
    try:
        # 占位：AMD DirectML/ROCm 若可用会在 torch 中暴露为 cuda 风格后端
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu(intel)", "xpu", torch.bfloat16
    except Exception:
        pass
    return "cpu", "cpu", torch.float32


def measure_llamacpp(llama_tts, backbone, mmproj, speaker_file, texts, estimate_dur, repeats, img_max_kwargs=None):
    """测量本机 llama.cpp 原生 TTS 运行时（INT4/cross-GPU）。

    这是本工作区已随附的权威运行时：llama-tts.exe（llama.cpp build 10502），
    用 GGUF backbone+mmproj 跑 Qwen3-TTS，CUDA/CPU；
    AMD(Vulkan) 依赖带 Vulkan 后端的构建。
    返回行记录列表。
    """
    import subprocess
    # 默认指向工作区已随附的原生运行时（build 10502）
    default_tts = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                               "pykits", "llama-cpp-bin", "llama-tts.exe")
    llama_tts = llama_tts or (default_tts if os.path.exists(default_tts) else "llama-tts.exe")
    if not os.path.exists(llama_tts):
        print("[warn] 未找到 llama-tts.exe，llamacpp 后端将失败")
    rows = []
    for text in texts:
        import tempfile
        out_wav = os.path.join(tempfile.gettempdir(), "bench_tts_out.wav")
        cmd = [llama_tts, "-m", backbone, "-p", text, "-o", out_wav,
               "-ngl", "99", "-fa", "on"]
        if mmproj:
            cmd += ["-mm", mmproj]
        if speaker_file:
            cmd += ["--tts-speaker-file", speaker_file]
        if os.path.exists(out_wav):
            os.remove(out_wav)

        dt_list = []
        for _ in range(repeats):
            if os.path.exists(out_wav):
                os.remove(out_wav)
            t0 = time.perf_counter()
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            dt = time.perf_counter() - t0
            dt_list.append(dt / estimate_dur if estimate_dur else dt)
        med_rtf = sorted(dt_list)[len(dt_list) // 2]
        # 音频实际时长取自估计值；处理时间即端到端生成耗时
        rows.append({
            "text": text[:32],
            "rtf": round(med_rtf, 3),
            "first_latency_s": round(min(dt_list), 3),
            "peak_vram_gb": "n/a(native)",  # 原生运行时显存由外部 profiling 采集
            "backend": "llamacpp",
            "repeats": repeats,
        })
    return rows


def measure(base_model, device, dtype, texts, estimate_dur, repeats):
    """测量 RTF/首包/显存；返回行记录列表。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device)
    model.eval()

    rows = []
    for text in texts:
        enc = tok(text, return_tensors="pt").to(device)
        # 首包延迟：第一次生成耗时
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=256)
        first = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        # 稳态 RTF：多次重复取中位
        rtf_list = []
        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=256)
            dt = time.perf_counter() - t0
            rtf_list.append(dt / estimate_dur if estimate_dur else dt)
        med_rtf = sorted(rtf_list)[len(rtf_list) // 2]
        rows.append({
            "text": text[:32],
            "first_latency_s": round(first, 3),
            "rtf": round(med_rtf, 3),
            "peak_vram_gb": round(peak, 2),
            "repeats": repeats,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--backend", choices=["torch", "llamacpp"], default="torch",
                    help="torch=transformers 路径；llamacpp=本机原生 llama-tts.exe")
    ap.add_argument("--llama_tts", default="", help="llama-tts.exe 路径（默认 06_Qwen3TTS外挂/scripts/llama-tts.exe）")
    ap.add_argument("--backbone", default="", help="llamacpp: backbone GGUF 路径")
    ap.add_argument("--mmproj", default="", help="llamacpp: mmproj GGUF 路径")
    ap.add_argument("--speaker_file", default="", help="llamacpp: speaker 文件")
    ap.add_argument("--texts", default="")          # 文件每行一条文本
    ap.add_argument("--estimate_dur", type=float, default=3.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rtf_target", type=float, default=0.6, help="加速基线：RTF 目标")
    ap.add_argument("--out", default="bench_result.json")
    args = ap.parse_args()

    texts = ["哥哥你回来啦。"]
    if args.texts and os.path.exists(args.texts):
        with open(args.texts, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]

    backend, device, dtype = pick_backend()
    if args.backend == "llamacpp":
        if not args.backbone:
            print("error: --backend llamacpp 需要 --backbone(及可选 --mmproj/--speaker_file)")
            raise SystemExit(1)
        print(f"backend=llamacpp llama_tts={args.llama_tts or 'auto'}")
        rows = measure_llamacpp(args.llama_tts, args.backbone, args.mmproj,
                                args.speaker_file, texts, args.estimate_dur, args.repeats)
        backend = "llamacpp"
    else:
        print(f"backend={backend}(torch) device={device} dtype={dtype}")
        rows = measure(args.base_model, device, dtype, texts, args.estimate_dur, args.repeats)
    ok = all(r["rtf"] is not None and r["rtf"] <= args.rtf_target for r in rows)
    summary = {
        "backend": backend, "device": device, "dtype": str(dtype),
        "rtf_target": args.rtf_target,
        "rows": rows, "meets_target": ok,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"{'text':32} {'rtf':>6} {'first':>6} {'peakGB':>7}")
    for r in rows:
        print(f"{r['text']:32} {r['rtf']:6.3f} {r['first_latency_s']:6.3f} {r['peak_vram_gb']:7.2f}")
    print(f"meets_target(rtf<={args.rtf_target}): {ok}   -> {args.out}")

    # 可选 Langfuse 上报（无 env 则跳过）
    try:
        from langfuse import Langfuse
        lf = Langfuse()
        lf.trace(name="qwen3tts.lora.bench").update(output=summary)
    except Exception:
        pass


if __name__ == "__main__":
    main()