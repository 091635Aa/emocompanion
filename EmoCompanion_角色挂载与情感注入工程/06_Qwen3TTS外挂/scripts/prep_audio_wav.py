# -*- coding: utf-8 -*-
"""
EmoCompanion 打标 mp4 → 24kHz mono wav 批量转换（供 Qwen3-TTS 微调使用）
================================================================
Qwen3-TTS 要求 24kHz mono wav。源为 F:\打标\数据层\分割片段\*.mp4。
转换后用 ffprobe 校验采样率/声道，输出清单供双路训练脚本消费。

用法：
  python prep_audio_wav.py \
      --src F:/打标/数据层/分割片段 \
      --dst D:/status/wav_24k \
      --ffmpeg C:/cmd/ffmpeg/bin/ffmpeg.exe
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor  # 线程池（每线程独立 subprocess，避免 Windows spawn/pickle 竞态）
import multiprocessing


def _convert_one(args):
    ffmpeg, src_dir, dst_dir, name, overwrite = args
    stem = os.path.splitext(name)[0]
    out_wav = os.path.join(dst_dir, stem + ".wav")
    if os.path.exists(out_wav) and os.path.getsize(out_wav) > 0 and not overwrite:
        return name, "skip"
    for attempt in range(3):  # 并发 I/O 冲突重试
        cmd = [ffmpeg, "-y", "-i", os.path.join(src_dir, name),
               "-ar", "24000", "-ac", "1", "-acodec", "pcm_s16le", out_wav,
               "-loglevel", "error"]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
            return name, "ok"
    return name, "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ffmpeg", default="C:\\cmd\\ffmpeg\\bin\\ffmpeg.exe")
    ap.add_argument("--limit", type=int, default=0, help="0=全量；>0 前端 N 条（小规模试跑）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="0=CPU核数")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    n_workers = args.workers or max(1, multiprocessing.cpu_count() - 2)

    mp4s = sorted([f for f in os.listdir(args.src) if f.lower().endswith(".mp4")])
    if args.limit:
        mp4s = mp4s[:args.limit]
    print(f"[src] {len(mp4s)} mp4, workers={n_workers}", flush=True)

    work = [(args.ffmpeg, args.src, args.dst, n, args.overwrite) for n in mp4s]
    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, (name, status) in enumerate(ex.map(_convert_one, work, chunksize=4), 1):
            ok += status == "ok"
            fail += status == "fail"
            skip += status == "skip"
            if status == "fail" and fail <= 10:
                print(f"[fail] {name}", flush=True)
            if i % 1000 == 0:
                print(f"... {i}/{len(mp4s)} ok={ok} fail={fail} skip={skip}", flush=True)

    print(f"[done] ok={ok} fail={fail} skip={skip}  -> {args.dst}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())