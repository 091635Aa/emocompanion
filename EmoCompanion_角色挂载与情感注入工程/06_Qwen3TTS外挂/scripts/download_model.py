# -*- coding: utf-8 -*-
"""Qwen3-TTS 可训练权重下载（tokenizer + Base）。"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out_done", default="")
    args = ap.parse_args()
    os.environ["MODELSCOPE_CACHE"] = args.cache
    os.environ["HF_HOME"] = os.path.join(os.path.dirname(args.cache), "hf_cache")
    from modelscope import snapshot_download
    p = snapshot_download(args.model, cache_dir=args.cache)
    print("DONE:", p, flush=True)
    if args.out_done:
        with open(args.out_done, "w", encoding="utf-8") as f:
            f.write(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())