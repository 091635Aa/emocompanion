# coding=utf-8
"""快速验证 Qwen3TTS Base 能否在 CUDA 上加载，并打印可训练参数统计。"""
import os, sys, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from transformers import AutoConfig

BASE = r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\master"

def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    print("cuda avail:", torch.cuda.is_available(), flush=True)

    step("loading model…")
    t0 = time.time()
    qwen3tts = Qwen3TTSModel.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )
    step(f"model loaded in {time.time()-t0:.1f}s")
    dev = next(qwen3tts.model.parameters()).device
    print("model device:", dev, flush=True)
    n_params = sum(p.numel() for p in qwen3tts.model.parameters())
    print("total params:", n_params, flush=True)
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"CUDA mem used: {mem:.2f} GiB", flush=True)
    print("OK", flush=True)

if __name__ == "__main__":
    main()