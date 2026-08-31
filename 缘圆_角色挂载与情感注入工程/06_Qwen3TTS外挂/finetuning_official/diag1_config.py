# coding=utf-8
"""分层诊断 Qwen3TTS 加载卡点：config / 结构构建 / 权重加载 分别计时。"""
import os, sys, time, signal
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoConfig

BASE = r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\master"

def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(BASE)
    step(f"[1] AutoConfig.from_pretrained OK in {time.time()-t0:.1f}s model_type={cfg.model_type}")
    sys.exit(0)

if __name__ == "__main__":
    main()