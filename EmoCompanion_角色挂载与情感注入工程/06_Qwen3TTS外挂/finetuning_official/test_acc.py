# coding=utf-8
"""孤立测试 Accelerator 创建是否卡死。"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
step("import torch")
import torch
step("import accelerate")
from accelerate import Accelerator
step("create Accelerator(log_with=tensorboard)")
t0=time.time()
acc = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")
step(f"Accelerator created in {time.time()-t0:.1f}s device={acc.device}")
step("DONE OK")