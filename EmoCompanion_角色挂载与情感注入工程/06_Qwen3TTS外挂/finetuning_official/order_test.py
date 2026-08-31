# coding=utf-8
"""验证 import 顺序：先 qwen_tts 再 peft/accelerate，是否能避免卡死。"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
def step(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

step("import qwen_tts FIRST")
t=time.time()
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
step(f"qwen_tts import in {time.time()-t:.1f}s")

step("import peft")
t=time.time()
from peft import LoraConfig, get_peft_model
step(f"peft import in {time.time()-t:.1f}s")

step("import accelerate")
t=time.time()
from accelerate import Accelerator
step(f"accelerate import in {time.time()-t:.1f}s")

step("DONE OK")