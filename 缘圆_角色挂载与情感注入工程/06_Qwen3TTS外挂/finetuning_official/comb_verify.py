# coding=utf-8
"""复现 sft_lora 的 import + Accelerator + from_pretrained，定位 GPU 不占用的卡点。"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

step("import torch...")
import torch
step("import accelerate...")
from accelerate import Accelerator
step("import peft...")
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
step("import qwen_tts model...")
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from transformers import AutoConfig

BASE = r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\master"

step("create Accelerator(log_with=tensorboard)...")
t0=time.time()
os.makedirs("out/_verify_logs", exist_ok=True)
acc = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16",
                  log_with="tensorboard", logging_dir="out/_verify_logs")
step(f"Accelerator OK in {time.time()-t0:.1f}s device={acc.device}")

step("from_pretrained (cuda bf16)...")
t0=time.time()
qwen3tts = Qwen3TTSModel.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cuda",
    local_files_only=True, trust_remote_code=True)
step(f"from_pretrained OK in {time.time()-t0:.1f}s")
dev = next(qwen3tts.model.parameters()).device
print("device:", dev, flush=True)
print("cuda alloc GiB:", round(torch.cuda.memory_allocated()/1024**3,2), flush=True)

step("build_lora_model...")
t0=time.time()
model = get_peft_model(qwen3tts.model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"))
step(f"lora build OK in {time.time()-t0:.1f}s")
model.print_trainable_parameters()
step("ALL DONE OK")