# coding=utf-8
"""用 faulthandler 定位 import qwen_tts 卡住的准确位置。"""
import os, sys, time, threading, faulthandler
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def dump():
    print("\n===== DUMP after 30s =====", flush=True)
    faulthandler.dump_traceback()

timer = threading.Timer(30, dump)
timer.start()
t0 = time.time()
print("start import", flush=True)
import torch
print(f"torch {time.time()-t0:.1f}s", flush=True)
import librosa
print(f"librosa {time.time()-t0:.1f}s", flush=True)
from transformers import AutoConfig
print(f"transformers.AutoConfig {time.time()-t0:.1f}s", flush=True)
timer.cancel()
print("STDLIB+librosa OK", flush=True)