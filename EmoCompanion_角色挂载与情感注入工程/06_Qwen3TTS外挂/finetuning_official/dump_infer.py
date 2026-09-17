# coding=utf-8
"""绕过 qwen_tts 顶层 __init__，直接加载 inference 模块，测量耗时。"""
import os, sys, time, threading, faulthandler
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def dump():
    print("\n===== DUMP =====", flush=True)
    faulthandler.dump_traceback()
timer = threading.Timer(40, dump)
timer.start()
t0 = time.time()
print("start", flush=True)
# 直接导入 qwen_tts 包内不 import __init__ 很难，因为 Python 导入子模块会先执行包 __init__。
# 但 *.__init__ 才是拉 tokenizer 的地方。先看 inference 单独导入
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
print(f"qwen3_tts_model {time.time()-t0:.1f}s", flush=True)
timer.cancel()
print("DONE OK", flush=True)