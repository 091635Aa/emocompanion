# -*- coding: utf-8 -*-
"""实证：llama-tts 整段一次合成 + 内嵌情感/语速指令 + 固定 seed + GPU 最大化。"""
import os, sys, time, wave, subprocess, tempfile, io
import numpy as np

LLAMA = r"D:\AI情感\pykits\llama-cpp-bin\llama-tts.exe"
GGUF = r"D:\AI情感\pykits\models\Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"
MMPROJ = r"D:\AI情感\pykits\models\mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
REF = r"D:\ACQ富\wav_24k\100_116_117_12_001.wav"
VOICE = r"D:\AI情感\pykits\models\voice_lora_qwen3tts.gguf"
EMO = r"D:\AI情感\pykits\models\emotion_lora_qwen3tts.gguf"

# 整段带指令（模型原生演绎情感/语速曲线）
TEXT = ("[开心]哥哥你终于回来啦，我好想你呀！（语速正常）"
        "[温柔]今天辛苦啦，我陪你慢慢聊（语速放慢）。"
        "[兴奋]你看我给你准备了什么惊喜！超开心的（语速加快）")

def synth(text, out, seed="0", extra=()):
    tmp = tempfile.mkdtemp(prefix="probe_")
    outwav = os.path.join(tmp, "o.wav")
    cmd = [LLAMA, "-m", GGUF, "-mm", MMPROJ, "-p", text, "--tts-lang", "zh",
           "--tts-speaker-file", REF, "--lora", f"{VOICE},{EMO}",
           "--seed", str(seed), "-o", outwav, "-ngl", "99"]
    cmd += list(extra)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=os.path.dirname(LLAMA), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=600)
    wall = time.perf_counter() - t0
    if r.returncode != 0 or not os.path.isfile(outwav):
        print(f"FAIL code={r.returncode}"); return
    with wave.open(outwav, "rb") as w:
        sr = w.getframerate(); n = w.getnframes(); raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32767.0
    if a.ndim > 1: a = a.mean(axis=1)
    dur = len(a) / sr
    # 检测时长（语速是否原生）
    import shutil
    data = open(outwav, "rb").read()
    open(out, "wb").write(data)
    print(f"OK seed={seed} dur={dur:.2f}s wall={wall:.2f}s RTF={wall/dur:.3f} sr={sr} {os.path.getsize(out)}B extra={extra}")

if __name__ == "__main__":
    os.makedirs(r"d:\AI情感\缘圆_角色挂载与情感注入工程\_probe_out", exist_ok=True)
    base = r"d:\AI情感\缘圆_角色挂载与情感注入工程\_probe_out"
    print("== 整段一次 + 内嵌指令 + 固定seed=0 (no extra) ==")
    synth(TEXT, os.path.join(base, "full_s0.wav"), seed="0")
    print("== 相同 seed=0 复现测试（音色/语速应一致）==")
    synth(TEXT, os.path.join(base, "full_s0b.wav"), seed="0")
    print("== 加 flash-attn + threads 16 + batch 512 (GPU/CPU 并行) ==")
    synth(TEXT, os.path.join(base, "full_s0_fa.wav"), seed="0",
          extra=["-fa", "on", "-t", "16", "-b", "512", "-ub", "256", "--mlock"])
    print("== 仅 ASCII 对比：seed 可复现性 ==")
    synth("[平静]测试一下，今天天气不错。", os.path.join(base, "t_s0.wav"), seed="0")
    synth("[平静]测试一下，今天天气不错。", os.path.join(base, "t_s1.wav"), seed="1")