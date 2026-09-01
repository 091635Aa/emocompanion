# -*- coding: utf-8 -*-
"""R11 离线回归：Qwen3RapSynth TTS 基座封装纯逻辑（无需模型/torch/GPU）
覆盖：EMOTION_VOCAB 情感标签映射、缺件校验 _check（TTSUnavailable 结构化报错）。
运行：python3 /workspace/_regression/test_x_rapsynth.py
"""
import sys, os, tempfile

DIR = "/workspace/EmoCompanion_角色挂载与情感注入工程/07_说唱合成Qwen3RapSynth"
sys.path.insert(0, DIR)

from tts.synthesizer import EMOTION_VOCAB, EMOTION_FALLBACK, RaSynthCore, TTSUnavailable  # noqa: E402

PASS = 0
FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")

print("== EMOTION_VOCAB 情感标签映射 ==")
check("含核心情感", {"开心", "悲伤", "俏皮", "平静", "兴奋", "硬核"} <= set(EMOTION_VOCAB) or {"开心", "悲伤"} <= set(EMOTION_VOCAB))
快乐 = EMOTION_VOCAB.get("开心")
check("开心映射为 [emotion] 包裹", isinstance(快乐, str) and "[emotion]开心[/emotion]" in 快乐, f"got={快乐!r}")
check("未知情感回退默认存在", EMOTION_FALLBACK in EMOTION_VOCAB, f"fallback={EMOTION_FALLBACK}")

print("== 缺件校验（结构化检查，不加载模型）==")
with tempfile.TemporaryDirectory() as d:
    core = RaSynthCore(
        base_model=os.path.join(d, "base"),
        adapter_dirs={"voice": os.path.join(d, "voice"), "emotion": os.path.join(d, "emotion")},
        speaker_emb=os.path.join(d, "emb.pt"),
    )
    try:
        core._check()
        check("缺件时抛 TTSUnavailable", False, "未抛异常（不应走到）")
    except TTSUnavailable as e:
        msg = str(e)
        check("缺件时抛 TTSUnavailable", "缺少 TTS 组件" in msg and "Base" in msg, f"msg={msg[:80]!r}")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)