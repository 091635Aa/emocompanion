# -*- coding: utf-8 -*-
"""R10 离线回归：Qwen3RapSynth 间接韵律注入器（无 librosa 路径的编排逻辑）
覆盖：_semitone_from_f0（半音换算，纯 numpy）
      inject()（拍对齐编排：gap 静音补齐、行拼接、能量夹限、尾静音补足）
约束：仅测"输入已在 SR + 时长匹配 + mean_f0=base"的免 librosa 路径，避免库依赖抖动。
运行：python3 /workspace/_regression/test_x_rap_injector.py
"""
import sys, os
import numpy as np

DIR = "/workspace/EmoCompanion_角色挂载与情感注入工程/07_说唱合成Qwen3RapSynth"
sys.path.insert(0, DIR)

from integration.injector import _semitone_from_f0, inject, SR  # noqa: E402
from prosody_model.rules import ProsodyPlan, LinePlan  # noqa: E402

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

print("== _semitone_from_f0（12 半音 = 1 个八度）==")
check("same_f0 → 0", abs(_semitone_from_f0(180, 180)) < 1e-9,
      f"got={_semitone_from_f0(180,180)}")
check("octave_up → 12", abs(_semitone_from_f0(360, 180) - 12) < 1e-9,
      f"got={_semitone_from_f0(360,180)}")
check("octave_down → -12", abs(_semitone_from_f0(90, 180) + 12) < 1e-9,
      f"got={_semitone_from_f0(90,180)}")

print("== inject 拍对齐编排（免 librosa 路径）==")
def _tone(sec):
    # 恒 S 波形（SR 采样率），时长=sec
    return np.full(int(sec * SR), 0.4, dtype="float32")

beat = 0.25
plan = ProsodyPlan(bpm=240, style="快嘴", beat_sec=beat)
plan.lines = [
    LinePlan(index=0, text="a", syllables=1, start_sec=0.0, duration_sec=0.5,
             mean_f0=180.0, f0_style="flat", energy=1.0, jump=1),
    LinePlan(index=1, text="b", syllables=1, start_sec=1.0, duration_sec=0.5,
             mean_f0=180.0, f0_style="flat", energy=1.0, jump=1),
]
neutral = [(_tone(0.5), SR), (_tone(0.5), SR)]  # 时长与 duration_sec 匹配
wav, onsets = inject(plan, neutral, base_f0=180.0, src_sr=SR)

check("onsets 长度=行数", len(onsets) == 2, f"got={len(onsets)}")
check("首 onset=0", abs(onsets[0]) < 1e-6, f"got={onsets[0]}")
check("二 onset 含 gap 静音", abs(onsets[1] - 1.0) < 1e-3, f"got={onsets[1]}")

# 期望总长 = 首行0.5 + gap0.5 + 二行0.5 + 尾部补到 plan.total_seconds(2*0.25*4=2.0)
# t 结束=1.5 => 补 0.5s 静音 => 总 2.0s
check("总时长=plan total", round(len(wav) / SR, 2) == 2.0, f"got={len(wav)/SR:.2f}s")
check("输出有限", np.isfinite(wav).all())

# 能量夹限：给 energy>1 的行，输出 ∈ [-1,1]
plan2 = ProsodyPlan(bpm=240, style="硬核", beat_sec=beat)
plan2.lines = [LinePlan(index=0, text="x", syllables=1, start_sec=0.0, duration_sec=0.5,
                        mean_f0=180.0, f0_style="punch", energy=5.0, jump=1)]
wav2, _ = inject(plan2, [(_tone(0.5), SR)], base_f0=180.0)
check("能量增益被夹限到 [-1,1]", float(np.abs(wav2).max()) <= 1.0, f"got={float(np.abs(wav2).max())}")

print("== inject 空行缺失 ==")
plan3 = ProsodyPlan(bpm=240, style="快嘴", beat_sec=beat)
plan3.lines = [LinePlan(index=0, text="a", syllables=1, start_sec=0.0, duration_sec=0.5,
                        mean_f0=180.0, f0_style="flat", energy=1.0, jump=1)]
# neutral_lines 行数不足 → 该行跳过但 onset 仍记录
wav3, ons3 = inject(plan3, [], base_f0=180.0)
check("缺行时 onset 仍记录", len(ons3) == 1, f"got={len(ons3)}")
check("缺行输出为纯静音", float(np.abs(wav3).max()) < 1e-6)

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)