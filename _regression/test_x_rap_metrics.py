# -*- coding: utf-8 -*-
"""R9 离线回归：Qwen3RapSynth 评价指标纯逻辑评估指标（无需 librosa/GPU）
覆盖：beat_alignment_error / bpm_drift / rhyme_hit / energy_contrast / _end_char
运行：python3 /workspace/_regression/test_x_rap_metrics.py
"""
import sys, os
import numpy as np

DIR = "/workspace/EmoCompanion_角色挂载与情感注入工程/07_说唱合成Qwen3RapSynth"
sys.path.insert(0, DIR)

from eval.metrics import (beat_alignment_error, bpm_drift, rhyme_hit,  # noqa: E402
                          energy_contrast, _end_char)

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

print("== beat_alignment_error ==")
# 拍=0.5s，onset 全落网格 → 误差 0
check("on-beat 误差=0", beat_alignment_error([0.5, 1.0, 1.5], 0.5) < 1e-6,
      f"got={beat_alignment_error([0.5,1.0,1.5], 0.5)}")
# 空 onset → 0
check("空 onset=0", beat_alignment_error([], 0.5) == 0.0)
# 偏拍 0.25 → 误差=0.25
check("半拍偏移误差=0.25", abs(beat_alignment_error([0.25], 0.5) - 0.25) < 1e-6,
      f"got={beat_alignment_error([0.25], 0.5)}")

print("== bpm_drift ==")
# 测到 60 BPM（间隔 1s），期望 60 → drift 0
check("bpm 匹配 drift=0", abs(bpm_drift([0, 1, 2, 3], 60)) < 1e-6, f"got={bpm_drift([0,1,2,3],60)}")
# 期望 120，实测 60 → drift=|60-120|/120=0.5
check("bpm 双倍 drift=0.5", abs(bpm_drift([0, 1, 2, 3], 120) - 0.5) < 1e-6,
      f"got={bpm_drift([0,1,2,3],120)}")
check("仅一点 drift=0", bpm_drift([5.0], 60) == 0.0)

print("== rhyme_hit / _end_char ==")
check("_end_char 取末汉字", _end_char("押韵吧") == "吧", f"got={_end_char('押韵吧')!r}")
check("空文本", _end_char("") == "")
# 相邻尾字相同：ends=[苍,苍,方]，命中1/2=0.5
check("押韵命中 0.5", abs(rhyme_hit(["天空苍苍", "大地苍苍", "走向远方"]) - 0.5) < 1e-9,
      f"got={rhyme_hit(['天空苍苍','大地苍苍','走向远方'])}")
check("全押韵=1", rhyme_hit(["一样", "这样", "那样"]) >= 0.9,
      f"got={rhyme_hit(['一样','这样','那样'])}")
check("单行=0", rhyme_hit(["只有一行"]) == 0.0)

print("== energy_contrast（合成 wav：重音处能量高）==")
# 构造 24k 采样；拍 hop=256，SR=24000
SR = 24000
hop = 256
# 前 20 帧安静，之后 20 帧响亮；onsets 落在响亮区
静_n = hop * 20
响_n = hop * 20
wav = np.concatenate([np.zeros(静_n, dtype="float32"),
                      np.ones(响_n, dtype="float32") * 0.5,])
# onset 放响亮区起点（时间 = 20*hop/SR）
t_on = 20 * hop / SR
c = energy_contrast(wav, [t_on], beat_sec=0.5)
check("重音段能量对比>1", c > 1.0, f"got={c:.3f}")
c2 = energy_contrast(wav, [], 0.5)
check("空 onsets=0", c2 == 0.0)

# ---- librosa 依赖路径（R11 已安装 librosa 1.0.0）----
import librosa  # noqa: E402
from eval.metrics import onset_beat_error, f0_tracking  # noqa: E402
print("== f0_tracking / onset_beat_error（librosa 入集）==")
# onset_beat_error：脉冲 wav 落在拍网格
beat0 = 0.5
# 250ms 整拍前生成 8 个等间隔脉冲（间隔=beat0）
tones = []
hop0 = 256
n = 8
for k in range(n):
    t0 = int((k * beat0) / (hop0 / 24000))
    ones = np.zeros(hop0, dtype="float32"); ones[:80] = 1.0
    tones.append(ones)
pulses = np.concatenate(tones)
err = onset_beat_error(pulses.astype("float32"), beat0)
check("on-beat 脉冲对拍误差小", err < 0.25, f"got={err:.3f}")
# f0_tracking 用纯音 + 匹配的 LinePlan 目标
from prosody_model.rules import LinePlan  # noqa: E402
sr0 = 24000
freq = 220.0
t_ = np.arange(0, int(sr0 * 0.5), dtype="float32") / sr0
tone = (0.5 * np.sin(2 * np.pi * freq * t_)).astype("float32")
lp = LinePlan(index=0, text="x", syllables=1, start_sec=0.0, duration_sec=0.5,
              mean_f0=freq, f0_style="flat", energy=1.0, jump=1)
rmse = f0_tracking(tone, [lp], base_f0=freq)
check("纯音 F0 RMSE 小", rmse < 50.0, f"got={rmse:.1f}")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)