# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · 生成质量客观指标

- `beat_alignment_error(wav, onsets, beat_sec)`: 逐行 onset 与最近节拍/句拍的绝对误差均值
- `bpm_drift(wav, expected_bpm)`: 检测到的节拍周期相对期望 BPM 的漂移
- `f0_tracking(wav, plan)`: 用 librosa.pyin 采实测 F0，比对规则轮廓的均方根误差(RMSE)
- `rhyme_hit(texts)`: 简单尾音押韵命中率（末字求同，中文近音近似）
- `energy_contrast(wav, segments)`: 重音段/轻音段的能量对比（说唱起伏度代理）

所有函数对输入的 wav（float32, sr=24000）只读，便于批量化评估本方案 vs 基线。
"""
import numpy as np

SR = 24000


def beat_alignment_error(onsets_sec, beat_sec) -> float:
    """逐行 onset 相对最近节拍网格的均值绝对误差（秒）。小=对拍准。"""
    if not onsets_sec:
        return 0.0
    errs = []
    for o in onsets_sec:
        b = max(beat_sec, 1e-6)
        nearest = round(o / b) * b
        errs.append(abs(o - nearest))
    return float(np.mean(errs))


def bpm_drift(onsets_sec, expected_bpm) -> float:
    """实测行间隔与期望节拍周期的相对漂移（|估计BPM-期望|/期望）。"""
    if lens := len(onsets_sec) - 1:
        diffs = np.diff(onsets_sec)
        if diffs.min() > 1e-3:
            est_bpm = 60.0 / np.mean(diffs[diffs > 1e-3])
            return float(abs(est_bpm - expected_bpm) / expected_bpm)
    return 0.0


def f0_tracking(wav, plan_lines, base_f0=180.0) -> float:
    """实测 F0 与规则目标 F0 的 RMSE（Hz）。"""
    import librosa
    f0, _, _ = librosa.pyin(wav.astype("float32"), fmin=60, fmax=500,
                            sr=SR, frame_length=1024, hop_length=256)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return 0.0
    # 行级期望 F0：按行时长摊开
    target = []
    for lp in plan_lines:
        target += [lp.mean_f0] * max(int(lp.duration_sec * SR / 256), 1)
    if not target:
        return 0.0
    ta = np.asarray(target[: voiced.size], dtype="float64")
    if ta.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((voiced[: ta.size] - ta) ** 2)))


def _end_char(text: str) -> str:
    import re
    m = re.findall(r"[\u4e00-\u9fff]", text)
    return m[-1] if m else text[-1:] if text else ""


def rhyme_hit(texts) -> float:
    """相邻行尾字相同的押韵命中率。"""
    ends = [_end_char(t) for t in texts]
    if len(ends) < 2:
        return 0.0
    hit = sum(1 for a, b in zip(ends, ends[1:]) if a == b)
    return float(hit / (len(ends) - 1))


def energy_contrast(wav, onsets_sec, beat_sec) -> float:
    """重音段相对平均能量的对比（起伏度代理，越大越有说唱冲击）。"""
    if not onsets_sec:
        return 0.0
    hop = 256
    # 帧能量
    w = np.abs(wav.astype("float32"))
    seg = max(int(hop * 0.5), 1)
    frames = np.array([w[i * hop: i * hop + seg].mean() for i in range(int(len(w) / hop))])
    out = []
    for o in onsets_sec:
        idx = int(o * SR / hop)
        if 0 <= idx < len(frames):
            out.append(frames[idx])
    if not out:
        return 0.0
    overall = frames.mean() + 1e-6
    return float(np.mean(out) / overall)


def onset_beat_error(wav, beat_sec) -> float:
    """从音频**实际检测** onset（librosa.onset_detect）再与拍网格对齐，
    消除「规划 onset」口径伪影，使 baseline/ours 公平可比。返回均值绝对误差秒。"""
    import librosa
    ons = librosa.onset.onset_detect(y=wav.astype("float32"), sr=SR, hop_length=256,
                                     backtrack=True)
    if ons.size == 0:
        return 0.0
    t = ons * 256.0 / SR
    b = max(beat_sec, 1e-6)
    errs = np.abs((np.round(t / b) * b) - t)
    return float(np.mean(errs))


def summarize(plan, wav, onsets, texts) -> dict:
    """一键汇总客观指标（供对比输出）。"""
    return {
        "beat_alignment_error_s": round(beat_alignment_error(onsets, plan.beat_sec), 4),
        "bpm_drift": round(bpm_drift(onsets, plan.bpm), 4),
        "f0_rmse_hz": round(f0_tracking(wav, plan.lines), 2),
        "rhyme_hit": round(rhyme_hit(texts), 3),
        "energy_contrast": round(energy_contrast(wav, onsets, plan.beat_sec), 3),
    }