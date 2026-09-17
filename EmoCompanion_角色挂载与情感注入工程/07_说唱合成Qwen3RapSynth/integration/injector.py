# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · 间接韵律注入（后处理对拍控制）

因公开 Base 不暴露声学参数，本模块把 ProsodyPlan 的 F0/时长/能量约束落到
**采样级后处理**（librosa time_stretch + pitch_shift）+ **拍对齐编排**（rest/拼接）：

  1) 每行先中性合成（好音质、音色稳定）；
  2) `time_stretch` 把该行时长缩放到 `LinePlan.duration_sec`（快嘴压缩、旋律舒展）；
  3) `pitch_shift` 按 `mean_f0` 相对默认音高的半音差升降（旋律拱形、硬核下探）；
  4) 依 `start_sec` 把各行放到拍网格，前置/句间休止由静音补齐，得到全局干声 + 逐行 onset。

输出最终 float32 wav（sr = 24000）与逐行 onset 秒表，供评估脚本做对拍对齐测量。
"""
import numpy as np

SR = 24000


def _semitone_from_f0(mean_f0: float, base_f0: float) -> float:
    return float(12.0 * np.log2(max(mean_f0, 1e-3) / max(base_f0, 1e-3)))


def _resample_mono(wav: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == SR:
        return wav.astype("float32")
    import librosa
    return librosa.resample(wav.astype("float32"), orig_sr=src_sr, target_sr=SR)


def _apply_duration(wav: np.ndarray, dur: float) -> np.ndarray:
    import librosa
    cur = wav.shape[0] / SR
    if cur <= 1e-3:
        return wav
    rate = cur / max(dur, 1e-3)
    rate = float(np.clip(rate, 0.7, 1.5))  # 限制拉伸幅度防音质崩坏
    if abs(rate - 1.0) < 1e-3:
        return wav
    return librosa.effects.time_stretch(wav, rate=rate)


def _apply_pitch(wav: np.ndarray, semitones: float) -> np.ndarray:
    import librosa
    if abs(semitones) < 0.5:
        return wav
    return librosa.effects.pitch_shift(wav, sr=SR, n_steps=float(semitones))


# ---------------- 逐音节 F0 轮廓注入（音素≈音节，Chinese） ----------------
def apply_f0_contour(wav: np.ndarray, deltas: list,
                     crossfade_sec: float = 0.02) -> np.ndarray:
    """把 wav 按音节数分窗，逐窗按各自的半音差做 time-varying pitch，段缘淡变防爆音。

    相比行级单次 pitch_shift，这能在**行内**还原 F0 起伏（快嘴疏密 / 旋律拱形 / 硬核下探），
    是「音素级 vs 音节级」粒度的注入实现，供学习预测器输出的逐音节 deltas 消费。
    """
    import librosa
    if not deltas or len(deltas) < 2:
        return wav
    n = len(deltas)
    total = wav.shape[0]
    seg = max(total // n, 1)
    cf = max(int(crossfade_sec * SR), seg // 4 or 1)
    out = wav.astype("float32").copy()
    edge_in = np.linspace(0.0, 1.0, cf, dtype="float32")
    edge_out = np.linspace(1.0, 0.0, cf, dtype="float32")
    for i in range(n):
        s = i * seg
        e = total if i == n - 1 else min(s + seg, total)
        if e <= s:
            continue
        d = float(deltas[min(i, len(deltas) - 1)])
        sw = out[s:e]
        if abs(d) >= 0.5:
            sw = librosa.effects.pitch_shift(sw, sr=SR, n_steps=d)
        # 段缘淡变（避免逐音节变速的接缝爆音）
        c1 = min(cf, (e - s) // 2)
        if c1 > 0:
            sw[:c1] = (sw[:c1] * edge_in[:c1]).astype("float32")
            sw[-c1:] = (sw[-c1:] * edge_out[:c1]).astype("float32")
        out[s:e] = sw
    return out.astype("float32")


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype="float32")


def inject(plan, neutral_lines, base_f0: float = 180.0,
           src_sr: int = SR) -> tuple:
    """把逐行中性合成 `neutral_lines`（list[(wav,sr)]）按 ProsodyPlan 编排为干声。

    参数:
        plan: prosody_model.rules.ProsodyPlan
        neutral_lines: 与 plan.lines 等长的 (wav, sr) 列表，长度不足时跳过该行
        base_f0: 中性合成默认音高基准（Hz），用于半音差计算
    返回:
        (final_wav, onsets)  onsets 为逐行起始秒数列表
    """
    import librosa
    pieces: list = []
    onsets: list = []
    t = 0.0
    plan_dur = plan.total_seconds()
    for i, lp in enumerate(plan.lines):
        # 当前行起始不早于计划拍（用静音补齐空拍）
        gap = lp.start_sec - t
        if gap > 1e-4:
            pieces.append(_silence(gap))
            t += gap
        onsets.append(t)
        if i < len(neutral_lines) and neutral_lines[i] is not None:
            raw, sr = neutral_lines[i]
            wav = _resample_mono(raw, sr)
            wav = _apply_duration(wav, lp.duration_sec)
            # 逐音节 F0 轮廓优先（学习模型提供）、否则行级单次升降
            if getattr(lp, "syllable_f0_delta", None):
                wav = apply_f0_contour(wav, lp.syllable_f0_delta)
            else:
                wav = _apply_pitch(wav, _semitone_from_f0(lp.mean_f0, base_f0))
            wav = wav * float(lp.energy)          # 能量系数（线性增益，含剪裁夹限）
            wav = np.clip(wav, -1.0, 1.0)
            pieces.append(wav)
            t += wav.shape[0] / SR
        # 行间休止：由计划里 start_sec 的下一行跳进在循环顶部补齐
    # 尾部不足 plan 时长则补静音（供对拍测量稳定窗口）
    if t < plan_dur - 1e-4:
        pieces.append(_silence(plan_dur - t))
    if not pieces:
        return _silence(plan_dur), onsets
    return np.concatenate(pieces).astype("float32"), onsets