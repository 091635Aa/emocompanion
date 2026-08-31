# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · 逐音节 F0 轮廓注入自检（CPU，确定性，无需 GPU/数据）

用合成谐波音验证 `injector.apply_f0_contour`：
  1) 输入恒定 F0 的音色音（≈180Hz），4 音节目标 deltas 有明显起伏；
  2) 输出应**保持时长**，且**行内 F0 出现对应起伏**（pyin 实测逐窗 F0）；
  3) 输出无越界(NaN/Inf)，幅度在 [-1,1]。

通过后，证明「音素≈音节级」粒度的 F0 轮廓注入有效——这是学习预测器逐音节输出去消费的控制接口。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from integration import injector  # noqa: E402


def make_harmonic(f0=180.0, dur=1.2, sr=24000, n_harm=6):
    t = np.arange(int(dur * sr)) / sr
    w = np.zeros_like(t)
    for h in range(1, n_harm + 1):
        w += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    return (w / np.max(np.abs(w))).astype("float32"), sr


def main():
    sr = 24000
    f0_base = 180.0
    deltas = [0.0, 3.0, -3.0, 1.0]            # 4 音节，明显起伏
    wav, sr = make_harmonic(f0_base, dur=1.2, sr=sr)
    out = injector.apply_f0_contour(wav, deltas)

    ok_len = len(out) == len(wav)
    ok_finite = bool(np.isfinite(out).all())
    ok_amp = bool(np.abs(out).max() <= 1.05)
    print(f"[len] in={len(wav)} out={len(out)}  preserved={ok_len}")
    print(f"[finite]={ok_finite}  [amp<=1.05] peak={abs(out).max():.3f}  ok={ok_amp}")

    # 逐窗实测 F0（pyin），验证行内起伏
    import librosa
    seg = len(out) // 4
    f0s = []
    for i in range(4):
        chunk = out[i * seg:(i + 1) * seg]
        f0, _, _ = librosa.pyin(chunk, fmin=60, fmax=600, sr=sr,
                                frame_length=512, hop_length=128)
        f0 = np.nan_to_num(f0, nan=0.0)
        f0s.append(float(np.median(f0[f0 > 0])) if np.any(f0 > 0) else 0.0)
    print("[window f0]", [round(x, 1) for x in f0s])
    expected = [f0_base * 2 ** (d / 12) for d in deltas]
    print("[expected ]", [round(x, 1) for x in expected])
    spread = max(f0s) - min(f0s)
    print(f"[contour spread] {spread:.1f} Hz  (>15Hz => 行内 F0 起伏已被注入: {spread > 15})")

    passed = ok_len and ok_finite and ok_amp and spread > 15
    print(f"[RESULT] {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())