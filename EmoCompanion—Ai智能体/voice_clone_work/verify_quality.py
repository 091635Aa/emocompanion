# -*- coding: utf-8 -*-
"""精确诊断 v2：修正最大停顿计算，直接比对当前已选片段 vs 全域候选。"""
import numpy as np, soundfile as sf, os
from scipy import ndimage
from scipy.signal import find_peaks

SR = 24000
FRAME, HOP = int(0.025 * SR), int(0.010 * SR)
WORK = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(WORK, "full_24k_mono.wav")

def load(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr

def frame_rms(x, sr):
    fr, hp = int(0.025 * sr), int(0.010 * sr)
    nf = (len(x) - fr) // hp + 1
    out = np.empty(nf, dtype=np.float32)
    for i in range(nf):
        s = x[i * hp: i * hp + fr]
        out[i] = np.sqrt(np.mean(s * s) + 1e-12)
    return out

def spectral_feats(x, sr):
    fr = int(0.025 * sr)
    hp = int(0.010 * sr)
    nf = (len(x) - fr) // hp + 1
    band_r = np.empty(nf, dtype=np.float32)
    flat = np.empty(nf, dtype=np.float32)
    win = np.hanning(fr)
    freqs = np.fft.rfftfreq(fr, 1 / sr)
    m_band = (freqs >= 300) & (freqs <= 3400)
    for i in range(nf):
        seg = x[i * hp: i * hp + fr] * win
        mag = np.abs(np.fft.rfft(seg)) + 1e-10
        tot = mag.sum()
        band_r[i] = mag[m_band].sum() / tot
        g = np.exp(np.log(mag).mean())
        a = mag.mean()
        flat[i] = g / a if a > 0 else 0.0
    return band_r, flat

def max_pause_msk(m):
    """正确计算最大连续停顿（帧数）"""
    mp = 0; cur = 0
    for v in m:
        cur = cur + 1 if not v else 0
        if cur > mp:
            mp = cur
    return mp

def segment_metrics(x, sr, a_s, b_s):
    seg = x[int(a_s * sr): int(b_s * sr)]
    rms = frame_rms(seg, sr)
    band_r, flat = spectral_feats(seg, sr)
    noise_floor = float(np.percentile(rms, 15))
    thr = max(noise_floor * 4.0, 0.002)
    speech = rms > thr
    speech = ndimage.median_filter(speech.astype(np.uint8), size=3).astype(bool)
    speech = ndimage.binary_closing(speech, structure=np.ones(7))
    speech = ndimage.binary_opening(speech, structure=np.ones(3))
    env = ndimage.gaussian_filter1d(rms, 3)
    peaks, _ = find_peaks(env, height=thr, distance=int(0.12 / (HOP / SR)))
    return {
        "dur": len(seg) / sr,
        "speech_ratio": float(speech.mean()),
        "max_pause_s": max_pause_msk(speech) * HOP / sr,
        "band_ratio": float(band_r[speech].mean()),
        "flatness": float(flat[speech].mean()),
        "snr_db": float(10 * np.log10(np.mean(rms[speech] ** 2) / (noise_floor ** 2) + 1e-9)),
        "density": float(len(peaks) / max(len(seg) / sr, 1e-6)),
        "cv": float(rms[speech].std() / (rms[speech].mean() + 1e-9)),
    }

x, sr = load(SRC)

print("==== 当前已选片段 vs 新Top候选区 ====")
targets = [
    ("当前已选 [1524.77,1540.29]", 1524.77, 1540.29),
    ("候选区1 [1560,1575]", 1560.0, 1575.0),
    ("候选区2 [1563,1575]", 1563.0, 1575.0),
    ("邻近对照 [1530,1545]", 1530.0, 1545.0),
    ("邻近对照 [1515,1530]", 1515.0, 1530.0),
]
for name, a, b in targets:
    m = segment_metrics(x, sr, a, b)
    gates = []
    if m["speech_ratio"] < 0.60: gates.append("语音占比<0.60")
    if m["max_pause_s"] > 2.0: gates.append("停顿>2s")
    if m["band_ratio"] < 0.60: gates.append("频带占比<0.60")
    if m["flatness"] < 0.04: gates.append("平坦度<0.04")
    print(f"{name}: speech={m['speech_ratio']:.2%} pause={m['max_pause_s']:.2f}s "
          f"band={m['band_ratio']:.2f} flat={m['flatness']:.3f} "
          f"snr={m['snr_db']:.1f}dB density={m['density']:.2f}/s cv={m['cv']:.2f}")
    print(f"   门禁判定: {'全部通过' if not gates else '未过 -> ' + ','.join(gates)}")

# 全域重排（修正停顿）: 10~20s 窗
rms = frame_rms(x, sr)
band_r, flat = spectral_feats(x, sr)
nf = len(rms)
noise_floor = float(np.percentile(rms, 15))
thr = max(noise_floor * 4.0, 0.002)
speech = rms > thr
speech = ndimage.median_filter(speech.astype(np.uint8), size=3).astype(bool)
speech = ndimage.binary_closing(speech, structure=np.ones(7))
speech = ndimage.binary_opening(speech, structure=np.ones(3))
env = ndimage.gaussian_filter1d(rms, 3)

cands = []
for dur_s in (10.0, 12.0, 15.0, 18.0, 20.0):
    win_n = int(dur_s / (HOP / SR))
    for st in range(0, nf - win_n + 1, 50):  # 0.5s
        m = speech[st:st + win_n]
        sr_ratio = m.mean()
        if sr_ratio < 0.60:
            continue
        if max_pause_msk(m) * HOP / SR > 2.0:
            continue
        w_band = band_r[st:st + win_n][m].mean()
        if w_band < 0.60:
            continue
        w_flat = flat[st:st + win_n][m].mean()
        if w_flat < 0.04:
            continue
        w_rms = rms[st:st + win_n]
        w_peaks, _ = find_peaks(env[st:st + win_n], height=thr,
                                distance=int(0.12 / (HOP / SR)))
        w_den = len(w_peaks) / dur_s
        w_snr = 10 * np.log10(np.mean(w_rms[m] ** 2) / (noise_floor ** 2) + 1e-9)
        w_cv = float(w_rms[m].std() / (w_rms[m].mean() + 1e-9))
        score = (0.30 * min(w_den / 5.0, 1) + 0.25 * min(w_snr / 40.0, 1)
                 + 0.15 * sr_ratio + 0.15 * min(w_cv / 1.5, 1)
                 + 0.15 * min(w_flat / 0.15, 1))
        cands.append(dict(dur=dur_s, st=st, score=score, speech_ratio=sr_ratio,
                          pause=max_pause_msk(m) * HOP / SR, band=w_band,
                          flat=w_flat, snr=w_snr, density=w_den, cv=w_cv))
cands.sort(key=lambda c: -c["score"])
print(f"\n==== 全域重排（修正停顿后）过门禁候选数 = {len(cands)} ====")
for i, c in enumerate(cands[:10], 1):
    a_s, b_s = c["st"] * HOP / SR, (c["st"] + int(c["dur"] / (HOP / SR))) * HOP / SR
    flag = " <== 包含当前已选区域" if abs(a_s - 1524.77) < 5 else ""
    print(f"{i}. score={c['score']:.3f} dur={c['dur']:.0f}s t=[{a_s:.1f},{b_s:.1f}] "
          f"speech={c['speech_ratio']:.2%} pause={c['pause']:.2f}s "
          f"band={c['band']:.2f} flat={c['flat']:.3f} snr={c['snr']:.1f}dB "
          f"density={c['density']:.2f}/s{flag}")

# TTS 输出核查（用原生采样率）
if os.path.exists(os.path.join(WORK, "tts_output.wav")):
    ty, tsr = load(os.path.join(WORK, "tts_output.wav"))
    tm = segment_metrics(ty, tsr, 0, len(ty) / tsr)
    print(f"\n==== TTS 输出核查（原生 {tsr}Hz）====")
    print(f"dur={tm['dur']:.2f}s speech={tm['speech_ratio']:.2%} "
          f"pause={tm['max_pause_s']:.2f}s snr={tm['snr_db']:.1f}dB "
          f"peak={np.abs(ty).max():.3f} clip={np.mean(np.abs(ty) >= 0.99):.4%}")
