# -*- coding: utf-8 -*-
"""
从拼接好的 24kHz 单声道全长音频中，用算法挑选
"信息密度最高、最清晰、最符合阿里云声音复刻要求"的 10~20s 片段。

评分维度（均归一化后加权）：
  1) speech_ratio  语音占比        —— 连续清晰人声覆盖率
  2) max_pause     最大停顿        —— 硬约束：<= 2.0s
  3) snr           信噪比          —— 语音帧能量 / 底噪能量
  4) info_density  信息密度        —— 音节数/秒（语速代理，即"承载信息量"）
  5) cv            能量调制系数    —— 语音起伏程度，音乐/平稳噪声低

输出：
  - best_segment.wav   (24kHz 16bit 单声道, 10~20s, <=10MB)
  - best_segment.mp3   (试听用 192k)
  - selection_report.json
"""
import numpy as np
import soundfile as sf
from scipy import ndimage
from scipy.signal import find_peaks
import json, os

SR = 24000
FRAME = int(0.025 * SR)   # 25ms
HOP   = int(0.010 * SR)   # 10ms
WIN_S = 15.0              # 候选窗口长度(秒)
STEP  = 0.5               # 滑窗步长(秒)
WORK  = os.path.dirname(os.path.abspath(__file__))
SRC   = os.path.join(WORK, "full_24k_mono.wav")

def load(path):
    x, sr = sf.read(path, dtype="float32")
    if sr != SR:
        raise SystemExit(f"unexpected sr {sr}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x

def frame_rms(x):
    n = len(x)
    nf = (n - FRAME) // HOP + 1
    out = np.empty(nf, dtype=np.float32)
    for i in range(nf):
        s = x[i * HOP: i * HOP + FRAME]
        out[i] = np.sqrt(np.mean(s * s) + 1e-12)
    return out

def main():
    x = load(SRC)
    total_s = len(x) / SR
    print(f"[1/5] loaded {total_s:.1f}s / {len(x)/SR/60:.1f}min")

    rms = frame_rms(x)
    nf = len(rms)
    t = np.arange(nf) * HOP / SR

    # ---- VAD：自适应能量阈值 ----
    noise_floor = float(np.percentile(rms, 15))
    thr = max(noise_floor * 4.0, 0.002)
    speech = rms > thr
    speech = ndimage.median_filter(speech.astype(np.uint8), size=3).astype(bool)
    speech = ndimage.binary_closing(speech, structure=np.ones(7))
    speech = ndimage.binary_opening(speech, structure=np.ones(3))
    print(f"[2/5] VAD thr={thr:.5f} noise_floor={noise_floor:.5f} "
          f"speech_ratio={speech.mean():.2%}")

    # ---- 语音帧特征 ----
    s_rms = rms[speech]
    snr_db = 10.0 * np.log10((s_rms ** 2) / (noise_floor ** 2) + 1e-9)
    mean_snr = float(snr_db.mean())
    cv = float(s_rms.std() / (s_rms.mean() + 1e-9))

    # 音节率（信息密度）：对 rms 包络找峰
    env = ndimage.gaussian_filter1d(rms, 3)
    peaks, _ = find_peaks(env, height=thr, distance=int(0.12 / (HOP / SR)))
    syl_per_s = float(len(peaks) / max(total_s, 1e-6))
    print(f"[3/5] mean_snr={mean_snr:.1f}dB cv={cv:.2f} syllables_total={len(peaks)} "
          f"rate={syl_per_s:.2f}/s")

    # ---- 滑窗评分 ----
    win_n = int(WIN_S / (HOP / SR))
    step_n = int(STEP / (HOP / SR))
    best = None
    for st in range(0, nf - win_n + 1, step_n):
        m = speech[st:st + win_n]
        sr_ratio = m.mean()
        # 最大连续停顿
        max_pause = 0.0
        cnt = 0
        for v in m:
            cnt = cnt + 1 if not v else 0
            if cnt > max_pause:
                max_pause = cnt
        max_pause_s = max_pause * HOP / SR
        if sr_ratio < 0.55 or max_pause_s > 2.0:
            continue
        w_rms = rms[st:st + win_n]
        w_env = env[st:st + win_n]
        w_peaks, _ = find_peaks(w_env, height=thr, distance=int(0.12 / (HOP / SR)))
        w_den = len(w_peaks) / WIN_S
        w_snr = 10.0 * np.log10(np.mean(w_rms[m] ** 2) / (noise_floor ** 2) + 1e-9)
        w_cv = float(w_rms[m].std() / (w_rms[m].mean() + 1e-9))
        score = (0.35 * min(w_den / 5.0, 1.0)
                 + 0.25 * min(w_snr / 40.0, 1.0)
                 + 0.20 * sr_ratio
                 + 0.20 * min(w_cv / 1.5, 1.0))
        if best is None or score > best["score"]:
            best = dict(start=st, end=st + win_n, score=float(score),
                        speech_ratio=float(sr_ratio), max_pause_s=max_pause_s,
                        snr_db=float(w_snr), info_density=w_den, cv=float(w_cv),
                        t_start=t[st], t_end=t[st + win_n])

    if best is None:
        raise SystemExit("no window passed constraints")

    # ---- 裁剪到语音范围并限制 10~20s ----
    st, en = best["start"], best["end"]
    idx = np.where(speech[st:en])[0]
    a = st + idx[0] - int(0.3 / (HOP / SR))
    b = st + idx[-1] + int(0.3 / (HOP / SR))
    a = max(a, 0)
    b = min(b, nf - 1)
    dur = (b - a) * HOP / SR
    if dur > 20.0:
        b = a + int(20.0 / (HOP / SR))
    if dur < 10.0:
        # 向后扩展；不够则向前
        ext = int((10.0 - dur) / (HOP / SR)) // 2
        a = max(0, a - ext)
        b = min(nf - 1, b + ext)
    a_s, b_s = a * HOP / SR, b * HOP / SR
    seg = x[int(a_s * SR): int(b_s * SR)]

    # ---- 导出 ----
    wav_path = os.path.join(WORK, "best_segment.wav")
    mp3_path = os.path.join(WORK, "best_segment.mp3")
    sf.write(wav_path, seg, SR, subtype="PCM_16")
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_path,
                    "-b:a", "192k", mp3_path], check=True)

    report = {
        "file": wav_path, "mp3": mp3_path,
        "duration_s": float(len(seg) / SR),
        "sample_rate": SR, "channels": 1, "bits": 16,
        "size_MB": round(os.path.getsize(wav_path) / 1e6, 3),
        "start_s": round(a_s, 2), "end_s": round(b_s, 2),
        "score": best["score"], "speech_ratio": best["speech_ratio"],
        "max_pause_s": best["max_pause_s"], "snr_db": best["snr_db"],
        "info_density_syl_per_s": best["info_density"],
        "modulation_cv": best["cv"],
        "global_mean_snr_db": round(mean_snr, 2),
        "global_cv": round(cv, 2),
    }
    with open(os.path.join(WORK, "selection_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[5/5] done -> {wav_path}  {len(seg)/SR:.1f}s  {report['size_MB']}MB")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
