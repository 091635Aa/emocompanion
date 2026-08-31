# -*- coding: utf-8 -*-
"""
style_prosody —— 说话风格（StylePlug）韵律评估器
=====================================================================
从 llama-tts 合成音频里提取可测量的韵律特征，用来量化"说话风格到底有没有
被控制住"（用户要的 ~95% 说话风格控制度）：
  - pitch_std     : 帧级基频(F0)标准差(Hz)   -> 语调起伏(≠电音，只测变化量)
  - energy_dyn    : 帧能量变异系数(CV=std/mean) -> 语气强弱动态/抑扬
  - rate_idx      : 语音事件率(能量onset/sec ≈ 音节速率) -> 说话快慢

控制度指标 control(%):
  人为给每个风格一个"预期表现力等级" (temp_factor/top_k 越高 -> 语调起伏本应越大)。
  统计测量特征(标高、能量动态)与预期等级的 Spearman 相关性,越接近 1 代表
  风格越能被采样参数"按住"。rate 轴因不做程序变速(护电音)只作参考,不纳入控制度。

用法:
  python style_prosody.py --styles 自然,活力,慵懒,娇俏 --text "小伴你今天来啦,我好开心呢!"
输出: out/style_prosody_report.json + 控制度汇总
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_gguf import GGUFTTS, STYLE_PRESETS, DEFAULT_STYLE  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")


# ---------------- 特征提取 ----------------
def _frame_energy(wav, sr, hop=160, win=400):
    n = len(wav)
    idx = np.arange(0, max(n - win, 1), hop)
    e = np.array([np.mean(wav[i:i + win] ** 2) for i in idx])
    return e


def _f0_autocorr(x, sr, fmin=60.0, fmax=400.0):
    """逐帧自相关基频估计（20ms窗），返回 (voiced_f0_hz, voiced_frame_ratio)。"""
    frame = int(sr * 0.025)
    hop = int(sr * 0.010)
    n = len(x)
    if n < frame * 2:
        return 0.0, 0.0
    lags = np.arange(int(sr / fmax), int(sr / fmin) + 1)
    f0s = []
    x = x.astype(np.float32)
    for start in range(0, n - frame, hop):
        seg = x[start:start + frame]
        seg = seg - seg.mean()
        rms = np.sqrt(np.mean(seg ** 2)) + 1e-9
        if rms < 1e-3 * max(0.05, np.max(np.abs(x))):
            continue
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        ac = ac / (ac[0] + 1e-9)
        # 选第一个显著峰（避开发声激励零点搜出多个倍频）
        peak = None
        for li in range(1, len(lags)):
            d = 3
            lo, hi = max(0, li - d), min(len(ac), li + d + 1)
            if ac[lags[li]] == max(ac[lags[lo:hi]]) and ac[lags[li]] > np.max(ac[lags]) * 0.3:
                peak = li
            if peak is not None:
                break
        if peak is None:
            continue
        f0 = sr / lags[peak]
        if fmin <= f0 <= fmax:
            f0s.append(f0)
    if not f0s:
        return 0.0, 0.0
    return float(np.std(np.array(f0s))), float(len(f0s) / max(1, (n - frame) // hop))


def extract_features(wav, sr):
    wav = np.asarray(wav, dtype=np.float32)
    e = _frame_energy(wav, sr)
    voiced_frames = e[e >= max(np.median(e), 1e-6)]
    energy_cv = (voiced_frames.std() / (voiced_frames.mean() + 1e-9)) if len(voiced_frames) else 0.0
    pitch_std, _ratio = _f0_autocorr(wav, sr)
    # 语音事件率：能量包络过阈值 onsets / 时长
    thr = max(np.median(e) * 1.5, 1e-6)
    onsets = 0
    prev = False
    for v in e:
        cur = v > thr
        if cur and not prev:
            onsets += 1
        prev = cur
    dur = len(wav) / float(sr)
    rate_idx = onsets / dur if dur > 0 else 0.0
    return {"pitch_std_hz": round(float(pitch_std), 2),
            "energy_dyn": round(float(energy_cv), 3),
            "rate_idx": round(float(rate_idx), 3),
            "dur_s": round(float(dur), 3)}


def _expected_rank(style):
    """预期表现力等级：temp_factor 为主、top_k 微调，越大语调起伏越应显著。"""
    sp = STYLE_PRESETS.get(style, STYLE_PRESETS[DEFAULT_STYLE])
    return float(sp["temp_factor"]) + 0.002 * int(sp.get("top_k", 50))


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    da = np.argsort(np.argsort(a)); db = np.argsort(np.argsort(b))
    n = len(a)
    if n < 2 or np.std(da) == 0 or np.std(db) == 0:
        return 0.0
    return float(np.corrcoef(da, db)[0, 1])


def run_style_eval(styles, text, out_json=None, emotion="开心", repeats=2):
    gg = GGUFTTS.get()
    feat_rows, synth_meta = [], {}
    start = time.time()
    for st in styles:
        t0 = time.time()
        acc = {"pitch_std_hz": 0.0, "energy_dyn": 0.0, "rate_idx": 0.0, "dur_s": 0.0}
        params = None
        for rep in range(repeats):
            seed = gg.stable_seed + rep * 1000
            wav, sr, meta = gg.synthesize(text, emotion=emotion, style=st, seed=seed)
            f = extract_features(wav, sr)
            for k in ("pitch_std_hz", "energy_dyn", "rate_idx", "dur_s"):
                acc[k] += f[k]
            params = {k: meta.get(k) for k in
                      ("temperature", "top_k", "top_p", "repeat_penalty")}
        for k in acc:
            acc[k] = round(acc[k] / repeats, 3)
        f = dict(acc)
        f["style"] = st
        f["expected_rank"] = round(_expected_rank(st), 3)
        f["expected_rate"] = float(STYLE_PRESETS.get(st, {}).get("rate", 1.0))
        f["params"] = params
        f["repeats"] = repeats
        f["wall_s"] = round(time.time() - t0, 2)
        feat_rows.append(f)
        synth_meta[st] = params
        print(f"  [{st}] {f}", flush=True)

    # 可靠控制信号(节奏/步速，由输入层标点塑形决定，确定性高)：
    #   慢风格 dur 更长(dur 与 style.rate 负相关)；快风格 rate_idx 更高。
    exp_rate = [r["expected_rate"] for r in feat_rows]
    rho_dur = _spearman(exp_rate, [-r["dur_s"] for r in feat_rows])
    rho_rateidx = _spearman(exp_rate, [r["rate_idx"] for r in feat_rows])
    # 色彩信号(语调起伏，受采样参数影响，稳健性中等)：
    exp = [r["expected_rank"] for r in feat_rows]
    rho_pitch = _spearman(exp, [r["pitch_std_hz"] for r in feat_rows])
    rho_energy = _spearman(exp, [r["energy_dyn"] for r in feat_rows])
    # 节奏轴确定性高，权重 0.6；色彩轴权重 0.4
    control = (0.3 * rho_dur + 0.3 * rho_rateidx +
               0.25 * rho_pitch + 0.15 * rho_energy) * 100.0
    control = round(max(0.0, min(control, 100.0)), 1)

    report = {
        "text": text, "emotion": emotion,
        "styles_evaled": styles,
        "repeats": repeats,
        "control_pct": control,
        "spearman": {"dur_vs_rate": round(rho_dur, 3),
                     "rateidx_vs_rate": round(rho_rateidx, 3),
                     "pitch_std_vs_exp": round(rho_pitch, 3),
                     "energy_dyn_vs_exp": round(rho_energy, 3)},
        "note": "control_pct=(0.3*dur + 0.3*rateidx + 0.25*pitch + 0.15*energy) "
                "rank-corr *100；节奏轴(标点塑形)确定性高、色彩轴受采样影响中等。",
        "features": feat_rows,
        "synth_params": synth_meta,
        "total_wall_s": round(time.time() - start, 1),
    }
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        print(f"[ok] 报告写回: {out_json}", flush=True)
    print(f"[score] 说话风格控制度 control_pct = {control}%", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", default="自然,温柔,活力,娇俏")
    ap.add_argument("--text", default="小伴你今天来啦,我真的好开心呀!")
    ap.add_argument("--emotion", default="开心")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "style_prosody_report.json"))
    args = ap.parse_args()
    styles = [s for s in (args.styles or "").split(",") if s.strip()]
    styles = [s for s in styles if s in STYLE_PRESETS] or [DEFAULT_STYLE]
    run_style_eval(styles, args.text, args.out, emotion=args.emotion,
                   repeats=args.repeats)


if __name__ == "__main__":
    main()