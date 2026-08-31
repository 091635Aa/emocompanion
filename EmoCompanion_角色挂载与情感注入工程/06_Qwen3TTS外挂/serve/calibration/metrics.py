# -*- coding: utf-8 -*-
"""calibration/metrics —— 生成语音 vs 黄金样本 的相似度度量

用于参考文本闭环校准：text→TTS 生成 vs 原训练样本，逐项打分后加权成相似度 S。
所有依赖(尤其 ASR/说话人编码/GPU)均为可选：缺失时该维 metric 置 n/a 并在加权时归一化，保证框架缺件也能跑（降级为韵律/语速维度）。

对外:
  prosody_features(wav) -> dict           # 韵律描述（CPU, librosa）
  speaker_emb(engine, wav) -> np.ndarray   # 说话人嵌入（GPU 模型，可选）
  composite(report) -> dict                # 加权相似度
"""
import os

import numpy as np


def load_audio(path: str):
    """读 wav，返回 (float32 mono, sr)。"""
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype("float32"), int(sr)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def prosody_features(path_or_y, sr=None):
    """韵律/语音速度特征（纯 CPU）。支持 路径 str 或 (y, sr) 元组。"""
    import librosa
    if isinstance(path_or_y, str):
        y, sr = load_audio(path_or_y)
    elif isinstance(path_or_y, (tuple, list)) and len(path_or_y) == 2:
        y = np.asarray(path_or_y[0], dtype="float32")
        sr = int(path_or_y[1]) if path_or_y[1] else sr or 24000
    else:
        y = np.asarray(path_or_y, dtype="float32")
        if sr is None:
            sr = 24000
    y = np.asarray(y, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=-1)
    dur = y.shape[0] / float(sr) if sr else 0.0

    # 音高：yin 优先，pyin 兜底；取有限帧的中位数/方差，全无则 nan
    f0_mean = f0_std = np.nan
    try:
        f0, voiced, _ = librosa.yin(y, fmin=60, fmax=800, sr=sr, frame_length=1024)
        f0_v = f0[np.isfinite(f0) & voiced]
        if f0_v.size < 8:
            f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=800, sr=sr, frame_length=1024)
            f0 = np.asarray(f0, dtype="float64")
            f0_v = f0[np.isfinite(f0)]
        if f0_v.size >= 8:
            f0_mean = float(np.nanmedian(f0_v))
            f0_std = float(np.nanstd(f0_v))
    except Exception:
        pass

    # 能量：RMS 短窗，去静音后取均/方差(起伏)
    win = int(sr * 0.06)
    nframes = len(y) // win
    if nframes:
        frames = y[: nframes * win].reshape(nframes, win)
        rms = np.sqrt((frames ** 2).mean(axis=-1))
        rms = rms[rms > 1e-4]
        e_mean = float(rms.mean()) if rms.size else 0.0
        e_std = float(rms.std()) if rms.size else 0.0
    else:
        e_mean = e_std = 0.0

    return {
        "duration_s": float(round(dur, 3)),
        "f0_mean": f0_mean, "f0_std": f0_std,
        "energy_mean": e_mean, "energy_std": e_std,
    }


def speak_rate(seconds: float, n_chars: int):
    """语速：每秒字数。"""
    return round(n_chars / seconds, 3) if seconds > 0 else 0.0


def edit_similarity(a: str, b: str) -> float:
    """字符级归一化编辑相似度(1 - dist/max)；用作 ASR 转写保真。"""
    if not a or not b:
        return 0.0
    na, nb = len(a), len(b)
    dp = list(range(nb + 1))
    for i in range(1, na + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, nb + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    dist = dp[nb]
    return round(_clamp(1 - dist / max(na, nb), 0, 1), 4)


def speaker_emb(engine, wav):
    """用模型自带的说话人编码器提取嵌入(cos 相似度用)。wav 可为路径或 (y,sr) 内存。
    缺失/失败返回 None。"""
    try:
        if getattr(engine, "_model", None) is None:
            engine.load()
        import librosa
        if isinstance(wav, str):
            y, sr = librosa.load(wav, sr=None, mono=True)
        else:
            y, sr = wav[0].astype("float32"), int(wav[1])
        if sr != engine._model.model.speaker_encoder_sample_rate:
            y = librosa.resample(y.astype("float32"), orig_sr=int(sr),
                                 target_sr=engine._model.model.speaker_encoder_sample_rate)
        emb = engine._model.model.extract_speaker_embedding(
            audio=y.astype("float32"),
            sr=engine._model.model.speaker_encoder_sample_rate)
        import torch
        return emb.detach().cpu().float().flatten() if torch.is_tensor(emb) else np.asarray(emb, dtype="float32").flatten()
    except Exception:
        return None


def _rel_err(x, gt):
    """相对误差，容忍度归一：(1 - |x-gt|/(gt+eps)) 截断到 [0,1]。"""
    gt = float(gt)
    if np.isnan(gt) or gt <= 0:
        return None
    return _clamp(1 - abs(float(x) - gt) / gt, 0, 1)


def composite(*, gen_pros, gt_pros, gen_nchars, gen_emb=None, gt_emb=None,
              asr_sim=None, weights=None, report: dict = None):
    """加权相似度 S ∈ [0,1]。缺失维自动归一权重。

    weights 默认: prosody{rate 0.35, f0 0.20, energy 0.15}, voice 0.30, asr 0.30
    """
    if weights is None:
        weights = {"voice": 0.30, "asr": 0.30,
                   "rate": 0.15, "f0": 0.15, "energy": 0.10}
    if report is None:
        report = {}

    parts = {}
    # 语速相似
    g_rate = gen_nchars / gen_pros["duration_s"] if gen_pros["duration_s"] > 0 else 0
    t_rate = gen_nchars / gt_pros["duration_s"] if gt_pros["duration_s"] > 0 else g_rate
    r = _rel_err(g_rate, t_rate)
    if r is not None:
        parts["rate"] = r
    # 音高(均值/方差)相似
    if not (np.isnan(gt_pros["f0_mean"]) or np.isnan(gen_pros["f0_mean"])):
        fm = _rel_err(gen_pros["f0_mean"], gt_pros["f0_mean"])
        fs = _rel_err(gen_pros["f0_std"], gt_pros["f0_std"])
        if fm is not None and fs is not None:
            parts["f0"] = (fm + fs) / 2
    # 能量起伏
    fe = _rel_err(gen_pros["energy_std"], gt_pros["energy_std"])
    if fe is not None:
        parts["energy"] = fe
    # 说话人相似
    if gen_emb is not None and gt_emb is not None:
        ge, te = np.asarray(gen_emb, dtype="float32"), np.asarray(gt_emb, dtype="float32")
        ge /= max(np.linalg.norm(ge), 1e-9)
        te /= max(np.linalg.norm(te), 1e-9)
        parts["voice"] = _clamp(float(np.dot(ge, te)), 0, 1)
    # ASR 保真
    if asr_sim is not None:
        parts["asr"] = asr_sim

    # 归一权重到可得维度
    avail = {k: w for k, w in weights.items() if k in parts}
    if not avail:
        report["composite"] = 0.0
        report["parts"] = {k: None for k in weights}
        return report
    wsum = sum(avail.values())
    score = sum(parts[k] * (w / wsum) for k, w in avail.items())
    report["parts"] = {k: parts.get(k) for k in weights}
    report["composite"] = round(_clamp(score, 0, 1), 4)
    return report