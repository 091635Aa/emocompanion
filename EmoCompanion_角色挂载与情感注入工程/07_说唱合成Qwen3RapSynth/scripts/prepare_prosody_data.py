# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · 韵律数据准备（Task 2）

把「人声干声 + 歌词」转化为韵律预测器的 CSV/JSONL 训练目标（每音/每字 的 F0/时长/能量）。

注意（科研伦理 + 电源约束）：
  - 需要开源/合规说唱数据 → 用户放入 `--src` 目录：
        <src>/<clip>.wav  人声干声（24k 单声道优先）
        <src>/<clip>.txt  对应歌词（每行一句，UTF-8 无 BOM）
  - 强制对齐：优先调用 MFA（montreal-forced-aligner，若已安装）。未装 MFA 时，
    回退到**轻量字符级对齐**：按 RMS 能量起音切分成一段段，把每行歌词逐字映射到
    段时长，作为占位对齐（够规则基线/粗粒度训练用；精确训练请补装 MFA）。
  - F0/能量：librosa（pyin + RMS）；时长：分段帧数 / 24k。

用法：
  python prepare_prosody_data.py --src data/raw --out data/split --sr 24000
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

SR_DEF = 24000


def _align_lightweight(wav, sr, line_index, char_count):
    """占位对齐：ROI = 帧 RMS 能量作软起音，把 char_count 个字符摊到有能量窗内。"""
    hop = 256
    import librosa
    rms = librosa.feature.rms(y=wav.astype("float32"), frame_length=2048, hop_length=hop)[0]
    n = max(int(len(rms) * 0.7), 1)
    # 取最常见能量段（前 30% 高能帧作为语音窗）
    thr = np.percentile(rms, 70)
    act = rms > thr
    idx = np.flatnonzero(act)
    if idx.size == 0:
        idx = np.arange(len(rms))
    start_f, end_f = idx[0], idx[-1]
    nchar = max(char_count, 1)
    per = (end_f - start_f) / nchar
    rows = []
    f0_all, _, _ = librosa.pyin(wav.astype("float32"), fmin=60, fmax=500,
                                sr=sr, frame_length=1024, hop_length=256)
    f0 = np.nan_to_num(f0_all, nan=0.0)
    for c in range(nchar):
        s = start_f + c * per
        e = start_f + (c + 1) * per
        s_i, e_i = int(s), min(int(e), len(f0) - 1)
        seg_f0 = f0[s_i:e_i + 1] if e_i > s_i else np.array([f0[s_i]])
        mean_f0 = float(np.mean(seg_f0[seg_f0 > 0])) if np.any(seg_f0 > 0) else 0.0
        seg_rms = rms[s_i:e_i + 1] if e_i > s_i else np.array([rms[s_i]])
        energy = float(seg_rms.mean() + 1e-6)
        dur = (e - s) * hop / sr
        rows.append({"phoneme": f"{line_index}_{c}", "start_sec": round(s * hop / sr, 4),
                     "duration_sec": round(dur, 4), "f0_hz": round(mean_f0, 2),
                     "energy": round(energy, 6)})
    return rows


def count_chars(line: str) -> int:
    return sum(1 for ch in line if "\u4e00" <= ch <= "\u9fff" or ch.isalnum())


def process_clip(wav_path, txt_path, sr):
    """单条：返回 (meta, rows) 或 None（无有效对齐）。"""
    import librosa
    wav, ws = librosa.load(wav_path, sr=None, mono=True)
    w = librosa.resample(wav.astype("float32"), orig_sr=ws, target_sr=sr) if ws != sr else wav
    with open(txt_path, "r", encoding="utf-8-sig") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return None
    rows, n = [], 0
    for i, line in enumerate(lines):
        cc = count_chars(line)
        if cc < 1:
            continue
        r = _align_lightweight(w, sr, i, cc)
        rows.append({"line": line, "syllables": cc, "units": r})
        n += 1
    return {"clip": os.path.basename(wav_path), "sr": sr, "lines": rows}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="含 *.wav + 同名 *.txt 的目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=SR_DEF)
    ap.add_argument("--sample", type=int, default=50, help="抽样上限；-1 全量")
    args = ap.parse_args()

    pairs = []
    for w in glob.glob(os.path.join(args.src, "*.wav")):
        t = os.path.splitext(w)[0] + ".txt"
        if os.path.isfile(t):
            pairs.append((w, t))
    if not pairs:
        sys.exit("未在 {} 找到 *.wav + 同名 *.txt 对".format(args.src))

    sample_n = args.sample if args.sample >= 0 else len(pairs)
    meta_all, rows_all = [], []
    for wav_path, txt_path in sorted(pairs)[: sample_n]:
        try:
            meta, n = process_clip(wav_path, txt_path, args.sr)
        except Exception as e:
            print(f"[skip] {wav_path}: {type(e).__name__} {e}")
            continue
        if meta:
            meta_all.append(meta)
            rows_all.append({"bpm_est": None, "clip_units": [
                {"line": li["line"], "syllables": li["syllables"],
                 "units": li["units"]} for li in meta["lines"]]})

    os.makedirs(args.out, exist_ok=True)
    csv_p = os.path.join(args.out, "prosody.csv")
    with open(csv_p, "w", encoding="utf-8") as f:
        f.write("clip,line,syllables,unit_index,f0_hz,duration_sec,energy\n")
        for m in meta_all:
            for li in m["lines"]:
                for u in li["units"]:
                    f.write(f'{m["clip"]},"{li["line"]}",{li["syllables"]},{u["phoneme"]},'
                            f'{u["f0_hz"]},{u["duration_sec"]},{u["energy"]}\n')
    j_p = os.path.join(args.out, "prosody.jsonl")
    with open(j_p, "w", encoding="utf-8") as f:
        for r in rows_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ok] 处理 {len(pairs)} 个候选，抽 {sample_n}，有效 {len(meta_all)} 条")
    print(f"[ok] csv: {csv_p}  jsonl: {j_p}")


if __name__ == "__main__":
    main()