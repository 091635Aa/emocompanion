# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · 对拍表离线重算（不依赖 GPU）

读取 output/ablation/<style>_baseline|ours|syllable.wav（任意已写出的），
用与 eval/metrics 相同的口径重算并输出对比表 JSON —— 供文档在不等待完整进程时引用。

用法： python scripts/report_ablation_metrics.py
"""
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from prosody_model.rules import predict_lyrics  # noqa: E402
from eval import metrics  # noqa: E402
from generate_rap import EXAMPLES  # noqa: E402

SR = 24000
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path and os.path.dirname(sys.executable) != _SCRIPTS:
    sys.path.insert(0, _SCRIPTS)
from generate_rap import EXAMPLES  # noqa: E402


def _onsets_naive(neutral_durs):
    t = 0.0
    out = []
    for d in neutral_durs:
        out.append(t)
        t += d
    return out


def main():
    import soundfile as sf
    ab = os.path.join(_ROOT, "output", "ablation")
    styles = {"快嘴": 96.0, "旋律说唱": 84.0, "硬核": 72.0}
    report = {}
    for style, bpm in styles.items():
        lyrics = "\n".join(EXAMPLES[style])
        plan = predict_lyrics(lyrics, bpm=bpm, style=style)
        neutral_durs = [round(lp.duration_sec, 3) for lp in plan.lines]
        texts = [lp.text for lp in plan.lines]
        row = {}
        for tag in ("baseline", "ours", "syllable"):
            p = os.path.join(ab, f"{style}_{tag}.wav")
            if not os.path.isfile(p):
                row[tag] = None
                continue
            w, _ = sf.read(p, dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=-1)
            if tag == "baseline":
                ons = _onsets_naive(neutral_durs)
            else:
                ons = [lp.start_sec for lp in plan.lines]
            m = metrics.summarize(plan, w, ons, texts)
            m["onset_beat_error_s"] = round(metrics.onset_beat_error(w, plan.beat_sec), 4)
            row[tag] = m
        report[style] = row
        print(f"[{style}]", json.dumps(row, ensure_ascii=False))
    with open(os.path.join(ab, "ablation_metrics_offline.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[ok] ablations_offline.json")


if __name__ == "__main__":
    main()