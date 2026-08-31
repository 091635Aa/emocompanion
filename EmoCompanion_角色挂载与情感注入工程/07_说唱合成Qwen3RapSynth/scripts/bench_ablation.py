# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · 消融/基线对比基准（Task 5.2 补跑）

一次加载 Base+LoRA，对三风格各合成中性干声，然后推导两个对照版本：
  - baseline      : 无注入（直接用 Qwen3-TTS + 情感标签，逐行顺次拼接，无对拍/变速/变调/能量），即任务书的基线；
  - injected(ours): 走 `integration.injector` 的间接韵律注入（对拍 time-stretch + pitch + energy）；
  - syllable      : 行级注入的基础上，在每行 start 处叠加音节级重音增益（能量按音符节拍做微脉动），
                   用于「行级 vs 音节级」粒度对照（真实形素级须由学习预测器逐帧注入，见报告）。

输出：
  output/ablation/<style>_baseline.wav、<style>_ours.wav、<style>_syllable.wav
  output/ablation/ablation_metrics.json   （与 eval/metrics 口径一致）
"""
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from prosody_model.rules import predict_lyrics  # noqa: E402
from integration import injector  # noqa: E402
SR = injector.SR
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from generate_rap import EXAMPLES  # noqa: E402  (复用内置示例歌词)


def baseline_concat(neutral, rms_normalize=True):
    """无注入基线：将各行中性干声顺次拼接，仅做归一避免爆音。"""
    import librosa
    pieces = []
    for wav, sr in neutral:
        w = injector._resample_mono(wav, sr)
        if rms_normalize:
            rms = np.sqrt(np.mean(w ** 2)) + 1e-8
            w = w / rms * 0.3  # 统一 0.3 响度，纯音质对比
        pieces.append(np.clip(w, -1.0, 1.0))
    if not pieces:
        return None, []
    w = np.concatenate(pieces).astype("float32")
    return w, [0.0] + np.cumsum(
        [p.shape[0] / SR for p in pieces])[:-1].tolist()


def syllable_pulse(plan, neutral):
    """行级注入 + 音节级能量脉动（每行按 syllable 数在拍内做轻重交替 0.7–1.3）。"""
    wav, onsets = injector.inject(plan, neutral)
    if wav is None:
        return None, []
    # 对已在拍的时间轴上，把每个节拍窗做强拍/弱拍增益脉动
    hop = int(SR * plan.beat_sec / 2.0)
    out = wav.copy()
    n_full = int(len(out) / hop)
    for k in range(n_full):
        seg = out[k * hop: k * hop + hop]
        gain = 1.3 if k % 2 == 0 else 0.7
        out[k * hop: k * hop + hop] = np.clip(seg * gain, -1.0, 1.0)
    return out, onsets


def main():
    out_dir = os.path.join(_ROOT, "output", "ablation")
    os.makedirs(out_dir, exist_ok=True)

    import soundfile as sf
    from eval.metrics import summarize
    from tts.synthesizer import RaSynthCore

    tts = RaSynthCore(); tts.load()  # 单次加载

    styles = {"快嘴": 96.0, "旋律说唱": 84.0, "硬核": 72.0}
    report = {}
    for style, bpm in styles.items():
        lyrics = "\n".join(EXAMPLES[style])
        plan = predict_lyrics(lyrics, bpm=bpm, style=style)
        # 1) 中性干声（一次）
        neutral = [tts.synthesize(lp.text, emotion="硬核")[:2] for lp in plan.lines]

        # 2) baseline（无注入）
        b_wav, b_onset = baseline_concat(neutral)
        # 3) ours（行级注入）
        o_wav, o_onset = injector.inject(plan, neutral)
        # 4) syllable（音节级脉动）
        s_wav, s_onset = syllable_pulse(plan, neutral)

        texts = [lp.text for lp in plan.lines]
        row = {
            "baseline": summarize(plan, b_wav, b_onset, texts) if b_wav is not None else None,
            "ours_line": summarize(plan, o_wav, o_onset, texts) if o_wav is not None else None,
            "ours_syllable": summarize(plan, s_wav, s_onset, texts) if s_wav is not None else None,
        }
        for tag, w in (("baseline", b_wav), ("ours", o_wav), ("syllable", s_wav)):
            if w is not None:
                sf.write(os.path.join(out_dir, f"{style}_{tag}.wav"), np.asarray(w), SR)
        report[style] = row
        print(f"[{style}] BPM={bpm}", json.dumps(row, ensure_ascii=False))

    with open(os.path.join(out_dir, "ablation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.makedirs(out_dir, exist_ok=True)
    print("[ok] 已写 output/ablation/*.wav 与 ablation_metrics.json")


if __name__ == "__main__":
    main()