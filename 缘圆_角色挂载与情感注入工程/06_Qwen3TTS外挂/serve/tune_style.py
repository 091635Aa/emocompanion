# -*- coding: utf-8 -*-
"""
tune_style —— 说话风格（StylePlug）自整定/粗略校准
=====================================================================
在推理层做「反复调参」：对每个说话风格，合成一次、测韵律、按误差微调该风格的
rate/gap（节奏轴，确定性强），多轮逼近目标时长。不用紧急训练、不占 GPU 重训、
不触发电音 —— 是"跑测试来回反复训练"在推理层的轻量实现。

目标：把「慢风格 vs 快风格」在时长/停顿上拉开且稳定（-> ~95% 说话风格可控）。
用法：
  python tune_style.py --style 慵懒 --target_dur 5.2 --iters 3
输出: 建议更新后的 STYLE_PRESETS.（rate/gap 增量打印，供手动写入 tts_gguf.py）
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tts_gguf as M  # noqa: E402
from style_prosody import extract_features  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
_TEXT = "圆圆你今天来啦,我真的好开心呀,想不想听我给你唱首歌呢"


def autotune_style(style, target_dur, iters=3, text=_TEXT, emotion="开心"):
    gg = M.GGUFTTS.get()
    orig_rate = M.STYLE_PRESETS[style]["rate"]
    orig_gap = M.STYLE_PRESETS[style]["gap"]
    rate = orig_rate
    gap = orig_gap
    history = []
    for i in range(1, iters + 1):
        wav, sr, _ = gg.synthesize(text, emotion=emotion, style=style,
                                   seed=gg.stable_seed + i * 500)
        f = extract_features(wav, sr)
        dur = f["dur_s"]
        err = target_dur - dur
        history.append({"iter": i, "dur_s": round(dur, 3), "rate": round(rate, 3),
                        "gap": round(gap, 3), "err": round(err, 3)})
        # 时长偏短 -> 放慢(rate 降、gap 增)；偏长 -> 加快
        if i < iters:
            rate = min(max(rate + 0.03 * np.sign(-err) * min(3.0, abs(err) / target_dur), 0.80), 1.20)
            gap = min(max(gap - 0.02 * np.sign(err), 0.03), 0.30)
            # 把整定结果临时写回预设，确保下一轮用新参数合成
            M.STYLE_PRESETS[style]["rate"] = round(float(rate), 3)
            M.STYLE_PRESETS[style]["gap"] = round(float(gap), 3)
    # 收敛度：最后一轮误差相对目标
    final = history[-1]
    control = max(0.0, 100.0 - 100.0 * abs(final["err"]) / target_dur)
    print(f"\n[{style}] 自整定完成 iters={iters} target_dur={target_dur}s")
    for h in history:
        print("   ", h)
    print(f"[score] 目标达成度 = {control:.1f}%  (err={final['err']:.3f}s)")
    print(f"[suggest] 建议写入 tts_gguf.STYLE_PRESETS['{style}']: "
          f"rate={round(rate,3)}, gap={round(gap,3)} "
          f"(原 rate={orig_rate}, gap={orig_gap})")
    report = {"style": style, "target_dur": target_dur, "iters": iters,
              "control_pct": round(control, 1), "history": history,
              "suggest": {"rate": round(rate, 3), "gap": round(gap, 3)},
              "text": text}
    out = os.path.join(OUT_DIR, f"tune_style_{style}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(f"[ok] 写回 {out}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="慵懒")
    ap.add_argument("--target_dur", type=float, required=True)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--text", default=_TEXT)
    ap.add_argument("--emotion", default="开心")
    args = ap.parse_args()
    autotune_style(args.style, args.target_dur, args.iters, args.text, args.emotion)


if __name__ == "__main__":
    main()