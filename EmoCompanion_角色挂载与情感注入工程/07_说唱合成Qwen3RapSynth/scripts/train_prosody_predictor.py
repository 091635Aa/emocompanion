# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · 学习型韵律预测器训练（Phase 2 冒烟/管道冒烟）

以规则 teacher 生成的逐音节伪标签训练小型 LSTM（教师蒸馏），产生产物权重，并：
  1) 训练后滚动一个推理自检（learned.predict 输出 ProsodyPlan）；
  2) 自动选型 `rules.get_predictor()` 应改走学习模型（is_learned=True）。

真实数据就绪后：把 `learned.build_label_dataset` 换成 MFA 对齐的真实 F0/时长，重训即可。

用法：
  python scripts/train_prosody_predictor.py --epochs 8 --device cpu --out prosody_model/weights/prosody_lstm.pt
"""
import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from generate_rap import EXAMPLES  # noqa: E402  (内置示例歌词作种子语料)
from prosody_model import learned  # noqa: E402
from prosody_model import rules  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=rules._DEFAULT_WEIGHTS)
    args = ap.parse_args()

    # 种子语料：内置三风格歌词 + 若干不同音节数扩展行
    seed = []
    for v in EXAMPLES.values():
        seed += list(v)
    seed += ["日复一日我把词往节拍里塞", "安静的时候也能听见鼓点敲",
             "把这口火焰烧成一句句刀", "从不回头因为前方有人叫",
             "凌晨三点录音棚亮着灯", "我把生活写进每一个回声"]
    # 语料域增强：音节数近似的伪行（供风格/时长泛化）
    seed = list(dict.fromkeys(seed))  # 去重保序

    # 训练（教师蒸馏，CPU/小模型，数秒级）
    print(f"[train] lines={len(seed)} 语料种子上限 12 行；域增强后各风格×BPM 组合")
    models_ = [s for s in seed]
    model, history = learned.train_predictor(
        models_[:12], bpms=[72, 84, 96, 112], styles=list(EXAMPLES.keys()),
        epochs=args.epochs, batch=args.batch, device=args.device, out_path=args.out,
    )
    print(f"[ok] 权重已存: {args.out}")

    # 推理自检：学习模型 vs 规则基线 的差异
    pred = rules.get_predictor(args.out)
    print(f"[selector] is_learned={pred.is_learned()}")
    plan = pred.predict("\n".join(EXAMPLES["硬核"]), bpm=80, style="硬核")
    base = rules.predict_lyrics("\n".join(EXAMPLES["硬核"]), bpm=80, style="硬核")
    diff = [round(lp.mean_f0 - b.mean_f0, 1) for lp, b in zip(plan.lines, base.lines)]
    print("[learned] 硬核 80bpm 各行 F0(Hz):",
          [round(lp.mean_f0, 1) for lp in plan.lines])
    print("[delta]  相对规则基线 ΔF0:", diff)

    # 记录训练摘要
    meta = {"epochs": args.epochs, "batch": args.batch, "device": args.device,
            "loss_history": [round(h, 4) for h in history], "lines": len(models_[:12]),
            "style": "rule-distillation (teacher), 待真实数据重训"}
    out_dir = os.path.dirname(args.out)
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[meta]", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()