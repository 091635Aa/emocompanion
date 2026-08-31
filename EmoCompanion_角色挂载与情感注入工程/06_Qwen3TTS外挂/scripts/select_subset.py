# -*- coding: utf-8 -*-
"""
从双路 raw jsonl 中精选代表性子集（先跑通训练链路，再扩量）
================================================================
原因：全量 6035 条 audio_codes 编码在 RTX 3080 上需 1+ 小时。
先取每情感均衡子集（如总数 ~400-600）快速验证训练，成功后再全量。

用法：
  python select_subset.py \
      --in_jsonl data/voice_train_raw.jsonl --out_jsonl data/voice_train_sub.jsonl \
      --per_emotion 80 --emotion_field emotion --seed 42
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--per_emotion", type=int, default=80,
                    help="每情感保留条数（0=全部保留该情感）")
    ap.add_argument("--emotion_field", default="emotion",
                    help="jsonl 中情感字段名；无则用 text 前缀检测")
    ap.add_argument("--max_total", type=int, default=0, help="0=不限制总条数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    from collections import defaultdict
    buckets = defaultdict(list)
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            emo = rec.get(args.emotion_field, "")
            if not emo:
                # 回退：从 text 前缀提取 [emotion]xxx[/emotion]
                t = rec.get("text", "")
                if t.startswith("[emotion]") and "[/emotion]" in t:
                    emo = t.split("[emotion]")[1].split("[/emotion]")[0]
                else:
                    emo = "neutral"
            buckets[emo].append(rec)

    chosen = []
    for emo, items in buckets.items():
        if args.per_emotion and len(items) > args.per_emotion:
            items = random.sample(items, args.per_emotion)
        chosen.extend(items)
        print(f"[{emo}] keep {len(items)}")

    if args.max_total and len(chosen) > args.max_total:
        chosen = random.sample(chosen, args.max_total)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[out] {len(chosen)} 条 -> {args.out_jsonl}")


if __name__ == "__main__":
    main()