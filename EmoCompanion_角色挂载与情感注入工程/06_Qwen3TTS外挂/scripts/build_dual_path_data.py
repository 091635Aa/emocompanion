# -*- coding: utf-8 -*-
"""
EmoCompanion 双路 TTS 微调数据构建（用户指定方案：音色路 + 情感路）
================================================================
根据官方 Qwen3-TTS fine-tuning JSONL 格式（{"audio":"..","text":"..","ref_audio":".."}）生成。

两条训练路径（非传统单一打标训练）：
  1) 音色路 voice  : audio + 原始transcript + 固定中性 ref_audio -> 锁定声纹/韵律基线
  2) 情感路 emotion: audio + [emotion]<标签>[/emotion] + transcript + 该情感 ref_audio
                     -> 让模型学会把情感标签/情感 ref 映射到情感韵律

输出 4 个 raw jsonl（后续 prepare_data.py 编码 audio_codes 后喂训练）：
  voice_train_raw.jsonl / voice_val_raw.jsonl
  emotion_train_raw.jsonl / emotion_val_raw.jsonl

用法：
  python build_dual_path_data.py \
      --labels 情感打标训练集_*_合格6951_总7030.jsonl \
      --wav_root D:/ACQ富/wav_24k \
      --out data
"""
import argparse
import json
import os
import random
from collections import Counter

EMOTION_TOKEN_TMPL = "[emotion]{emotion}[/emotion]"

# 打标情感标签 -> 泛化情感（用于情感路条件）
EMOTION_ALIAS = {
    "开心": "开心", "快乐": "开心", "高兴": "开心", "活泼": "开心",
    "俏皮": "俏皮", "调皮": "俏皮", "卖萌": "俏皮",
    "撒娇": "撒娇", "甜": "撒娇",
    "悲伤": "悲伤", "难过": "悲伤", "委屈": "悲伤", "哭": "悲伤",
    "平静": "平静", "中性": "平静", "淡定": "平静", "轻松": "平静",
    "激动": "激动", "兴奋": "激动", "热情": "激动",
    "温柔": "温柔", "亲切": "温柔", "感谢": "温柔",
}
VAL_RATIO = 0.1


def normalize_emotion(emo):
    return EMOTION_ALIAS.get(emo, "平静")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--wav_root", required=True, help="24k mono wav 根目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    records = []
    with open(args.labels, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"[labels] {len(records)} 条")

    samples, skip = [], 0
    for rec in records:
        inp = rec.get("输入", {})
        base = os.path.basename(inp.get("片段路径", ""))
        stem = os.path.splitext(base)[0]
        wav = os.path.join(args.wav_root, stem + ".wav")
        if not os.path.exists(wav):
            skip += 1
            continue
        outb = rec.get("输出", {})
        transcript = (outb.get("transcript") or "").strip()
        if not transcript:
            skip += 1
            continue
        emo = normalize_emotion(outb.get("discrete_emotion_primary", "平静"))
        samples.append({"wav": wav, "stem": stem, "text": transcript,
                        "emotion": emo, "source_file": inp.get("原文件", stem)})
    print(f"[samples] useable {len(samples)} (skip {skip})")

    emo_stat = Counter(s["emotion"] for s in samples)
    print(f"[emotion-dist] {dict(emo_stat)}")

    # 情感 -> 代表 ref（每个情感取中间长度文本样本的 wav）
    ref_pool = {e: [] for e in emo_stat}
    for s in samples:
        ref_pool[s["emotion"]].append(s)

    def pick_ref(emo):
        pool = sorted(ref_pool.get(emo, []), key=lambda x: len(x["text"]))
        return pool[len(pool) // 2]["wav"] if pool else None

    # 统一中性 ref（音色路固定 use）
    neutral_ref = pick_ref("开心") or samples[0]["wav"]
    print(f"[neutral-ref] {neutral_ref}")

    # 防泄漏按 source_file 卷划分
    by_file = {}
    for s in samples:
        by_file.setdefault(s["source_file"], []).append(s)
    keys = list(by_file.keys())
    random.shuffle(keys)
    n_val = max(1, round(len(keys) * VAL_RATIO))
    val_keys = set(keys[:n_val])

    voice_train, voice_val, emo_train, emo_val = [], [], [], []
    for k, items in by_file.items():
        vset = k in val_keys
        for s in items:
            voice_row = {"audio": s["wav"], "text": s["text"], "ref_audio": neutral_ref}
            emo_ref = pick_ref(s["emotion"]) or neutral_ref
            emo_text = f"{EMOTION_TOKEN_TMPL.format(emotion=s['emotion'])}{s['text']}"
            emo_row = {"audio": s["wav"], "text": emo_text, "ref_audio": emo_ref, "emotion": s["emotion"]}
            (voice_val if vset else voice_train).append(voice_row)
            (emo_val if vset else emo_train).append(emo_row)

    def write(name, rows):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write("voice_train_raw.jsonl", voice_train)
    write("voice_val_raw.jsonl", voice_val)
    write("emotion_train_raw.jsonl", emo_train)
    write("emotion_val_raw.jsonl", emo_val)
    print(f"[split] voice train {len(voice_train)}/val {len(voice_val)} | "
          f"emotion train {len(emo_train)}/val {len(emo_val)}")
    print(f"[out] 已写出四路 raw jsonl 到 {args.out}")


if __name__ == "__main__":
    main()