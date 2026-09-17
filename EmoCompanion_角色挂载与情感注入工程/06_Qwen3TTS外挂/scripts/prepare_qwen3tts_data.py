# -*- coding: utf-8 -*-
"""
Qwen3-TTS 角色外挂 —— 数据准备脚本（Task 1）
=================================================
输入：纠正后打标 JSONL（情感打标训练集_*_合格6951_总7030.jsonl）
输出：Qwen3-TTS 训练/验证 JSONL + CSV 清单
  1. 加载纠正版 6951；过滤质量评估报告判为不合格的样本
  2. 映射音频片段 -> Qwen3-TTS 元数据(input_text/audio/emotion)
  3. 90/10 划分 train/val（同一原文件同卷，防跨集泄漏）
用法：
  python prepare_qwen3tts_data.py <labels.jsonl> <split_dir> --quality quality_report.json
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict

QA_MIN_CONFIDENCE = 0.6
QA_MIN_CONTENT_LEN = 50
QA_MIN_QUESTIONS = 3


def load_quality_badlist(quality_report, label_pairs):
    """从质量评估报告中提取不合格样本路径集合（原始片段路径 key）。"""
    bad = set()
    box = quality_report.get("不合格样本列表", [])
    for item in box:
        p = str(item.get("路径", "")).replace("\\", "/")
        bad.add(p)
    return bad


def normalize_path(p):
    return str(p).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="纠正版打标 JSONL 路径")
    ap.add_argument("split_dir", help="输出目录")
    ap.add_argument("--quality", help="质量评估报告 JSON 路径", default=None)
    ap.add_argument("--audio_root", help="音频根目录，若片段路径缺失则用它拼接", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()
    random.seed(args.seed)

    os.makedirs(args.split_dir, exist_ok=True)

    # 1) 读取纠正版标签
    records = []
    with open(args.labels, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"[1] 打标记录: {len(records)}")

    # 2) 读取质量报告（可选），构建不合格路径集
    bad_paths = set()
    if args.quality and os.path.exists(args.quality):
        with open(args.quality, "r", encoding="utf-8") as f:
            bad_paths = load_quality_badlist(json.load(f), records)
        print(f"[2] 不合格样本(按路径): {len(bad_paths)}")

    # 3) 映射为 Qwen3-TTS 训练元数据，并过滤不合格
    rows = []
    dropped_conf = dropped_field = 0
    for rec in records:
        seg_path = normalize_path(rec.get("输入", {}).get("片段路径", ""))
        outbox = rec.get("输出", {})
        transcript = (outbox.get("transcript") or "").strip()
        conf = rec.get("置信度") if rec.get("置信度") is not None else rec.get("输出", {}).get("置信度", 0)
        emotion_tags = outbox.get("情感标签") or [outbox.get("discrete_emotion_primary", "中性")]
        # 置信度过滤（不合格判据之一）
        if conf is None or float(conf) < QA_MIN_CONFIDENCE:
            dropped_conf += 1
            continue
        # 必填字段过滤
        if not transcript or len(transcript) < 8:
            dropped_field += 1
            continue
        # 显式命中质量报告不合格路径
        if seg_path in bad_paths:
            dropped_field += 1
            continue
        row = {
            # Qwen3-TTS EasyFinetuning 兼容字段
            "audio_filepath": seg_path if os.path.exists(seg_path) else (args.audio_root + "/" + seg_path.split("/")[-1]),
            "text": transcript,
            "emotion": "|".join(emotion_tags),            # 情感条件（多标签）
            "emotion_primary": outbox.get("discrete_emotion_primary", "中性"),
            "emotion_intensity": outbox.get("emotion_intensity", 0),
            "valence": outbox.get("valence", 0),
            "arousal": outbox.get("arousal", 0),
            "dominance": outbox.get("dominance", 0),
            "source_clip": seg_path,
            "source_file": rec.get("输入", {}).get("原文件", ""),
            "f0_mean_hz": outbox.get("f0_mean_hz", 0),
            "speech_rate_syll": outbox.get("speech_rate_syll_per_sec", 0),
        }
        rows.append(row)

    print(f"[3] 合格样本进入映射: {len(rows)}  (丢弃: 置信度{ dropped_conf }, 字段/报告{ dropped_field })")

    # 4) 防泄漏按 原文件(source_file) 分组 90/10 划分
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["source_file"]].append(r)
    keys = list(by_file.keys())
    random.shuffle(keys)
    n_val = max(1, round(len(keys) * args.val_ratio))
    val_keys = set(keys[:n_val])
    train, val = [], []
    for k, items in by_file.items():
        (val if k in val_keys else train).extend(items)
    print(f"[4] 分层划分: train {len(train)} / val {len(val)} (按原文件{ len(keys) }卷)")

    # 5) 写出 JSONL + CSV
    t_jsonl = os.path.join(args.split_dir, "train.jsonl")
    v_jsonl = os.path.join(args.split_dir, "val.jsonl")
    for path, data in ((t_jsonl, train), (v_jsonl, val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[5] train.jsonl / val.jsonl 已写出到 {args.split_dir}")

    def write_csv(path, data):
        if not data:
            return
        fields = ["audio_filepath", "text", "emotion", "emotion_primary", "emotion_intensity",
                  "valence", "arousal", "dominance", "source_file", "f0_mean_hz", "speech_rate_syll"]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in data:
                w.writerow({k: r.get(k, "") for k in fields})

    write_csv(os.path.join(args.split_dir, "train.csv"), train)
    write_csv(os.path.join(args.split_dir, "val.csv"), val)
    print("[5] train.csv / val.csv 已写出")


if __name__ == "__main__":
    main()