# -*- coding: utf-8 -*-
"""
打标工具 — 中文打标工具（输出到 f:\\打标）
==========================================
- 批量生成回复：用 推理框架 逐条生成，记录
  {提示词, 回复文本, 平均熵, 重复率, 情感命中率, λ, γ, τ}
- 输出标注任务：为每条补充 情感维度 / 质量评分(1-5,默认3) / 标注状态(待标注) /
  模型 / 量化 / 时间戳，写 标注任务_批次名.json（UTF-8, ensure_ascii=False）
  与同名 .csv（utf-8-sig，Excel 可读）

用法示例：
    python 打标工具.py --模型 "c:\\...\\Qwen2.5-1.5B-Instruct" --量化 4bit --批次 批次1
    python 打标工具.py --模型 ... --提示词 "自定义提示词.txt" --输出 "f:\\打标"
"""
import sys
import os
import json
import csv
import time
import argparse
from datetime import datetime

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

import 推理框架
from echo_common import 测试提示词

量化映射 = {"fp16": None, "4bit": "4bit"}


def 推断情感维度(提示词):
    """按提示词所属维度映射：命中 测试提示词 字典的某维度即用该维度，否则'待定'"""
    for 维度, 列表 in 测试提示词.items():
        if 提示词 in 列表:
            return 维度
    return "待定"


def 批量生成回复(模型路径, 量化, 提示词集, 动态策略="B", max_new_tokens=128, 长上下文=False):
    """用 推理框架 逐条生成，返回 list[dict]（单条失败记录错误文本不中断）"""
    框架 = 推理框架.推理框架(模型路径, 量化=量化, rag=False, lora=None,
                          动态策略=动态策略, 长上下文=长上下文)
    结果列表 = []
    for i, 提示 in enumerate(提示词集):
        print(f"[打标] 生成中 {i + 1}/{len(提示词集)}: {提示[:30]}")
        try:
            结果 = 框架.生成(提示, max_new_tokens=max_new_tokens)
            结果列表.append({
                "提示词": 提示,
                "回复文本": 结果["文本"],
                "平均熵": round(结果["平均熵"], 4),
                "重复率": round(结果["重复率"], 4),
                "情感命中率": round(max(0.0, min(1.0, 结果["情感命中率"])), 4),  # 命中率口径修复：clip 到 [0,1]
                "λ": 结果["λ"],
                "γ": 结果["γ"],
                "τ": 结果["τ"],
            })
        except Exception as e:
            print(f"[打标] 生成失败: {e}")
            结果列表.append({
                "提示词": 提示,
                "回复文本": f"[生成失败] {e}",
                "平均熵": 0.0, "重复率": 0.0, "情感命中率": 0.0,
                "λ": 0.0, "γ": 0.0, "τ": 0.0,
            })
    return 结果列表


def 输出标注任务(结果列表, 输出目录=r"f:\打标", 批次名="批次1", 模型路径="", 量化=None):
    """为每条补充标注字段并写 JSON + CSV，返回 (json路径, csv路径)"""
    os.makedirs(输出目录, exist_ok=True)
    时间戳 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    模型名 = os.path.basename(os.path.normpath(模型路径)) if 模型路径 else ""
    for 条 in 结果列表:
        条["情感维度"] = 推断情感维度(条["提示词"])
        条["质量评分"] = 3          # 整数 1-5，默认 3 待人工复核
        条["标注状态"] = "待标注"
        条["模型"] = 模型名
        条["量化"] = 量化 or "fp16"
        条["时间戳"] = 时间戳

    # JSON（UTF-8, ensure_ascii=False）
    json路径 = os.path.join(输出目录, f"标注任务_{批次名}.json")
    with open(json路径, "w", encoding="utf-8") as f:
        json.dump({
            "批次名": 批次名, "时间戳": 时间戳, "条目数": len(结果列表),
            "条目": 结果列表,
        }, f, ensure_ascii=False, indent=2)

    # CSV（utf-8-sig，Excel 可读）
    csv路径 = os.path.join(输出目录, f"标注任务_{批次名}.csv")
    表头 = ["提示词", "情感维度", "质量评分", "标注状态", "平均熵", "重复率",
            "情感命中率", "λ", "γ", "τ", "回复文本", "模型", "量化", "时间戳"]
    with open(csv路径, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=表头, extrasaction="ignore")
        w.writeheader()
        for 条 in 结果列表:
            w.writerow({k: 条.get(k, "") for k in 表头})

    return json路径, csv路径


def 读取提示词文件(路径):
    """逐行读取提示词文件（跳过空行与注释行）"""
    提示词列表 = []
    with open(路径, encoding="utf-8") as f:
        for 行 in f:
            行 = 行.strip()
            if 行 and not 行.startswith("#"):
                提示词列表.append(行)
    return 提示词列表


def 主流程(args):
    量化值 = 量化映射.get(args.量化, None)
    if args.提示词:
        提示词集 = 读取提示词文件(args.提示词)
    else:
        # 默认使用 测试提示词 全部 15 条
        提示词集 = [提示 for 列表 in 测试提示词.values() for 提示 in 列表]
    if args.重复 > 1:
        # 每条提示词重复生成 args.重复 次（如 15 条 × 3 = 45 条）
        提示词集 = [提示 for 提示 in 提示词集 for _ in range(args.重复)]
    print(f"[打标] 提示词共 {len(提示词集)} 条")

    t0 = time.time()
    结果列表 = 批量生成回复(
        args.模型, 量化值, 提示词集,
        动态策略=args.动态策略, max_new_tokens=args.最大token,
        长上下文=args.长上下文)
    print(f"[打标] 生成总耗时 {time.time() - t0:.1f}s")

    json路径, csv路径 = 输出标注任务(
        结果列表, 输出目录=args.输出, 批次名=args.批次,
        模型路径=args.模型, 量化=量化值)
    print(f"[打标] JSON: {json路径}")
    print(f"[打标] CSV : {csv路径}")

    # 统计：条数、各维度分布
    维度分布 = {}
    for 条 in 结果列表:
        维度分布[条["情感维度"]] = 维度分布.get(条["情感维度"], 0) + 1
    print("-" * 60)
    print(f"[打标] 统计：共 {len(结果列表)} 条")
    for 维度, 数量 in 维度分布.items():
        print(f"  {维度}: {数量} 条")
    print(f"[打标] 标注任务输出完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文打标工具")
    parser.add_argument("--模型", required=True, help="模型路径（必填）")
    parser.add_argument("--量化", choices=["fp16", "4bit"], default="fp16")
    parser.add_argument("--提示词", default=None, help="提示词文件（每行一条）；缺省用 测试提示词 全部 15 条")
    parser.add_argument("--批次", default="批次1")
    parser.add_argument("--输出", default=r"f:\打标")
    parser.add_argument("--动态策略", choices=["A", "B", "C"], default="B")
    parser.add_argument("--最大token", type=int, default=128)
    parser.add_argument("--重复", type=int, default=1,
                        help="每条提示词重复生成次数（默认 1；15 条 × 3 = 45 条）")
    parser.add_argument("--长上下文", action="store_true")
    args = parser.parse_args()
    主流程(args)
