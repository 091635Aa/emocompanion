# -*- coding: utf-8 -*-
"""
实验_动态策略 — 3B Q4 上对比动态策略 A/B/C（λ=0.10, γ=0.08, τ=0.06 基准）
===========================================================================
- 高情感密度提示词：开心 + 悲伤 各 3 条（复用 echo_common.测试提示词）
- 每条 3 次重复、256 tokens、不启用长上下文衰减
- 记录每轮 {策略, 提示词, 重复, 平均熵, 重复率, 情感命中率, 动态触发}
- 汇总每策略均值（熵/重复率/命中率/触发次数），计算 B、C 相对 A 的熵提升百分比
- 输出 数据\\动态策略_二轮.csv（每轮明细）+ 控制台打印 G1 结论（是否达成 +5-10%）

注：3B（hidden_dim=2048）的扫描表推荐参数恰为 λ=0.10, γ=0.08, τ=0.06，
因此直接依赖 推理框架 的推荐参数即可得到任务要求的基准。

用法：
    python 实验_动态策略.py [--模型 ...] [--量化 4bit|fp16]
"""
import sys
import os
import csv
import argparse
from collections import defaultdict

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

import 推理框架
from echo_common import 测试提示词

默认模型 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-3B-Instruct"
数据目录 = r"f:\最终工程架构\数据"
输出文件 = os.path.join(数据目录, "动态策略_二轮.csv")

策略列表 = ["A", "B", "C"]
提示词集 = 测试提示词["开心"] + 测试提示词["悲伤"]   # 高情感密度 6 条
重复次数 = 3
最大token = 256
情感密度阈值 = 0.15


def 主流程(args):
    os.makedirs(数据目录, exist_ok=True)
    量化值 = None if args.量化 == "fp16" else args.量化

    print(f"[实验] 模型={args.模型} 量化={args.量化}")
    框架 = 推理框架.推理框架(args.模型, 量化=量化值, rag=False, lora=None,
                          动态策略="A", 长上下文=False)
    # 确认基准参数（3B 扫描表 = 0.10/0.08/0.06）
    print(f"[实验] 基准参数 λ={框架.λ基准} γ={框架.γ基准} τ={框架.τ基准}（来源={框架.参数来源}）")
    assert 框架.λ基准 == 0.10 and 框架.γ基准 == 0.08 and 框架.τ基准 == 0.06, \
        "基准参数必须为 λ=0.10, γ=0.08, τ=0.06（请确认模型为 Qwen2.5-3B）"

    # 每轮明细行：[策略, 提示词, 重复, 平均熵, 重复率, 情感命中率, 动态触发]
    行列表 = [["策略", "提示词", "重复", "平均熵", "重复率", "情感命中率", "动态触发"]]
    汇总 = defaultdict(lambda: {"熵": [], "重复率": [], "命中率": [], "触发": 0})

    for 策略 in 策略列表:
        print(f"\n[实验] ============ 策略 {策略} ============")
        框架.动态策略 = 策略  # 同一模型热切换策略（属性可变）
        for 重复 in range(1, 重复次数 + 1):
            for 提示 in 提示词集:
                try:
                    结果 = 框架.生成(提示, max_new_tokens=最大token)
                    密度 = 结果["动态信息"].get("情感密度") or 0.0
                    动态触发 = 1 if 密度 > 情感密度阈值 else 0
                    行列表.append([策略, 提示, 重复,
                                   round(结果["平均熵"], 4),
                                   round(结果["重复率"], 4),
                                   round(结果["情感命中率"], 4),
                                   动态触发])
                    汇总[策略]["熵"].append(结果["平均熵"])
                    汇总[策略]["重复率"].append(结果["重复率"])
                    汇总[策略]["命中率"].append(结果["情感命中率"])
                    汇总[策略]["触发"] += 动态触发
                    print(f"  [{策略} 重复{重复}] {提示[:20]} → "
                          f"熵{结果['平均熵']:.3f} 重复{结果['重复率']:.3f} 触发={动态触发}")
                except Exception as e:
                    import traceback
                    print(f"  [{策略} 重复{重复}] {提示[:20]} 失败: {e}")
                    行列表.append([策略, 提示, 重复, 0.0, 0.0, 0.0, 0])

    # 写每轮明细 CSV
    with open(输出文件, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(行列表)
    print(f"\n[实验] 每轮明细已写入: {输出文件}")

    # 汇总每策略均值
    print("-" * 72)
    print(f"{'策略':<6}{'平均熵':>10}{'重复率':>10}{'命中率':>10}{'触发次数':>10}")
    均值 = {}
    for 策略 in 策略列表:
        熵均值 = sum(汇总[策略]["熵"]) / len(汇总[策略]["熵"]) if 汇总[策略]["熵"] else 0.0
        重复均值 = sum(汇总[策略]["重复率"]) / len(汇总[策略]["重复率"]) if 汇总[策略]["重复率"] else 0.0
        命中均值 = max(0.0, min(1.0, sum(汇总[策略]["命中率"]) / len(汇总[策略]["命中率"]))) if 汇总[策略]["命中率"] else 0.0
        均值[策略] = {"熵": 熵均值, "重复率": 重复均值, "命中率": 命中均值,
                      "触发": 汇总[策略]["触发"]}
        print(f"{策略:<6}{熵均值:>10.4f}{重复均值:>10.4f}{命中均值:>10.4f}{汇总[策略]['触发']:>10}")

    # B、C 相对 A 的熵提升百分比
    print("-" * 72)
    A熵 = 均值["A"]["熵"]
    提升 = {}
    for 策略 in ("B", "C"):
        if A熵 > 0:
            提升[策略] = (均值[策略]["熵"] - A熵) / A熵 * 100
        else:
            提升[策略] = 0.0
        print(f"[G1] 策略{策略} 相对 策略A 熵提升: {提升[策略]:+.2f}%")
    达成B = 5.0 <= 提升["B"] <= 10.0
    达成C = 5.0 <= 提升["C"] <= 10.0
    print(f"[G1] 结论：策略B 是否达成 +5~10%：{'是' if 达成B else '否'}"
          f"（实测 {提升['B']:+.2f}%）")
    print(f"[G1] 结论：策略C 是否达成 +5~10%：{'是' if 达成C else '否'}"
          f"（实测 {提升['C']:+.2f}%）")
    if not (达成B or 达成C):
        print("[G1] 提示：未达成 +5~10% 目标，请复核动态策略参数（λ增量/τ目标值）或样本量")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="动态策略对比实验（3B Q4）")
    parser.add_argument("--模型", default=默认模型)
    parser.add_argument("--量化", choices=["fp16", "4bit"], default="4bit")
    args = parser.parse_args()
    主流程(args)
