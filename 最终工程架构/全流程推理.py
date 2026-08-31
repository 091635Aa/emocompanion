# -*- coding: utf-8 -*-
"""
全流程推理 — 端到端 CLI 应用（一条命令跑通）
============================================
用法示例：
    python 全流程推理.py --模型 "c:\\Users\\Administrator\\Documents\\论文+临时目录\\模型空间\\Qwen2.5-1.5B-Instruct"
    python 全流程推理.py --模型 ... --量化 4bit --rag --lora "f:\\lora外挂\\lora_adapters\\emotion_v1" \
                         --动态策略 C --长上下文 --提示 "你好" --提示 "再见"

流程：加载模型 → 构造 推理框架 → 对每条提示词生成 → 汇总打印指标表
      （提示词、熵、重复率、命中率、λ、γ、τ、耗时）→ 结果以 UTF-8 写入
      数据\\全流程_运行记录_时间戳.json（含模型/量化/参数/每条结果/汇总均值）。

全程 try/except：单条失败记录堆栈继续，最后打印成功率。
"""
import sys
import os
import json
import time
import argparse
import traceback
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


def 解析参数():
    parser = argparse.ArgumentParser(description="全流程自适应推理（端到端）")
    parser.add_argument("--模型", required=True, help="模型路径（必填）")
    parser.add_argument("--量化", choices=["fp16", "4bit"], default="fp16")
    parser.add_argument("--rag", action="store_true", help="启用 RAG（默认关）")
    parser.add_argument("--lora", default=None, help="LoRA 适配器路径（默认无）")
    parser.add_argument("--动态策略", choices=["A", "B", "C"], default="B")
    parser.add_argument("--长上下文", action="store_true", help="启用 λ 步数衰减（长上下文）")
    parser.add_argument("--提示", action="append", default=None,
                        help="提示词文本，可多次传入；默认用 测试提示词 的'开心'3 条")
    parser.add_argument("--最大token", type=int, default=128)
    parser.add_argument("--数据目录", default=r"f:\最终工程架构\数据")
    return parser.parse_args()


def 汇总均值(结果列表, 键):
    """对成功结果取某指标均值（无数据时返回 0）"""
    值列表 = [r.get(键, 0.0) for r in 结果列表 if 键 in r]
    # 命中率口径修复：命中分支不自增总检查数可能 >1，clip 到 [0,1]
    if 键 == "情感命中率":
        值列表 = [max(0.0, min(1.0, v)) for v in 值列表]
    return round(sum(值列表) / len(值列表), 4) if 值列表 else 0.0


def 主流程(args):
    数据目录 = args.数据目录
    os.makedirs(数据目录, exist_ok=True)
    量化值 = 量化映射.get(args.量化, None)

    print(f"[全流程] 模型={args.模型} 量化={args.量化} RAG={args.rag} "
          f"LoRA={args.lora or '无'} 动态策略={args.动态策略} 长上下文={args.长上下文}")
    t0 = time.time()
    框架 = 推理框架.推理框架(
        args.模型, 量化=量化值, rag=args.rag, lora=args.lora,
        动态策略=args.动态策略, 长上下文=args.长上下文)
    print(f"[全流程] 模型加载耗时 {time.time() - t0:.1f}s")

    提示词列表 = args.提示
    if not 提示词列表:
        # 默认 5 维度 × 每维度 4 条（可重复循环）= 20 条
        提示词列表 = []
        for 维度 in 测试提示词:
            列表 = 测试提示词[维度]
            提示词列表.extend([列表[i % len(列表)] for i in range(4)])
    print(f"[全流程] 共 {len(提示词列表)} 条提示词")

    # ── 逐条生成（单条失败记录堆栈继续） ──
    结果列表 = []
    失败记录 = []
    for i, 提示 in enumerate(提示词列表):
        t1 = time.time()
        print(f"[全流程] 生成中 {i + 1}/{len(提示词列表)}: {提示[:40]}")
        try:
            结果 = 框架.生成(提示, max_new_tokens=args.最大token)
            结果["耗时"] = round(time.time() - t1, 2)
            结果["提示词"] = 提示
            结果列表.append(结果)
            print(f"  → 完成 {结果['耗时']}s | 熵 {结果['平均熵']:.4f} | 重复率 {结果['重复率']:.4f}")
        except Exception as e:
            失败记录.append({"提示词": 提示, "错误": str(e), "堆栈": traceback.format_exc()})
            print(f"  → 失败: {e}")

    # ── 汇总指标表 ──
    成功率 = (len(提示词列表) - len(失败记录)) / len(提示词列表) if 提示词列表 else 0.0
    print("-" * 100)
    print(f"{'提示词':<28}{'熵':>8}{'重复率':>8}{'命中率':>8}{'λ':>7}{'γ':>7}{'τ':>7}{'耗时':>7}")
    for 结果 in 结果列表:
        print(f"{结果['提示词'][:26]:<28}{结果['平均熵']:>8.3f}{结果['重复率']:>8.3f}"
              f"{max(0.0, min(1.0, 结果['情感命中率'])):>8.3f}{结果['λ']:>7.3f}{结果['γ']:>7.3f}"
              f"{结果['τ']:>7.3f}{结果['耗时']:>7.1f}")
    print("-" * 100)
    print(f"成功率: {成功率 * 100:.1f}%（成功 {len(结果列表)}/{len(提示词列表)}）")

    # ── 写 JSON 运行记录（UTF-8） ──
    记录 = {
        "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模型": args.模型,
        "量化": args.量化,
        "RAG": args.rag,
        "LoRA": args.lora,
        "动态策略": args.动态策略,
        "长上下文": args.长上下文,
        "最大token": args.最大token,
        "每条结果": 结果列表,
        "失败记录": 失败记录,
        "汇总均值": {
            "平均熵": 汇总均值(结果列表, "平均熵"),
            "重复率": 汇总均值(结果列表, "重复率"),
            "情感命中率": 汇总均值(结果列表, "情感命中率"),
            "平均耗时": 汇总均值(结果列表, "耗时"),
            "步数": 汇总均值(结果列表, "步数"),
        },
        "成功率": round(成功率, 4),
    }
    输出文件 = os.path.join(数据目录, f"全流程_运行记录_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(输出文件, "w", encoding="utf-8") as f:
        json.dump(记录, f, ensure_ascii=False, indent=2)
    print(f"[全流程] 运行记录已写入: {输出文件}")


if __name__ == "__main__":
    try:
        主流程(解析参数())
    except Exception as e:
        print(f"[全流程] 致命错误: {e}")
        traceback.print_exc()
        sys.exit(1)
