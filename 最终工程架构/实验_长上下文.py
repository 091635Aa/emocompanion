# -*- coding: utf-8 -*-
"""
实验_长上下文 — 3B Q4 上对比 λ 固定 vs λ 步数衰减（2048 tokens）
================================================================
- 配置 1 = 无衰减（λ=0.10 固定）
- 配置 2 = λ 步数衰减（λ=0.10, 衰减起始 256, 终点比例 0.3）
- 每条配置 3 条提示词（开心/悲伤/中性 各 1）× 1 次 × 2048 tokens
- 按四区间（0-256/256-512/512-1024/1024-2048）统计平均熵与重复率
  （熵列表切片 + token 切片；token 列表由 回响引擎 返回的 生成 ids 提供）
- 输出 数据\\长上下文_稳定_2048.csv（列：配置, 提示词, 区间, 平均熵, 重复率）
  + 控制台打印 G2 结论（衰减后 1024-2048 区间熵较 0-256 下降 <15% 且重复率 <0.3 与否）
- 生成期间 print(flush=True) 每 256 步报告进度，便于后台监控

说明：配置 1 通过 运行回响_步数衰减 的退化参数（衰减起始步=10**9）实现——
调度函数恒返回 lam，数值等价于固定 λ。

用法：
    python 实验_长上下文.py [--模型 ...] [--量化 4bit|fp16]
"""
import sys
import os
import csv
import argparse

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

import 推理框架
import 回响引擎
from echo_common import 测试提示词, 计算重复率

默认模型 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-3B-Instruct"
数据目录 = r"f:\最终工程架构\数据"
输出文件 = os.path.join(数据目录, "长上下文_稳定_2048.csv")

最大token = 2048
区间定义 = [("0-256", 0, 256), ("256-512", 256, 512),
          ("512-1024", 512, 1024), ("1024-2048", 1024, 2048)]
# 配置：dict（名称, 衰减起始步, 终点λ比例, γ, 保留策略, 滑动窗口, repetition_penalty）
配置列表 = [
    {"名称": "配置1_无衰减", "衰减起始步": 10 ** 9, "终点λ比例": 0.3, "γ": 0.08,
     "保留策略": "衰减", "滑动窗口": 3, "repetition_penalty": 1.0},
    {"名称": "配置2_步数衰减", "衰减起始步": 256, "终点λ比例": 0.3, "γ": 0.08,
     "保留策略": "衰减", "滑动窗口": 3, "repetition_penalty": 1.0},
    {"名称": "配置3_激进稳定", "衰减起始步": 128, "终点λ比例": 0.1, "γ": 0.15,
     "保留策略": "滑动窗口", "滑动窗口": 2, "repetition_penalty": 1.2},
    {"名称": "配置4_更激进衰减", "衰减起始步": 128, "终点λ比例": 0.05, "γ": 0.15,
     "保留策略": "滑动窗口", "滑动窗口": 2, "repetition_penalty": 1.2},
]
提示词集 = {
    "开心": 测试提示词["开心"][0],
    "悲伤": 测试提示词["悲伤"][0],
    "中性": 测试提示词["中性"][0],
}


def 统计区间(熵列表, token列表):
    """按四区间切片统计平均熵与重复率，返回 [(区间名, 平均熵, 重复率), ...]"""
    结果 = []
    for 区间名, 起, 止 in 区间定义:
        熵切片 = 熵列表[起:止]
        token切片 = token列表[起:止]
        平均熵 = sum(熵切片) / len(熵切片) if 熵切片 else 0.0
        重复率 = 计算重复率(token切片) if len(token切片) >= 8 else 0.0
        结果.append((区间名, 平均熵, 重复率))
    return 结果


def 主流程(args):
    os.makedirs(数据目录, exist_ok=True)
    量化值 = None if args.量化 == "fp16" else args.量化

    print(f"[实验] 模型={args.模型} 量化={args.量化}")
    框架 = 推理框架.推理框架(args.模型, 量化=量化值, rag=False, lora=None,
                          动态策略="A", 长上下文=False)
    模型 = 框架.model
    tokenizer = 框架.tokenizer
    过滤器 = 框架._获取情感过滤器()
    print(f"[实验] 基准参数 λ={框架.λ基准} γ={框架.γ基准} τ={框架.τ基准}")
    assert 框架.λ基准 == 0.10 and 框架.γ基准 == 0.08, \
        "基准参数必须为 λ=0.10, γ=0.08（请确认模型为 Qwen2.5-3B）"

    # 只跑指定配置（--仅配置）；缺省全部。避免重跑既有配置1/2
    执行列表 = [cfg for cfg in 配置列表
             if args.仅配置 is None or cfg["名称"] == args.仅配置]
    if not 执行列表:
        print(f"[实验] 无匹配配置: {args.仅配置}")
        return
    判定配置名 = 执行列表[-1]["名称"]  # G2 判定最后一个执行的配置

    # CSV 追加：保留既有配置1/2 行，只追加本次运行的新配置行
    行列表 = [["配置", "提示词", "区间", "平均熵", "重复率"]]
    if os.path.exists(输出文件):
        try:
            with open(输出文件, encoding="utf-8-sig") as f:
                已有行 = list(csv.reader(f))
            if len(已有行) > 1 and 已有行[0] == 行列表[0]:
                行列表.extend(已有行[1:])
                print(f"[实验] 已保留既有 CSV 数据 {len(已有行) - 1} 行")
        except Exception as e:
            print(f"[实验] 读取既有 CSV 失败（忽略，从头写入）: {e}")

    配置统计 = {}  # 配置名 -> {维度: {区间名: (平均熵, 重复率)}}
    for cfg in 执行列表:
        配置名 = cfg["名称"]
        print(f"\n[实验] ============ {配置名}（λ=0.10, γ={cfg['γ']}, 起始步={cfg['衰减起始步']}, "
              f"终点比例={cfg['终点λ比例']}, 保留策略={cfg['保留策略']}, "
              f"滑动窗口={cfg['滑动窗口']}, rp={cfg['repetition_penalty']}）============")
        for 维度, 提示 in 提示词集.items():
            print(f"[实验] 提示词（{维度}）: {提示}")

            def 进度回调(当前步数, 总数, 配置名=配置名, 维度=维度):
                # 每 256 步一行，flush 便于后台监控
                if 当前步数 % 256 == 0:
                    print(f"  [进度] {配置名} {维度} 已生成 {当前步数}/{总数} 步", flush=True)

            结果 = 回响引擎.运行回响_步数衰减(
                模型, tokenizer, 提示,
                lam=0.10, gamma=cfg["γ"], 情感过滤器实例=过滤器,
                保留策略=cfg["保留策略"], 滑动窗口=cfg["滑动窗口"],
                max_new_tokens=最大token,
                归一化基准=框架.归一化基准,
                repetition_penalty=cfg["repetition_penalty"],
                衰减起始步=cfg["衰减起始步"], 终点λ比例=cfg["终点λ比例"],
                progress_callback=进度回调)

            熵列表 = 结果["熵列表"]
            token列表 = 结果["token列表"]
            区间统计 = 统计区间(熵列表, token列表)
            print(f"  [完成] 总步数={结果['步数']} 整体平均熵={结果['平均熵']:.4f} 整体重复率={结果['重复率']:.4f}")
            for 区间名, 平均熵, 重复率 in 区间统计:
                行列表.append([配置名, f"{维度}:{提示}", 区间名,
                               round(平均熵, 4), round(重复率, 4)])
                print(f"    区间 {区间名:<10} 平均熵 {平均熵:.4f}  重复率 {重复率:.4f}")
            配置统计.setdefault(配置名, {})[维度] = \
                {区间名: (平均熵, 重复率) for 区间名, 平均熵, 重复率 in 区间统计}

    # 写 CSV（保留既有行 + 本次新行）
    with open(输出文件, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(行列表)
    print(f"\n[实验] 区间统计已写入: {输出文件}（共 {len(行列表) - 1} 行数据）")

    # ── G2 结论：判定配置 衰减后 1024-2048 区间熵较 0-256 下降 <15% 且重复率 <0.3 ──
    print("-" * 72)
    print(f"[G2] 长上下文稳定性判定（{判定配置名}）")
    if 判定配置名 not in 配置统计:
        print(f"[G2] 判定配置 {判定配置名} 无本次运行数据，跳过判定")
    else:
        全部满足 = True
        for 维度, 统计 in 配置统计[判定配置名].items():
            if "0-256" not in 统计 or "1024-2048" not in 统计:
                continue
            熵0, _ = 统计["0-256"]
            熵3, 重复3 = 统计["1024-2048"]
            下降比例 = (熵0 - 熵3) / 熵0 * 100 if 熵0 > 0 else 0.0
            满足 = 下降比例 < 15 and 重复3 < 0.3
            全部满足 = 全部满足 and 满足
            print(f"  {维度}: 0-256熵={熵0:.4f} → 1024-2048熵={熵3:.4f}（下降 {下降比例:+.2f}%）"
                  f" | 1024-2048重复率={重复3:.4f} | {'达标' if 满足 else '未达标'}")
        print(f"[G2] 结论：{判定配置名} 1024-2048 熵较 0-256 下降 <15% 且重复率 <0.3："
              f"{'达成' if 全部满足 else '未达成'}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="长上下文稳定性实验（3B Q4，2048 tokens）")
    parser.add_argument("--模型", default=默认模型)
    parser.add_argument("--量化", choices=["fp16", "4bit"], default="4bit")
    parser.add_argument("--仅配置", default=None,
                        help="只运行指定配置名（如 配置3_激进稳定）；缺省运行全部")
    args = parser.parse_args()
    主流程(args)
