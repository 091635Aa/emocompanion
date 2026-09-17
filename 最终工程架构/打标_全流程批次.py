# -*- coding: utf-8 -*-
"""
打标_全流程批次 — 将最近一次 全流程推理 的运行记录转换为标注任务
================================================================
复用 打标工具.输出标注任务，产出 f:\\打标\\标注任务_全流程批次.json + CSV。

用法：
    python 打标_全流程批次.py
"""
import sys
import os
import json
import glob

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

import 打标工具

数据目录 = r"f:\最终工程架构\数据"
记录文件列表 = sorted(glob.glob(os.path.join(数据目录, "全流程_运行记录_*.json")))
if not 记录文件列表:
    print("[转换] 未找到 全流程_运行记录_*.json")
    sys.exit(1)
记录文件 = 记录文件列表[-1]
记录 = json.load(open(记录文件, encoding="utf-8"))
print(f"[转换] 使用记录: {记录文件}（时间戳 {记录['时间戳']}，成功 {记录['成功率']}）")

结果列表 = []
for 条 in 记录["每条结果"]:
    结果列表.append({
        "提示词": 条["提示词"],
        "回复文本": 条.get("文本", ""),
        "平均熵": round(条.get("平均熵", 0.0), 4),
        "重复率": round(条.get("重复率", 0.0), 4),
        # 命中率口径修复：clip 到 [0,1]
        "情感命中率": round(max(0.0, min(1.0, 条.get("情感命中率", 0.0))), 4),
        "λ": 条.get("λ"),
        "γ": 条.get("γ"),
        "τ": 条.get("τ"),
    })

json路径, csv路径 = 打标工具.输出标注任务(
    结果列表, 输出目录=r"f:\打标", 批次名="全流程批次",
    模型路径=记录["模型"], 量化=记录["量化"])
print(f"[转换] JSON: {json路径}")
print(f"[转换] CSV : {csv路径}")
print(f"[转换] 条目数: {len(结果列表)}")
