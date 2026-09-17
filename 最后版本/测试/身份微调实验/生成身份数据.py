# -*- coding: utf-8 -*-
"""
生成身份微调小样本数据集
========================
Task 7：身份微调可行性验证 — 数据生成脚本（纯标准库，无任何第三方依赖）。

产出（相对项目根 j:\\最后版本！）：
    数据/微调数据包/身份微调_小样本.jsonl   实验组：~60 条（身份指令类 + 自然口语语料类）
    数据/微调数据包/对照数据集.jsonl       对照组：同数量常规 assistant 式指令响应

数据格式：每行一个 JSON 对象 {"instruction": ..., "response": ...}，
UTF-8 编码，写入前统一清洗控制字符与换行，保证单行合法 JSON、无乱码。

用法：
    python 测试\\身份微调实验\\生成身份数据.py
"""

import json
import re
import sys
from pathlib import Path

# Windows 控制台统一输出 UTF-8，避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 项目根：本脚本位于 <项目根>/测试/身份微调实验/ 下，向上三级
项目根 = Path(__file__).resolve().parent.parent.parent
实验数据文件 = 项目根 / "数据" / "微调数据包" / "身份微调_小样本.jsonl"
对照数据文件 = 项目根 / "数据" / "微调数据包" / "对照数据集.jsonl"


def 清洗文本(文本: str) -> str:
    """清洗文本：剔除控制字符，把换行/制表符折叠为空格，压缩多余空白。

    目标：保证写入 JSONL 的每个字段是单行、无乱码、无不可见字符的干净文本。
    """
    # 1) 删除控制字符（保留 \\t \\n \\r，随后统一折叠为空格）
    清理 = []
    for 字符 in 文本:
        码点 = ord(字符)
        if 码点 < 32 and 码点 not in (9, 10, 13):
            continue
        if 码点 == 127:  # DEL
            continue
        清理.append(字符)
    文本 = "".join(清理)
    # 2) 换行/制表符统一折叠为空格，保证单行
    文本 = 文本.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # 3) 压缩连续空白并去除首尾空白
    文本 = re.sub(r"[ ]{2,}", " ", 文本).strip()
    return 文本


def 写入jsonl(路径: Path, 条目列表: list) -> int:
    """把条目列表写入 JSONL（UTF-8、ensure_ascii=False），返回实际写入条数。"""
    路径.parent.mkdir(parents=True, exist_ok=True)
    写入数 = 0
    with open(路径, "w", encoding="utf-8") as 文件:
        for 条目 in 条目列表:
            if not isinstance(条目, dict) or "instruction" not in 条目 or "response" not in 条目:
                continue
            干净条目 = {键: 清洗文本(str(值)) for 键, 值 in 条目.items()}
            if not 干净条目["instruction"] or not 干净条目["response"]:
                continue
            文件.write(json.dumps(干净条目, ensure_ascii=False) + "\n")
            写入数 += 1
    return 写入数
