# -*- coding: utf-8 -*-
"""
日志模块

提供统一的日志写入功能：控制台打印 + 追加写入 日志\系统日志-YYYYMMDD.log。
"""

import os
import sys
from datetime import datetime

# 项目根目录（本文件位于 业务逻辑层\ 下，上一级即项目根目录）
项目根目录 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if 项目根目录 not in sys.path:
    sys.path.insert(0, 项目根目录)

日志目录 = os.path.join(项目根目录, "日志")


def 获取日志文件路径():
    """返回当日日志文件的完整路径（日志\系统日志-YYYYMMDD.log）。"""
    os.makedirs(日志目录, exist_ok=True)
    今日日期 = datetime.now().strftime("%Y%m%d")
    return os.path.join(日志目录, "系统日志-{}.log".format(今日日期))


def 写日志(消息, 级别="信息"):
    """
    写入一条日志：控制台打印并追加到当日日志文件。

    参数:
        消息: 日志内容文本
        级别: 日志级别（信息/警告/错误）
    """
    时间戳 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    日志行 = "[{}][{}] {}".format(时间戳, 级别, 消息)
    print(日志行)
    try:
        with open(获取日志文件路径(), "a", encoding="utf-8") as 文件对象:
            文件对象.write(日志行 + "\n")
    except OSError as 异常:
        # 日志写入失败不影响主流程，仅提示
        print("[警告] 日志文件写入失败：{}（{}）".format(消息, 异常))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    写日志("日志模块自测：这是一条测试日志")
    print("日志文件：", 获取日志文件路径())
