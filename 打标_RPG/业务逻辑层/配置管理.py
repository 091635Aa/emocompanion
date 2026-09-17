# -*- coding: utf-8 -*-
"""
配置管理模块

负责系统配置文件的读取、解析与保存，以及工作目录下相对路径的拼接与自动创建。
"""

import json
import os
import sys

# 项目根目录（本文件位于 业务逻辑层\ 下，上一级即项目根目录）
项目根目录 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if 项目根目录 not in sys.path:
    sys.path.insert(0, 项目根目录)

# 配置文件固定位于项目根目录的"配置"文件夹中
配置文件路径 = os.path.abspath(
    os.path.join(项目根目录, "配置", "系统配置.json")
)


def 读取配置():
    """读取配置文件并解析为字典，返回完整配置数据。"""
    with open(配置文件路径, "r", encoding="utf-8") as 文件对象:
        return json.load(文件对象)


def 获取配置():
    """返回完整的系统配置字典。"""
    return 读取配置()


def 获取路径(键名):
    """
    根据键名拼接出完整路径，并自动创建不存在的目录。

    规则：
    - 若配置中的路径为绝对路径（如以盘符开头），则直接使用；
    - 若为相对路径，则与工作目录拼接后返回；
    - 目录不存在时自动创建。
    """
    配置 = 读取配置()
    路径配置 = 配置["路径"]
    工作目录 = 路径配置["工作目录"]
    相对路径 = 路径配置[键名]
    if os.path.isabs(相对路径):
        完整路径 = 相对路径
    else:
        完整路径 = os.path.join(工作目录, 相对路径)
    os.makedirs(完整路径, exist_ok=True)
    return 完整路径


def 保存配置(新配置):
    """将新的配置字典写回配置文件（ensure_ascii=False, indent=2）。"""
    配置目录 = os.path.dirname(配置文件路径)
    os.makedirs(配置目录, exist_ok=True)
    with open(配置文件路径, "w", encoding="utf-8") as 文件对象:
        json.dump(新配置, 文件对象, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 直接运行本模块时，打印配置解析结果，便于自检
    sys.stdout.reconfigure(encoding="utf-8")
    完整配置 = 获取配置()
    print("配置解析成功，项目名称：", 完整配置["项目名称"])
    print("源素材目录：", 获取路径("源素材目录"))
    print("分割素材目录：", 获取路径("分割素材目录"))
