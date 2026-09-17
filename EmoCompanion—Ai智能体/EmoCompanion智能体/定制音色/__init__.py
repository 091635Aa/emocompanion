# -*- coding: utf-8 -*-
"""定制音色：声音库（数据集 / 合成输出）等资源目录。

包初始化时把项目根目录插入 sys.path，保证任意 cwd 下中文导入可用。
"""
import sys
from pathlib import Path

_项目根 = str(Path(__file__).resolve().parent.parent)
if _项目根 not in sys.path:
    sys.path.insert(0, _项目根)
