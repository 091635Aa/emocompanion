# -*- coding: utf-8 -*-
"""核心模块：缘圆智能体 V2 的 Python 包。

加载本包时自动把项目根目录插入 sys.path，
保证在任何 cwd 下都能 `from 核心模块 import ...` 中文导入。
"""
import sys
from pathlib import Path

_项目根目录 = str(Path(__file__).resolve().parent.parent)
if _项目根目录 not in sys.path:
    sys.path.insert(0, _项目根目录)
