# -*- coding: utf-8 -*-
"""环境配置：路径管理、密钥加载、数据目录初始化。

- 本文件所在目录即项目根目录，模块加载时自动把项目根插入 sys.path，
  保证在任何 cwd 下都能 `from 核心模块 import ...` 中文导入。
- 模块加载时自动调用 加载密钥配置() 与 确保数据目录()。
"""
import os
import sys
from pathlib import Path

# ---------- 把本文件所在目录插入 sys.path（保证任何 cwd 下中文 import 可用） ----------
_当前目录 = str(Path(__file__).resolve().parent)
if _当前目录 not in sys.path:
    sys.path.insert(0, _当前目录)


def 项目根目录():
    """返回本文件所在目录（即项目根目录）。"""
    return Path(__file__).resolve().parent


def 数据目录():
    """可写数据目录：打包后为 EXE 所在目录，开发时为项目根。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return 项目根目录()


def 资源目录():
    """只读资源目录：打包后为 PyInstaller 解包目录（_MEIPASS），开发时为项目根。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", 项目根目录()))
    return 项目根目录()


def 加载密钥配置(路径=None):
    """解析 密钥配置.env 到环境变量（格式同 .env，不覆盖已存在的环境变量）。

    优先级：显式路径 > EXE 旁（可写数据目录，用户可直接改）> 打包资源目录（_MEIPASS）。
    """
    if 路径 is not None:
        配置文件 = Path(路径)
    else:
        配置文件 = None
        for 候选 in (数据目录() / "密钥配置.env", 资源目录() / "密钥配置.env"):
            if 候选.exists():
                配置文件 = 候选
                break
    if 配置文件 is None or not 配置文件.exists():
        return
    for 行 in 配置文件.read_text(encoding="utf-8").splitlines():
        行 = 行.strip()
        if not 行 or 行.startswith("#") or "=" not in 行:
            continue
        键, 值 = 行.split("=", 1)
        键 = 键.strip()
        值 = 值.strip().strip('"').strip("'")
        if 键 and 值 and 键 not in os.environ:
            os.environ[键] = 值


def 确保数据目录():
    """创建 数据缓存 下的四个缓存子目录。"""
    根 = 数据目录()
    for 子目录名 in ("音频缓存", "录制视频缓存", "截图缓存", "上传图像缓存"):
        (根 / "数据缓存" / 子目录名).mkdir(parents=True, exist_ok=True)


# ---------- 模块加载时自动初始化 ----------
加载密钥配置()
确保数据目录()
