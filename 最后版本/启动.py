# -*- coding: utf-8 -*-
"""
一键启动脚本
============
检查 Python 版本与关键依赖 → 显示横幅 → 启动 FastAPI 服务并自动打开浏览器。

用法：
    python 启动.py
    python 启动.py --port 9000
"""

import argparse
import os
import sys

# 确保项目根在模块搜索路径中（从任意目录运行均可用）
脚本目录 = os.path.dirname(os.path.abspath(__file__))
if 脚本目录 not in sys.path:
    sys.path.insert(0, 脚本目录)

# Windows 控制台统一输出 UTF-8，避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 关键依赖清单：模块名 -> pip 安装名（缺失时打印安装命令，不静默安装）
关键依赖 = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "torch": "torch",
    "transformers": "transformers",
}

清华镜像前缀 = "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "


def 检查Python版本() -> bool:
    """要求 Python 3.10 及以上，不满足时打印提示并返回 False。"""
    主版本, 次版本, _ = sys.version_info[:3]
    if (主版本, 次版本) < (3, 10):
        print(f"[启动] 错误：需要 Python 3.10 及以上，当前为 {主版本}.{次版本}")
        print("[启动] 请安装更高版本的 Python 后重试。")
        return False
    print(f"[启动] Python {主版本}.{次版本} 版本检查通过。")
    return True


def 检查关键依赖() -> bool:
    """检查关键依赖是否可导入，缺失时打印安装命令（不静默安装）。"""
    全部就绪 = True
    for 模块名, 包名 in 关键依赖.items():
        try:
            __import__(模块名)
        except ImportError:
            全部就绪 = False
            print(f"[启动] 缺少依赖：{模块名}")
            print(f"        安装命令：{清华镜像前缀}{包名}")
    if 全部就绪:
        print("[启动] 关键依赖检查通过。")
    else:
        print("[启动] 提示：请先按上述命令安装缺失依赖，再重新运行本脚本。")
    return 全部就绪


def 显示横幅(版本: str) -> None:
    """输出 UTF-8 中文横幅（项目名 + 版本）。"""
    横幅 = f"""
====================================================
     一体化全流程AI应用  v{版本}
     数据 → 打标 → 日记 → 微调 → 达标 → 推理
     一键启动 · 本机测试版
====================================================
"""
    print(横幅)


def 解析参数() -> argparse.Namespace:
    """解析命令行参数，支持 --port 覆盖默认端口。"""
    解析器 = argparse.ArgumentParser(description="一体化全流程AI应用 一键启动")
    解析器.add_argument("--port", type=int, default=None, help="覆盖默认端口（默认取配置中的端口）")
    return 解析器.parse_args()


def 主入口() -> None:
    """主流程：版本检查 → 依赖检查 → 横幅 → 启动服务。"""
    参数 = 解析参数()

    if not 检查Python版本():
        sys.exit(1)
    if not 检查关键依赖():
        # 依赖缺失时退出并提示安装命令（不静默安装）
        sys.exit(1)

    from 核心引擎 import 配置管理
    from 核心引擎.主服务 import 启动服务

    版本 = 配置管理.获取配置项("系统.版本", "0.1.0")
    显示横幅(版本)

    端口 = 参数.port or 配置管理.获取配置项("系统.默认端口", 8765)
    print(f"[启动] 服务地址：http://127.0.0.1:{端口}/前端/页面入口.html")
    启动服务(端口=端口)


if __name__ == "__main__":
    主入口()
