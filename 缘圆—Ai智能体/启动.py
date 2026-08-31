# -*- coding: utf-8 -*-
"""缘圆智能体自动启动器。

自动定位项目虚拟环境（tts_studio\.venv）并用其中的 Python 运行 启动入口.py，
避免因系统 Python 缺少 fastapi 等依赖而报错。命令行参数原样透传。

用法（与 启动入口.py 一致）：
    python 启动.py                          # 默认 127.0.0.1:8000，内置浏览器窗口
    python 启动.py --浏览器 系统             # 改用系统默认浏览器
    python 启动.py --port 9000
    python 启动.py --list-models             # 查询模型
    python 启动.py --list-voices             # 查询音色
"""
import subprocess
import sys
from pathlib import Path

项目根 = Path(__file__).resolve().parent
入口 = 项目根 / "缘圆智能体" / "启动入口.py"

# 虚拟环境解释器候选（按优先级）。开发期依赖装在 tts_studio\.venv。
候选解释器 = [
    项目根 / "tts_studio" / ".venv" / "Scripts" / "python.exe",
    项目根 / "缘圆智能体" / ".venv" / "Scripts" / "python.exe",
]


def 找解释器():
    for 候选 in 候选解释器:
        if 候选.exists():
            return 候选
    return None


def 主函数():
    if not 入口.exists():
        print(f"[启动器] 未找到入口文件：{入口}", flush=True)
        input("按回车键退出...")
        return 1

    解释器 = 找解释器()
    if 解释器 is None:
        # 未找到虚拟环境时回退当前解释器，能否运行取决于其依赖是否齐全
        解释器 = Path(sys.executable)
        print("[启动器] 未找到虚拟环境，使用当前 Python：", 解释器, flush=True)
    else:
        print("[启动器] 使用虚拟环境 Python：", 解释器, flush=True)

    命令 = [str(解释器), str(入口), *sys.argv[1:]]
    try:
        结果 = subprocess.run(命令, cwd=str(项目根))
        return 结果.returncode
    except FileNotFoundError:
        print(f"[启动器] 启动失败：解释器不存在 {解释器}", flush=True)
        print("提示：请先执行以下命令安装依赖：")
        print('    pip install -r "缘圆智能体\\依赖清单.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple', flush=True)
        input("按回车键退出...")
        return 1


if __name__ == "__main__":
    # 双击运行时防止报错闪退：非交互模式下也等待用户按回车
    try:
        退出码 = 主函数()
        if 退出码 != 0 and not sys.stdin.isatty():
            input("按回车键退出...")
    except KeyboardInterrupt:
        pass
    sys.exit(退出码 if "退出码" in dir() else 0)
