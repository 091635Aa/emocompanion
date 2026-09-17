# -*- coding: utf-8 -*-
"""EmoCompanion · 最终归档 —— 独立一键入口

放在 07_最终归档/ 下，独立于 backend/ 工作目录。
功能:
  1. 自动定位项目虚拟环境/依赖（无则引导）
  2. 自动扫描 MOD 路径下的模型 与 扩展包(角色包)
  3. 交互菜单:
     [1] 启动后端服务 (FastAPI)
     [2] CLI 对话（可定向 模型+扩展包）
     [3] 查看模型/扩展包清单
     [4] 打开接口文档/归档文档
  4. 支持命令行参数定向启动: app.py --serve / app.py --chat / app.py --list

用法:
  python app.py                  # 交互菜单
  python app.py --list           # 扫描清单
  python app.py --chat "你好呀"   # 直接对话
  python app.py --serve --port 8000 --model Qwen3-4B-Q4_K_M --pack default
  python app.py --open-docs      # 打开接口文档
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # 07_最终归档
ENGINE_ROOT = HERE.parent / "04_源码与原型"
BACKEND = ENGINE_ROOT / "backend"
PACK_DIR = ENGINE_ROOT / "data" / "role_pack"
PYTHON = sys.executable

# 桌面路径（放接口文档）
DESKTOP = Path(r"D:\Desktop")
DOC_API = HERE / "接口文档.md"
DOC_ARCHIVE = HERE / "最终归档.md"


def _banner():
    print("=" * 52)
    print("   EmoCompanion情感引擎 · 最终归档   v1.0")
    print("   角色挂载 + 情感注入 + llama.cpp 加速")
    print("=" * 52)


def ensure_venv():
    """确保 backend 的 .venv 存在且依赖齐全（复用 backend/app.py 逻辑的轻量版）"""
    venv = ENGINE_ROOT / ".venv"
    py = venv / "Scripts" / "python.exe"
    if not py.exists():
        print("[app] 创建虚拟环境 ...")
        r = subprocess.run([str(PYTHON), "-m", "venv", str(venv)])
        if r.returncode != 0:
            print("[app] 虚拟环境创建失败")
            return PYTHON
    # 关键依赖检查
    code = ("import importlib.util,sys\n"
            "mods=['fastapi','uvicorn','psutil','jinja2','numpy']\n"
            "for m in mods:\n"
            "  if importlib.util.find_spec(m) is None: sys.stdout.write(m+'\\n')\n")
    try:
        out = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=60)
        missing = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        missing = ["fastapi", "uvicorn", "psutil", "jinja2", "numpy"]
    if missing:
        pkgs = " ".join(missing)
        print(f"[app] 安装依赖(清华源): {pkgs}")
        r = subprocess.run([str(py), "-m", "pip", "install", "-i",
                            "https://pypi.tuna.tsinghua.edu.cn/simple",
                            "--timeout", "300", "--retries", "10", *pkgs.split()])
        if r.returncode != 0:
            print("[app] 依赖安装失败")
            return PYTHON
    return py


def _env(py):
    """构造子进程环境: 注入 llama.cpp CUDA 版 + torch 运行时"""
    env = dict(os.environ)
    llamacpp_pkg = HERE.parent.parent / "pykits" / "llamacpp"
    torch_lib = Path(r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib")
    env["PATH"] = str(llamacpp_pkg / "llama_cpp" / "lib") + os.pathsep + \
                  str(torch_lib) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(BACKEND) + os.pathsep + str(llamacpp_pkg) + \
                        os.pathsep + env.get("PYTHONPATH", "")
    return env


def cmd_list(py, env):
    print("\n[app] 扫描模型与扩展包 ...")
    r = subprocess.run([str(py), str(BACKEND / "cli.py"), "list"], env=env, cwd=str(BACKEND))
    return r.returncode


def cmd_chat(py, env, text, model, pack, interactive=False):
    argv = [str(py), str(BACKEND / "cli.py"), "chat"]
    if interactive:
        argv.append("-i")
    if model:
        argv += ["-m", model]
    if pack:
        argv += ["-p", pack]
    if text:
        argv.append(text)
    return subprocess.run(argv, env=env, cwd=str(BACKEND)).returncode


def cmd_serve(py, env, port, model, pack):
    argv = [str(py), str(BACKEND / "cli.py"), "serve", "--port", str(port)]
    if model:
        argv += ["-m", model]
    if pack:
        argv += ["-p", pack]
    return subprocess.run(argv, env=env, cwd=str(BACKEND)).returncode


def open_docs():
    """接口文档放到桌面并打开"""
    import shutil
    if DOC_API.exists():
        dst = DESKTOP / DOC_API.name
        shutil.copy2(DOC_API, dst)
        print(f"[app] 接口文档已复制到桌面: {dst}")
        os.startfile(dst)
    else:
        print(f"[app] 未找到接口文档: {DOC_API}")


def menu(py, env):
    while True:
        print("\n请选择操作:")
        print("  [1] 启动后端服务 (FastAPI)")
        print("  [2] CLI 对话（交互模式）")
        print("  [3] 查看模型/扩展包清单")
        print("  [4] 打开接口文档（并复制到桌面）")
        print("  [0] 退出")
        try:
            c = input("选择> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if c == "1":
            cmd_serve(py, env, 8000, None, None)
        elif c == "2":
            cmd_chat(py, env, None, None, None, interactive=True)
        elif c == "3":
            cmd_list(py, env)
        elif c == "4":
            open_docs()
        elif c == "0":
            break
        else:
            print("[app] 无效选择")


def main():
    p = argparse.ArgumentParser(description="EmoCompanion · 最终归档入口")
    p.add_argument("--list", action="store_true", help="扫描模型/扩展包")
    p.add_argument("--chat", nargs="?", const="-i", metavar="TEXT", help="CLI 对话（默认交互模式，可带文本）")
    p.add_argument("--serve", action="store_true", help="启动后端服务")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", help="定向模型文件名")
    p.add_argument("--pack", help="定向扩展包名")
    p.add_argument("--open-docs", action="store_true", help="打开接口文档")
    a = p.parse_args()

    _banner()
    py = ensure_venv()
    env = _env(py)
    print(f"[app] 解释器: {py}")

    if a.open_docs:
        open_docs()
    elif a.list:
        cmd_list(py, env)
    elif a.chat is not None:
        if a.chat == "-i":
            cmd_chat(py, env, None, a.model, a.pack, interactive=True)
        else:
            cmd_chat(py, env, a.chat, a.model, a.pack)
    elif a.serve:
        cmd_serve(py, env, a.port, a.model, a.pack)
    else:
        menu(py, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
