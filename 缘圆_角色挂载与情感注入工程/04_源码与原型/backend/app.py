# -*- coding: utf-8 -*-
"""缘圆情感引擎 一键启动器

功能:
  1. 自动定位/创建项目虚拟环境（.venv）
  2. 若缺依赖则用清华镜像安装（fastapi / uvicorn / llama-cpp-python / psutil / numpy / jinja2）
  3. 设置 llama.cpp 动态库 PATH，启动 FastAPI 后端服务
  4. 可选: --open 自动打开浏览器 / --port 覆盖端口

用法:
  python app.py               # 默认 127.0.0.1:8000
  python app.py --port 9000
  python app.py --open        # 启动后自动打开浏览器
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent          # backend/
ENGINE_ROOT = PROJECT_ROOT.parent                        # 04_源码与原型/
VENV = ENGINE_ROOT / ".venv"
SERVER = PROJECT_ROOT / "server.py"
PYTHON = sys.executable

# 清华镜像（用户环境要求）
MIRROR_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 服务端运行所需的依赖（模块 -> pip 包）
# 注: llama_cpp 由 pykits 提供(CUDA 版)，通过 PYTHONPATH 引入，不在 venv 内重复安装
DEPS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "psutil": "psutil",
    "jinja2": "jinja2",
    "numpy": "numpy",
}


def _run(cmd, cwd=None, echo=True):
    if echo:
        print(f"[app] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, shell=False)


def _venv_python():
    return VENV / "Scripts" / "python.exe"


def ensure_venv():
    """无虚拟环境则创建，缺依赖则安装"""
    if _venv_python().exists():
        print(f"[app] 虚拟环境已存在: {VENV}", flush=True)
    else:
        print(f"[app] 创建虚拟环境: {VENV}", flush=True)
        VENV.parent.mkdir(parents=True, exist_ok=True)
        r = _run([PYTHON, "-m", "venv", str(VENV)])
        if r.returncode != 0:
            sys.exit("创建虚拟环境失败")

    py = _venv_python()
    # 检查依赖：逐模块检测，每行一个缺失名
    missing = []
    code = (
        "import importlib.util,sys\n"
        "mods=['fastapi','uvicorn','psutil','jinja2','numpy']\n"
        "for m in mods:\n"
        "    if importlib.util.find_spec(m) is None:\n"
        "        sys.stdout.write(m+'\\n')\n"
    )
    try:
        out = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=60)
        missing = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception as e:
        print(f"[app] 依赖探测失败: {e}", flush=True)
        missing = list(DEPS.keys())

    if missing:
        pkgs = " ".join(DEPS[m] for m in missing if m in DEPS)
        if not pkgs:
            pkgs = " ".join(missing)
        print(f"[app] 安装缺失依赖: {pkgs}", flush=True)
        r = _run([str(py), "-m", "pip", "install", "-i", MIRROR_URL,
                  "--timeout", "300", "--retries", "10", *pkgs.split()])
        if r.returncode != 0:
            sys.exit("依赖安装失败")
    else:
        print("[app] 依赖齐全", flush=True)
    return py


def main():
    p = argparse.ArgumentParser(description="缘圆情感引擎 启动器")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    p.add_argument("--skip-deps", action="store_true", help="跳过依赖检查/安装")
    a = p.parse_args()

    py = _venv_python() if a.skip_deps else ensure_venv()

    # llama.cpp / numpy 的 CUDA 版由 pykits 提供
    torch_lib = Path(r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib")
    llamacpp_pkg = ENGINE_ROOT.parent / "pykits" / "llamacpp"
    llamacpp_lib = llamacpp_pkg / "llama_cpp" / "lib"
    env = dict(os.environ)
    env["PATH"] = str(llamacpp_lib) + os.pathsep + str(torch_lib) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + str(llamacpp_pkg) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"[app] 使用解释器: {py}", flush=True)
    print(f"[app] 启动服务 http://{a.host}:{a.port}", flush=True)

    cmd = [str(py), str(SERVER), "--host", a.host, "--port", str(a.port)]
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
        return proc.returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
