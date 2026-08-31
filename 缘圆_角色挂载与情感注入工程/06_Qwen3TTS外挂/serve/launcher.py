# -*- coding: utf-8 -*-
"""缘圆 前端后端一体化 一键启动器

功能:
  1. 定位/创建项目 venv，缺依赖用清华镜像安装（fastapi/uvicorn/psutil/numpy）
  2. 探测 TTS 推理依赖（torch/qwen_tts/peft）；缺失则以 --skip-tts 启动文本+前端，并给出安装提示
  3. 起 uvicorn 统一服务（默认 127.0.0.1:8070），可选 --open 浏览器；文本引擎默认代理 :8000

用法:
  python launcher.py [--port 8070] [--open] [--text-base http://127.0.0.1:8000]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

SERVE = Path(__file__).resolve().parent                 # serve/
PROJECT = SERVE.parent                                   # 06_Qwen3TTS外挂/
ENGINE_ROOT = PROJECT.parent                            # 缘圆_角色挂载与情感注入工程/
VENV = ENGINE_ROOT / "04_源码与原型" / ".venv"            # 复用现有 venv（含 fastapi/uvicorn/numpy）
SERVER = SERVE / "unified_server.py"
PYTHON = sys.executable
MIRROR_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

DEPS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "psutil": "psutil",
    "numpy": "numpy",
}
# 需要 'torch qwen-tts peft' 才能实现真实 TTS；缺失时降级为 --skip-tts
TTS_DEPS = ["torch", "qwen_tts", "peft", "soundfile", "librosa"]


def _run(cmd, cwd=None):
    print(f"[launcher] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, shell=False)


def venv_python():
    return VENV / "Scripts" / "python.exe"


def ensure_venv():
    py = venv_python()
    if not py.exists():
        print(f"[launcher] 创建虚拟环境: {VENV}", flush=True)
        VENV.parent.mkdir(parents=True, exist_ok=True)
        r = _run([PYTHON, "-m", "venv", str(VENV)])
        if r.returncode != 0:
            sys.exit("创建虚拟环境失败")

    code = ("import importlib.util,sys\n"
            "mods=['fastapi','uvicorn','psutil','numpy']\n"
            "for m in mods:\n"
            "    if importlib.util.find_spec(m) is None: sys.stdout.write(m+'\\n')\n")
    out = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=120)
    missing = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if missing:
        pkgs = " ".join(DEPS[m] for m in missing if m in DEPS)
        print(f"[launcher] 安装缺失依赖: {pkgs}", flush=True)
        r = _run([str(py), "-m", "pip", "install", "-i", MIRROR_URL,
                  "--timeout", "300", "--retries", "10", *pkgs.split()])
        if r.returncode != 0:
            sys.exit("依赖安装失败")
    return py


def probe_tts(py) -> bool:
    """全新架构：TTS 只需 GGUF 资产齐全，不再依赖 torch/qwen_tts/peft。"""
    assets = {
        r"D:\AI情感\pykits\llama-cpp-bin\llama-tts.exe": "llama-tts.exe",
        r"D:\AI情感\pykits\models\Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf": "backbone GGUF",
        r"D:\AI情感\pykits\models\mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf": "mmproj GGUF",
        r"D:\AI情感\pykits\models\voice_lora_qwen3tts.gguf": "voice LoRA",
        r"D:\AI情感\pykits\models\emotion_lora_qwen3tts.gguf": "emotion LoRA",
    }
    miss = [n for p, n in assets.items() if not os.path.isfile(p)]
    if miss:
        print(f"[launcher] GGUF TTS 资产缺失(将降级禁用 TTS): {', '.join(miss)}", flush=True)
        return False
    # gguf 后端需要 numpy 读 wav（unified_server 写 wav），launcher 已保证 numpy
    return True


def main():
    ap = argparse.ArgumentParser(description="缘圆前端后端一体化 启动器")
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--text-base", default="http://127.0.0.1:8000")
    ap.add_argument("--skip-deps", action="store_true")
    ap.add_argument("--tts-policy", default="bfloat16", help="bfloat16|float16|int8（int8 实测未降显存，勿依赖）")
    ap.add_argument("--vram-budget-gb", type=float, default=8.0, help="TTS 显存预算告警阈值(GB)")
    a = ap.parse_args()

    py = venv_python() if a.skip_deps else ensure_venv()
    tts_ok = probe_tts(py)

    env = dict(os.environ)
    env["YY_TTS_POLICY"] = a.tts_policy

    cmd = [str(py), str(SERVER), "--host", a.host, "--port", str(a.port),
           "--text-base", a.text_base]
    if not tts_ok:
        cmd.append("--skip-tts")

    url = f"http://{a.host}:{a.port}"
    print(f"[launcher] 启动服务 {url}  文本引擎={a.text_base}  TTS={'启用' if tts_ok else '跳过'}", flush=True)
    if a.open:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        proc = subprocess.run(cmd, cwd=str(SERVE), env=env)
        return proc.returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())