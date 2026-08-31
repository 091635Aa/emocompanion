# -*- coding: utf-8 -*-
"""EmoCompanion CLI 交互工具 —— 模型/扩展包扫描 + 定向启动 + 多请求格式

功能:
  scan       扫描 MOD 路径下的模型(GGUF) 与扩展包(角色包)
  chat       与模型对话（交互式 / 单行 / JSON）
  serve      启动 FastAPI 后端服务（定向模型+扩展包）
  list       列出全部模型与扩展包

请求格式支持（chat）:
  1) 空格分隔:    EmoCompanioncli.py chat 晚上好呀 今天开心吗
  2) 引号文本:    EmoCompanioncli.py chat "晚上好呀，今天开心吗"
  3) JSON 格式:   EmoCompanioncli.py chat '{"messages":[{"role":"user","content":"hi"}],"max_new":64}'
  4) 交互模式:    EmoCompanioncli.py chat -i         （多轮持续对话）
  5) 定向启动:    EmoCompanioncli.py chat -m Qwen3-4B-Q4_K_M.gguf -p default 晚上好呀

用法:
  python cli.py scan
  python cli.py list
  python cli.py chat "你好呀"
  python cli.py chat -i
  python cli.py chat -m <model> -p <pack> "内容"
  python cli.py serve --port 8000 --model <model> --pack <pack>
"""
import argparse
import json
import os
import sys
from pathlib import Path

# ---------------- 路径配置 ----------------
HERE = Path(__file__).resolve().parent
ENGINE_ROOT = HERE.parent                                   # 04_源码与原型
BACKEND = ENGINE_ROOT / "backend"
PROJ_ROOT = ENGINE_ROOT.parent                              # EmoCompanion_角色挂载与情感注入工程

# MOD 路径（模型扫描源，可扩展）
MOD_PATHS = [
    Path(r"d:\llama_models"),                               # GGUF 主目录
    Path(r"d:\AI情感\pykits\models"),                      # GGUF 备用
    Path(r"d:\AI情感\模型空间"),                            # safetensors 目录
    Path(r"d:\AI情感\微调文本\models"),                     # 微调模型
]
# 扩展包路径（角色包）
PACK_DIR = ENGINE_ROOT / "data" / "role_pack"
# 后端模块路径
LLAMACPP_DIR = Path(r"d:\AI情感\pykits\llamacpp")
TORCH_LIB = Path(r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib")


def setup_env():
    """注入 llama.cpp CUDA 版 与 torch CUDA 运行时 PATH/PYTHONPATH"""
    os.environ["PATH"] = str(LLAMACPP_DIR / "llama_cpp" / "lib") + os.pathsep + \
                         str(TORCH_LIB) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONPATH"] = str(BACKEND) + os.pathsep + str(LLAMACPP_DIR) + \
                               os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(BACKEND))
    sys.path.insert(0, str(LLAMACPP_DIR))


# ---------------- 扫描 ----------------
def scan_models():
    """扫描 MOD 路径下所有模型，返回 [{name, path, type, size_gb}]"""
    found = {}
    for p in MOD_PATHS:
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.is_file():
                if f.suffix.lower() == ".gguf":
                    found[f.stem] = {"name": f.stem, "path": str(f), "type": "gguf",
                                     "size_gb": round(f.stat().st_size / 1e9, 2)}
                elif f.suffix.lower() == ".safetensors":
                    d = f.parent
                    # 汇总目录内全部分片大小
                    total = sum(x.stat().st_size for x in d.glob("*.safetensors"))
                    found[d.name] = {"name": d.name, "path": str(d), "type": "safetensors",
                                     "size_gb": round(total / 1e9, 2)}
    return list(found.values())


def scan_packs():
    """扫描扩展包(角色包)目录，返回 [{name, persona_len, has_bias, bias_vocab}]"""
    packs = []
    if not PACK_DIR.exists():
        return packs
    for f in sorted(PACK_DIR.glob("role_pack*.json")):
        name = "default" if f.name == "role_pack.json" else f.name[len("role_pack_"):-len(".json")]
        try:
            data = json.load(open(f, encoding="utf-8"))
            bias_f = PACK_DIR / ("p3_bias.npy" if name == "default" else f"p3_bias_{name}.npy")
            import numpy as np
            bias = np.load(bias_f) if bias_f.exists() else None
            packs.append({"name": name, "persona_len": len(data.get("persona", "")),
                          "has_bias": bias is not None,
                          "bias_vocab": int(bias.shape[0]) if bias is not None else 0})
        except Exception as e:
            packs.append({"name": name, "error": str(e)[:80]})
    return packs


# ---------------- 引擎封装 ----------------
def build_engine(model_path, pack_name):
    """按模型+扩展包构造引擎（可复用 engine.emocompanionEngine 或独立加载）"""
    from engine import emocompanionEngine
    eng = emocompanionEngine.get()
    if model_path:  # 定向模型：重建
        import engine as engine_mod
        engine_mod.GGUF = str(model_path)
        # 简单方式：单例已载入则复用；不同模型需重启进程
        print(f"[cli] 定向模型: {Path(model_path).name}（当前进程引擎为 {eng.model_name}）")
        if Path(model_path).name != eng.model_name:
            print("[cli] 提示: 不同模型需新进程。请用: python cli.py serve -m <model>")
    pack = None
    if pack_name:
        import numpy as np
        pj = PACK_DIR / (f"role_pack_{pack_name}.json" if pack_name != "default" else "role_pack.json")
        if pj.exists():
            data = json.load(open(pj, encoding="utf-8"))
            bf = PACK_DIR / (f"p3_bias_{pack_name}.npy" if pack_name != "default" else "p3_bias.npy")
            bias = np.load(bf) if bf.exists() else eng.p3_bias
            pack = {"persona": data.get("persona", eng.persona), "p3_bias": bias,
                    "deai": data.get("deai", eng.deai)}
        else:
            print(f"[cli] 扩展包不存在: {pack_name}")
    return eng, pack


# ---------------- chat ----------------
def parse_request(text, is_json=False, interactive=False):
    """解析请求格式 -> (messages, gen_kwargs)"""
    if is_json:
        try:
            req = json.loads(text)
            msgs = req.get("messages")
            if not msgs:  # messages 缺失或为空列表时回退到顶层 content，避免丢内容
                msgs = [{"role": "user", "content": req.get("content", "")}]
            kw = {k: v for k, v in req.items() if k not in ("messages", "content")}
            return msgs, kw
        except json.JSONDecodeError as e:
            print(f"[cli] JSON 解析失败: {e}")
            return [{"role": "user", "content": text}], {}
    return [{"role": "user", "content": text}], {}


def cmd_chat(args):
    eng, pack = build_engine(args.model, args.pack)
    persona = pack["persona"] if pack else None

    if args.interactive:
        print("[cli] 交互模式（输入 exit 退出，/reset 清空历史）")
        history = []
        while True:
            try:
                q = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[cli] 再见~")
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break
            if q == "/reset":
                history = []
                print("[cli] 历史已清空")
                continue
            history.append({"role": "user", "content": q})
            r = eng.chat(history, max_new=args.max_new, persona=persona)
            reply = r["reply"]
            print(f"EmoCompanion> {reply}\n")
            history.append({"role": "assistant", "content": reply})
            # 简单上下文裁剪
            if len(history) > 10:
                history = history[-8:]
        return

    if not args.text:
        print("[cli] 请输入内容。示例: python cli.py chat \"你好呀\"")
        return
    msgs, kw = parse_request(args.text, is_json=args.json)
    kw.setdefault("max_new", args.max_new)
    kw.setdefault("temperature", args.temperature)
    r = eng.chat(msgs, persona=persona, **kw)
    print(f"EmoCompanion> {r['reply']}")
    if args.verbose:
        print(f"      ({r['latency_s']}s, {r['tok_s']} tok/s)")


def cmd_serve(args):
    # 定向模型启动后端服务（独立进程内按模型重建）
    eng, pack = build_engine(args.model, args.pack)
    if pack and args.pack:
        from server import _roles
        _roles[args.pack] = pack
        _roles["default"] = pack
    import uvicorn
    from server import app
    print(f"[cli] 启动服务 http://{args.host}:{args.port} (model={eng.model_name}, pack={args.pack or 'default'})")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_list(args):
    print("\n===== 已扫描模型 =====")
    for m in scan_models():
        print(f"  [{m['type']:>11}] {m['name']:<30} {m['size_gb']:>6} GB  {m['path']}")
    print("\n===== 已扫描扩展包(角色包) =====")
    for p in scan_packs():
        print(f"  [{p['name']:<10}] persona={p.get('persona_len',0)}字 bias_vocab={p.get('bias_vocab',0)}"
              + (f"  error={p['error']}" if 'error' in p else ""))


def cmd_scan(args):
    cmd_list(args)


def main():
    p = argparse.ArgumentParser(description="EmoCompanion CLI 交互工具")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("scan", help="扫描模型与扩展包")
    sub.add_parser("list", help="列出模型与扩展包")

    c = sub.add_parser("chat", help="对话")
    c.add_argument("text", nargs="*", help="请求内容（空格分隔或引号包住）")
    c.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    c.add_argument("-j", "--json", action="store_true", help="JSON 请求格式")
    c.add_argument("-m", "--model", help="定向模型文件名")
    c.add_argument("-p", "--pack", help="定向扩展包名")
    c.add_argument("--max-new", type=int, default=128)
    c.add_argument("--temperature", type=float, default=0.9)
    c.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("serve", help="启动后端服务")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("-m", "--model", help="定向模型文件名")
    s.add_argument("-p", "--pack", help="定向扩展包名")

    args = p.parse_args()
    setup_env()

    if args.cmd == "chat":
        args.text = " ".join(args.text) if args.text else None
        cmd_chat(args)
    elif args.cmd == "serve":
        cmd_serve(args)
    else:
        cmd_list(args)


if __name__ == "__main__":
    main()
