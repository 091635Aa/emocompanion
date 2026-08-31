# -*- coding: utf-8 -*-
"""EmoCompanion情感引擎 —— FastAPI 后端服务

接口:
  POST /chat            聊天（支持多层算法开关、角色热切换）
  GET  /roles           查看可用角色
  POST /roles/{name}    切换角色（角色包热加载）
  GET  /health          健康检查（模型状态 / 显存 / 内存 / 速度统计）
  GET  /stats           引擎统计（调用数 / token / 平均速度）

启动:
  python server.py [--port 8000]
"""
import argparse
import json
import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import emocompanionEngine, render_chat_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
ROLE_DIR = os.path.join(os.path.dirname(HERE), "data", "role_pack")
STATIC_DIR = os.path.join(HERE, "static")

app = FastAPI(title="EmoCompanion情感引擎", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

eng = emocompanionEngine.get()

# ---------------- 角色热切换 ----------------
_roles = {}


def _load_role_file(name: str) -> dict:
    """加载角色包: {name: {"persona": str, "p3_bias": ndarray|None, "deai": dict}}"""
    pack_path = os.path.join(ROLE_DIR, f"role_pack_{name}.json") if os.path.exists(
        os.path.join(ROLE_DIR, f"role_pack_{name}.json")) else os.path.join(ROLE_DIR, "role_pack.json")
    pack = json.load(open(pack_path, encoding="utf-8"))
    bias_path = os.path.join(ROLE_DIR, f"p3_bias_{name}.npy") if os.path.exists(
        os.path.join(ROLE_DIR, f"p3_bias_{name}.npy")) else os.path.join(ROLE_DIR, "p3_bias.npy")
    import numpy as np
    bias = np.load(bias_path).astype(np.float32) if os.path.exists(bias_path) else None
    return {"name": name, "persona": pack.get("persona", eng.persona),
            "p3_bias": bias, "deai": pack.get("deai", eng.deai)}


def init_roles():
    global _roles
    _roles = {"default": {"name": "default", "persona": eng.persona,
                          "p3_bias": eng.p3_bias, "deai": eng.deai}}
    # 额外角色包（若有）自动发现: role_pack_*.json
    for f in sorted(os.listdir(ROLE_DIR)):
        if f.startswith("role_pack_") and f.endswith(".json"):
            name = f[len("role_pack_"):-len(".json")]
            try:
                _roles[name] = _load_role_file(name)
            except Exception as e:
                print(f"[server] 角色 {name} 加载失败: {e}")
    print(f"[server] 可用角色: {list(_roles.keys())}")


init_roles()
_active = "default"

# 情感外挂路由最近一次请求状态（供 /debug 观测）
last_emo = {"emotion": None, "scale_emo": 1.0}


# ---------------- 请求模型 ----------------
class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new: int = 128
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    use_layers: bool = True
    role: str = "default"
    seed: Optional[int] = None
    emotion: Optional[str] = None
    scale_emo: float = 1.0


class RoleSwitch(BaseModel):
    name: str


# ---------------- 接口 ----------------
@app.post("/chat")
def chat(req: ChatRequest):
    global _active, last_emo
    role = req.role or _active
    if role not in _roles:
        raise HTTPException(404, f"未知角色: {role}（可用: {list(_roles.keys())}）")
    r = _roles[role]
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    last_emo["emotion"] = req.emotion
    last_emo["scale_emo"] = req.scale_emo
    try:
        out = eng.chat(
            msgs, max_new=req.max_new, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k, use_layers=req.use_layers,
            persona=r["persona"], seed=req.seed,
            emotion=req.emotion, scale_emo=req.scale_emo,
            p3_bias=r["p3_bias"], deai=r["deai"],
        )
    except Exception as e:
        raise HTTPException(500, f"生成失败: {e}")
    return {"role": role, **out}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE 流式聊天：逐 token 输出，末尾发 done 事件（含完整清洗后文本）。"""
    global _active, last_emo
    role = req.role or _active
    if role not in _roles:
        raise HTTPException(404, f"未知角色: {role}（可用: {list(_roles.keys())}）")
    r = _roles[role]
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    last_emo["emotion"] = req.emotion
    last_emo["scale_emo"] = req.scale_emo

    def gen():
        import json
        buf = []
        try:
            for delta in eng.chat_stream(
                    msgs, max_new=req.max_new, temperature=req.temperature,
                    top_p=req.top_p, top_k=req.top_k, use_layers=req.use_layers,
                    persona=r["persona"], seed=req.seed,
                    emotion=req.emotion, scale_emo=req.scale_emo,
                    p3_bias=r["p3_bias"], deai=r["deai"]):
                buf.append(delta)
                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            full = eng._strip_thinking("".join(buf))
            yield f"data: {json.dumps({'done': True, 'reply': full, 'topic_id': eng.current_topic}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)[:300]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/roles")
def roles():
    return {"active": _active, "roles": list(_roles.keys())}


@app.post("/roles/{name}")
def switch_role(name: str):
    global _active
    if name not in _roles:
        raise HTTPException(404, f"未知角色: {name}")
    _active = name
    return {"active": _active}


@app.get("/health")
def health():
    try:
        import subprocess
        g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        # 兼容多卡：每行一个 GPU，聚合求和（单卡则为单行）
        rows = [ln.strip() for ln in g.stdout.strip().splitlines() if ln.strip()]
        used = sum(int(ln.split(",")[0]) for ln in rows)
        total = sum(int(ln.split(",")[1]) for ln in rows)
        vram = {"used_mb": used, "total_mb": total}
    except Exception:
        vram = None
    import psutil
    mem = psutil.Process().memory_info()
    st = eng.stats
    return {
        "status": "ok",
        "model": eng.model_name,
        "vram_gb": vram,
        "process_ram_mb": round(mem.rss / 1024 / 1024, 1),
        "stats": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()},
        "avg_tok_s": round(st["tokens"] / st["seconds"], 2) if st["seconds"] > 0 else 0,
        "active_role": _active,
    }


@app.get("/stats")
def stats():
    st = eng.stats
    return {
        "calls": st["calls"], "tokens": st["tokens"], "seconds": round(st["seconds"], 2),
        "avg_tok_s": round(st["tokens"] / st["seconds"], 2) if st["seconds"] > 0 else 0,
    }


# ---------------- 前端页面 ----------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/debug")
def debug_info():
    """调试模式详情：相关标签(角色/锚点/去AI腔表) + 生成速度 + 显卡资源"""
    pack = eng.pack
    meta = pack.get("meta", {})
    deai = eng.deai or {}
    gpu = None
    try:
        import subprocess
        raw = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if raw:
            name, used, total, util = map(str.strip, raw.split(","))
            gpu = {"name": name, "used_mb": int(used), "total_mb": int(total),
                   "util_gpu": util}
    except Exception:
        gpu = None
    import psutil
    mem = psutil.Process().memory_info().rss / 1024 / 1024
    st = eng.stats
    return {
        "meta": meta,
        "persona": eng.persona,
        "deai_summary": {
            "pos_tok": len(deai.get("pos_tok", {})),
            "hol_tok": len(deai.get("hol_tok", {})),
            "pos_phr": len(deai.get("pos_phr", [])),
            "hol_phr": len(deai.get("hol_phr", [])),
            "ooc_phr": len(deai.get("ooc_phr", [])),
        },
        "emo": {
            "active_emotion": last_emo["emotion"],
            "scale_emo": last_emo["scale_emo"],
            "emo_table_loaded": eng.emo_vectors is not None,
            "emotions": eng.emo_names or [],
        },
        "gpu": gpu,
        "model": eng.model_name,
        "process_ram_mb": round(mem, 1),
        "stats": {
            "calls": st["calls"],
            "tokens": st["tokens"],
            "seconds": round(st["seconds"], 3),
            "avg_tok_s": round(st["tokens"] / st["seconds"], 2) if st["seconds"] > 0 else 0,
        },
    }


def main():
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()
    print(f"[server] EmoCompanion情感引擎启动: http://{a.host}:{a.port}  (角色: {list(_roles.keys())})")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
