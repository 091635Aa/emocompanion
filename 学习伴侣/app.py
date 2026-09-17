"""
学习伴侣 - 后端入口
FastAPI + WebSocket 全局推送：每 0.1s 刷新模拟引擎并广播状态快照到所有连接（学生端+管理端）。
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from engine.simulator import LearningCompanionSimulator
from engine.agents import AgentSupervisor

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

sim = LearningCompanionSimulator(baseline_temp=10.0)
clients: set[WebSocket] = set()
supervisor = AgentSupervisor(sim, broadcast=None)  # broadcast 在 lifespan 后注入


@asynccontextmanager
async def lifespan(app: FastAPI):
    supervisor.broadcast = broadcast
    supervisor.start()           # 并行启动 5 个领域智能体
    asyncio.create_task(simulation_loop())
    yield


app = FastAPI(title="学习伴侣 · 智能伙伴", lifespan=lifespan)


# ---------- 页面路由 ----------
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
async def admin():
    return FileResponse(STATIC_DIR / "admin.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- 操作处理 ----------
def handle_command(cmd):
    """统一处理来自任意端（学生端/管理端/语音）的控制指令。"""
    kind = cmd.get("type")
    if kind == "toggle_mode":
        sim.toggle_mode()
    elif kind == "set_mode":
        sim.set_mode(cmd.get("mode", "study"))
    elif kind == "set_noise_threshold":
        sim.set_noise_threshold(cmd.get("value", 60))
    elif kind == "set_baseline_temp":
        sim.set_baseline_temp(cmd.get("value", 10))
    elif kind == "agent_control":
        # 控制端派发：强制并行智能体立即重算并推送
        supervisor.refresh_all()


# ---------- WebSocket 全局推送 ----------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        # 连接即推送一次当前状态（含并行智能体快照）
        await ws.send_text(json.dumps({
            "type": "state",
            "data": sim.snapshot(),
            "agents": supervisor.current_snapshot(),
        }, ensure_ascii=False))
        # 处理来自客户端的控制指令
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
                handle_command(cmd)
            except Exception:
                continue
    except Exception:
        pass
    finally:
        clients.discard(ws)


async def broadcast(msg: dict):
    data = json.dumps(msg, ensure_ascii=False)
    dead = []
    for ws in list(clients):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def simulation_loop():
    """每 0.1s 推进一次引擎并全局推送状态快照。"""
    while True:
        sim.tick()
        await broadcast({
            "type": "state",
            "data": sim.snapshot(),
        })
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")