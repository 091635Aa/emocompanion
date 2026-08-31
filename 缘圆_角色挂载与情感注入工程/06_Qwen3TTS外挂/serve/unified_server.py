# -*- coding: utf-8 -*-
"""缘圆 前端后端一体化 统一服务

路径:
  - 原生 TTS:     GET  /api/tts/models        列 adapter/情感/采样率
                  GET  /api/tts/health         TTS 组件就绪状态（不强制加载模型）
                  POST /api/tts/synthesize     合成 → audio/wav
  - 文本引擎代理:  POST /api/text/chat         转发到 04 文本引擎（--text-base 可配）
  - 前端:         GET  /                       serve/web/index.html

启动:
  python unified_server.py [--host 127.0.0.1] [--port 8070]
                           [--text-base http://127.0.0.1:8000] [--skip-tts]
"""
import argparse
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tts_engine import get_engine, list_adapter_names, TTSUnavailable
from emo_detect import detect_emotion, _extract_text, EMOTIONS
from tts_gguf import GGUFTTS

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")

app = FastAPI(title="缘圆 前端后端一体化", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 运行参数（由 main() 注入）
_TEXT_BASE = "http://127.0.0.1:8000"
_SKIP_TTS = False
_GGUF = None


def _get_gguf():
    """GGUF 快速后端（懒加载）。"""
    global _GGUF
    if _GGUF is None:
        _GGUF = GGUFTTS.get()
    return _GGUF


# ---------------- 请求模型 ----------------
class ChatMsg(BaseModel):
    role: str = "user"
    content: str


class TTSReq(BaseModel):
    text: str
    emotion: str = "平静"
    backend: str = "gguf"   # gguf(默认, llama-tts + 训练音色LoRA) | tf(历史transformers,不建议)
    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    rate: float | None = None


class DetectReq(BaseModel):
    text: str
    user_msg: str = ""        # 用户语境，用于优先按用户真实情绪判定


class PipelineReq(BaseModel):
    messages: list[ChatMsg]
    max_new: int = 128
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    role: str = "default"
    seed: int | None = None
    auto_emotion: bool = True
    emotion: str = "平静"     # auto_emotion=False 时的显式情感
    backend: str = "gguf"     # gguf(默认, 训练音色LoRA) | tf(历史transformers)
    tts_seed: int | None = None
    tts_temperature: float | None = None
    tts_rate: float | None = None


class ChatReq(BaseModel):
    messages: list[ChatMsg]
    max_new: int = 128
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    role: str = "default"
    seed: int | None = None


# ---------------- TTS ----------------
@app.get("/api/tts/models")
def tts_models():
    if _SKIP_TTS:
        return {"backend": "gguf", "available": False, "reason": "disabled"}
    return _get_gguf().availability()


@app.get("/api/tts/health")
def tts_health():
    if _SKIP_TTS:
        return {"tts": "disabled", "loaded": False, "skip": True}
    gguf = _get_gguf().availability()
    # 主链已收敛为 gguf(single path)；tf 仅作兼容枚举，不再作为主链路
    tf_missing = True
    tf_reason = "已弃用 (transformers 主链移除)"
    return {"tts": "ready" if gguf.get("available") else "missing",
            "loaded": gguf.get("available"), "backend": "gguf",
            "reason": gguf.get("reason"), "gguf": gguf,
            "tf": {"available": False, "reason": tf_reason}}


@app.post("/api/tts/synthesize")
def tts_synthesize(req: TTSReq):
    if _SKIP_TTS:
        raise HTTPException(503, "TTS 已禁用(--skip-tts)")
    try:
        gg = _get_gguf()
        wav, sr, meta = gg.synthesize(req.text, req.emotion,
                                      temperature=req.temperature,
                                      top_k=req.top_k, rate=req.rate,
                                      seed=req.seed)
        import numpy as np
        arr = np.asarray(wav, dtype="float32")
        # 直接用标准 WAV 容器写入（24kHz mono），无需额外依赖
        buf = io.BytesIO()
        _write_wav(buf, arr, sr)
        return Response(content=buf.getvalue(), media_type="audio/wav",
                        headers={"X-TTS-Meta": json.dumps(meta, ensure_ascii=True)})
    except (TTSUnavailable, RuntimeError) as ex:
        raise HTTPException(503, str(ex))


# ---------------- 情感自动识别 ----------------
@app.post("/api/tts/detect")
def tts_detect(req: DetectReq):
    """对话→TTS 前的情感自动识别。文本引擎在线走 LLM(高精度)，离线走词典。
    可传 user_msg 提供用户语境，优先按用户真实情绪判定。"""
    res = detect_emotion(req.text or "", text_chat_fn=_call_text_engine,
                         user_msg=req.user_msg or "")
    return {"label": res.label, "confidence": res.confidence,
            "source": res.source, "detail": res.detail, "raw": res.raw[:200] if res.raw else ""}


# ---------------- 流水线：对话 → 自动情感 → TTS ----------------
@app.post("/api/pipeline/talk")
def pipeline_talk(req: PipelineReq):
    """对话完成即交互：文本回复 + 自动情感 + 合成音频。
    返回 audio/wav，附 X-Text(回复文本) 与 X-Pipe(情感/耗时/精度)。"""
    if _SKIP_TTS:
        raise HTTPException(503, "TTS 已禁用(--skip-tts)")
    # 1) 文本回复
    chat_resp = _call_text_engine({
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "max_new": req.max_new, "temperature": req.temperature,
        "top_p": req.top_p, "top_k": req.top_k, "role": req.role, "seed": req.seed,
    })
    reply = (_extract_text(chat_resp) or "").strip()
    if not reply:
        raise HTTPException(422, "文本引擎返回空回复")
    # 2) 情感：优先按用户真实情绪判定(结合用户最后一条消息语境)，再兼顾回复语气
    user_msg = ""
    for m in reversed(req.messages):
        if m.role in ("user", "human"):
            user_msg = (m.content or "").strip()
            break
    if req.auto_emotion:
        r = detect_emotion(reply, text_chat_fn=_call_text_engine, user_msg=user_msg)
        emotion = r.label
        emo = {"label": r.label, "confidence": r.confidence, "source": r.source,
               "user_msg": user_msg}
    else:
        emotion = req.emotion
        emo = {"label": emotion, "confidence": 1.0, "source": "manual"}
    # 3) 合成（主链 gguf；tf 仅历史兼容）
    try:
        if req.backend == "tf":
            wav, sr, tts_meta = get_engine().synthesize(reply, emotion)
        else:
            wav, sr, tts_meta = _get_gguf().synthesize(
                reply, emotion, temperature=req.tts_temperature,
                rate=req.tts_rate, seed=req.tts_seed)
        import numpy as np
        buf = io.BytesIO()
        _write_wav(buf, np.asarray(wav, dtype="float32"), sr)
        pipe_meta = {"reply": reply, "emotion": emotion, "confidence": emo["confidence"],
                     "source": emo["source"], "backend": req.backend, "tts": tts_meta}
        return Response(content=buf.getvalue(), media_type="audio/wav",
                        headers={"X-Text": json.dumps(reply, ensure_ascii=True),
                                 "X-Pipe": json.dumps(pipe_meta, ensure_ascii=True)})
    except Exception as ex:
        # 音频不可用时仍返回文本+识别结果便于前端展示
        raise HTTPException(503, f"{ex}\n[识别到情感] {emotion}")


# ---------------- 文本引擎代理 ----------------
def _call_text_engine(payload: dict) -> dict:
    """向 04 文本引擎 /chat 发请求（供代理与情感识别复用）。"""
    try:
        data = json.dumps(payload).encode("utf-8")
        r = urllib.request.Request(_TEXT_BASE.rstrip("/") + "/chat", data=data,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code or 502, f"文本引擎返回错误: {e.read().decode('utf-8', 'ignore')}")
    except Exception as e:
        raise HTTPException(502, f"无法连接文本引擎 {_TEXT_BASE}: {e}")


@app.post("/api/text/chat")
def text_chat(req: ChatReq):
    return _call_text_engine({
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "max_new": req.max_new, "temperature": req.temperature,
        "top_p": req.top_p, "top_k": req.top_k, "role": req.role,
        "seed": req.seed,
    })


# ---------------- 前端静态托管 ----------------
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


# ---------------- 极小 WAV 写入器（避免引入 soundfile） ----------------
def _write_wav(buf, arr, sr):
    import wave
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16 16bit
        w.setframerate(sr)
        import numpy as np
        pcm = (np.clip(np.asarray(arr).astype("float32"), -1.0, 1.0) * 32767).astype("int16")
        w.writeframes(pcm.tobytes())


def main():
    global _TEXT_BASE, _SKIP_TTS
    import uvicorn
    ap = argparse.ArgumentParser(description="缘圆前端后端一体化服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--text-base", default="http://127.0.0.1:8000")
    ap.add_argument("--skip-tts", action="store_true", help="禁用 TTS（仅文本/前端）")
    a = ap.parse_args()
    _TEXT_BASE = a.text_base
    _SKIP_TTS = a.skip_tts
    print(f"[unified] 前端后端一体化: http://{a.host}:{a.port}  文本引擎={_TEXT_BASE}  TTS={'跳过' if _SKIP_TTS else '启用'}")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()