# -*- coding: utf-8 -*-
"""EmoCompanion TTS Studio —— Qwen-Audio TTS 合成工作台（后端）

启动：  python app.py                    # 默认 127.0.0.1:8000
查询模型：python app.py --list-models    # 命令行查询支持的模型
查询音色：python app.py --list-voices    # 命令行查询已注册的复刻音色（需 API Key）
"""
import argparse
import sys
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import config
import deepseek
import tts

config.load_env_file()  # 读取 .env 中的 API Key

BASE_DIR = config.res_dir()      # 只读资源（static 等）
DATA_DIR = config.app_dir()      # 可写数据（audio_cache 等）
AUDIO_DIR = DATA_DIR / "audio_cache"
AUDIO_DIR.mkdir(exist_ok=True)

MEDIA_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/pcm"}

app = FastAPI(title="EmoCompanion模块", version="1.0.0")


# ---------------- 请求体 ----------------
class RenameModelReq(BaseModel):
    model_id: str
    alias: str = ""


class RenameVoiceReq(BaseModel):
    voice_id: str
    alias: str = ""


class StyleReq(BaseModel):
    name: str
    instruction: str = ""


class SynthReq(BaseModel):
    model: str = "qwen-audio-3.0-tts-plus"
    voice: str = ""
    text: str
    instruction: str = ""
    format: str = "wav"
    sample_rate: int = 48000


class AiAnnotateReq(BaseModel):
    text: str
    hint: str = ""


class AiOptInstReq(BaseModel):
    instruction: str


class PronunciationReq(BaseModel):
    word: str
    ph: str = ""


# ---------------- 工具函数 ----------------
def _models_with_alias():
    out = []
    for m in tts.SUPPORTED_MODELS:
        item = dict(m)
        item["alias"] = config.model_alias(m["id"])
        item["name"] = item["alias"] if item["alias"] != m["id"] else m["id"]
        out.append(item)
    return out


def _voice_label(v, aliases):
    alias = aliases.get(v["id"])
    return alias or v["name"] or v["id"]


# ---------------- 页面与静态 ----------------
@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/audio/{name}")
def audio_file(name: str):
    # 仅允许本服务缓存目录内的文件
    safe = Path(name).name
    path = AUDIO_DIR / safe
    if not path.exists():
        raise HTTPException(404, "音频不存在或已过期")
    ext = safe.rsplit(".", 1)[-1].lower()
    return FileResponse(path, media_type=MEDIA_TYPES.get(ext, "application/octet-stream"))


# ---------------- API：模型 ----------------
@app.get("/api/models")
def list_models():
    """查询支持的模型（含重命名后的别名）。"""
    return {"models": _models_with_alias(), "has_env_key": bool(tts.env_api_key())}


@app.post("/api/models/rename")
def rename_model(req: RenameModelReq):
    if req.model_id not in tts.MODEL_IDS:
        raise HTTPException(400, f"未知模型: {req.model_id}")
    config.rename_model(req.model_id, req.alias)
    return {"ok": True, "models": _models_with_alias()}


# ---------------- API：音色 ----------------
@app.get("/api/voices")
def list_voices(model: str = "qwen-audio-3.0-tts-plus"):
    """查询音色：系统音色 + 本地复刻 + DashScope 在线查询。"""
    aliases = config.load()["voice_aliases"]
    voices = []
    for v in tts.SYSTEM_VOICES:
        item = dict(v)
        item["kind"] = "system"
        item["name"] = _voice_label(v, aliases)
        voices.append(item)
    for v in tts.local_voices():
        item = dict(v)
        item["name"] = _voice_label(v, aliases)
        voices.append(item)
    try:
        live = tts.query_api_voices("")
    except Exception:
        live = []
    for v in live:
        item = dict(v)
        item["name"] = _voice_label(v, aliases)
        if not any(x["id"] == item["id"] for x in voices):
            voices.append(item)
    return {"voices": voices, "default_voice": config.last_used().get("voice") or ""}


@app.post("/api/voices/rename")
def rename_voice(req: RenameVoiceReq):
    config.rename_voice(req.voice_id, req.alias)
    return {"ok": True}


# ---------------- API：AI（DeepSeek） ----------------
@app.post("/api/ai/annotate")
def ai_annotate(req: AiAnnotateReq):
    """AI 自动识别语气，在原文中插入 [] 标签。"""
    if not (req.text or "").strip():
        raise HTTPException(400, "请先输入要标注的文本")
    try:
        return deepseek.annotate("", req.text, req.hint)
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.post("/api/ai/optimize_instruction")
def ai_optimize_instruction(req: AiOptInstReq):
    """AI 优化全局风格指令。"""
    if not (req.instruction or "").strip():
        raise HTTPException(400, "请先生成或填写指令")
    try:
        return deepseek.optimize_instruction("", req.instruction)
    except RuntimeError as e:
        raise HTTPException(502, str(e))


# ---------------- API：余额 ----------------
@app.get("/api/balance/deepseek")
def balance_deepseek():
    """查询 DeepSeek 余额（官方接口）。"""
    try:
        return {"ok": True, **deepseek.query_balance("")}
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.get("/api/balance/aliyun")
def balance_aliyun():
    """阿里云余额：官方未开放按 API Key 查询，这里做 Key 有效性与服务可用性探测。"""
    try:
        ok, msg = tts.test_api_key("")
        return {"ok": True, "valid": ok, "message": msg}
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.post("/api/balance/aliyun/probe")
def balance_aliyun_probe():
    """试合成 1 个字探测阿里云余额/配额（会消耗极小额度）。"""
    try:
        return {"ok": True, **tts.aliyun_probe("")}
    except RuntimeError as e:
        raise HTTPException(502, str(e))


# ---------------- API：风格预设 ----------------
@app.get("/api/styles")
def get_styles():
    return {"presets": config.style_presets()}


@app.post("/api/styles")
def add_style(req: StyleReq):
    try:
        p = config.add_style_preset(req.name, req.instruction)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "preset": p, "presets": config.style_presets()}


@app.delete("/api/styles")
def delete_style(name: str):
    config.delete_style_preset(name)
    return {"ok": True, "presets": config.style_presets()}


# ---------------- API：发音纠正（多音字） ----------------
@app.get("/api/pronunciations")
def get_pronunciations():
    return {"pronunciations": config.pronunciations()}


@app.post("/api/pronunciations")
def add_pronunciation(req: PronunciationReq):
    try:
        p, is_new = config.upsert_pronunciation(req.word, req.ph)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "created": is_new, "pronunciations": config.pronunciations()}


@app.delete("/api/pronunciations")
def delete_pronunciation(word: str):
    config.delete_pronunciation(word)
    return {"ok": True, "pronunciations": config.pronunciations()}


# ---------------- API：合成 ----------------
@app.post("/api/synthesize")
def synthesize(req: SynthReq):
    if req.model not in tts.MODEL_IDS:
        raise HTTPException(400, f"未知模型: {req.model}")
    fmt = req.format if req.format in ("wav", "mp3", "pcm") else "wav"
    warning = ""
    bad = tts.unsupported_tags(req.model, req.text)
    if bad:
        warning = (f"当前模型 {req.model} 不支持标签，以下标签将按原文读出："
                   + "、".join(f"[{b}]" for b in bad))
    try:
        data, request_id = tts.synthesize(
            api_key="", model=req.model, voice=req.voice,
            text=req.text, instruction=req.instruction,
            fmt=fmt, sample_rate=req.sample_rate,
            pronunciations=config.pronunciations())
    except RuntimeError as e:
        m = str(e)
        if "411" in m or "Engine error" in m:
            m += ("（提示：如果你用的是 sk-ws- 开头的「业务空间」专用密钥，它无法通过本接口合成语音；"
                  "请到百炼控制台右上角 API-KEY 创建普通密钥 sk- 开头，替换 .env 后重启）")
        raise HTTPException(502, m)

    name = f"{uuid.uuid4().hex[:12]}_{int(__import__('time').time())}.{fmt}"
    (AUDIO_DIR / name).write_bytes(data)
    config.set_last_used(model=req.model, voice=req.voice, format=fmt,
                         sample_rate=req.sample_rate)
    return {
        "ok": True,
        "audio_url": f"/audio/{name}",
        "request_id": request_id,
        "format": fmt,
        "size": len(data),
        "warning": warning,
        "model": req.model,
    }


# ---------------- 启动引导数据 ----------------
@app.get("/api/bootstrap")
def bootstrap():
    return {
        "models": _models_with_alias(),
        "tags": {"control": tts.CONTROL_TAGS, "rich": tts.RICH_TAGS},
        "voice_dimensions": tts.VOICE_DIMENSIONS,
        "scene_presets": tts.SCENE_PRESETS,
        "style_presets": config.style_presets(),
        "last_used": config.last_used(),
        "pronunciations": config.pronunciations(),
        "has_dashscope_key": bool(tts.env_api_key()),
        "has_deepseek_key": bool(deepseek.env_key()),
        "voice_id_files": [str(f) for f in tts.voice_id_files()],
    }


# ---------------- CLI ----------------
def cli_list_models():
    print("=" * 78)
    print(f"{'模型ID':<28}{'别名':<20}{'标签':<6}{'指令':<6}说明")
    print("-" * 78)
    for m in _models_with_alias():
        print(f"{m['id']:<28}{(m['alias'] if m['alias'] != m['id'] else '-'):<20}"
              f"{'支持' if m['tags'] else '—':<6}{'支持' if m['instruction'] else '—':<6}{m['note']}")
    print("=" * 78)


def cli_list_voices():
    voices = []
    for v in tts.SYSTEM_VOICES:
        voices.append(("system", v["id"], v["name"]))
    for v in tts.local_voices():
        voices.append(("local", v["id"], v["name"]))
    for v in tts.query_api_voices(""):
        if not any(x[1] == v["id"] for x in voices):
            voices.append(("dashscope", v["id"], v["name"]))
    print("=" * 78)
    print(f"{'来源':<12}{'音色ID':<40}名称")
    print("-" * 78)
    for kind, vid, name in voices:
        print(f"{kind:<12}{vid:<40}{name}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="EmoCompanion TTS Studio")
    parser.add_argument("--list-models", action="store_true", help="查询支持的模型")
    parser.add_argument("--list-voices", action="store_true", help="查询音色（需 API Key）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args()

    if args.list_models:
        cli_list_models()
        return
    if args.list_voices:
        cli_list_voices()
        return

    import uvicorn
    print(f"EmoCompanion TTS Studio 已启动： http://{args.host}:{args.port}")
    print(f"查询支持的模型： python {Path(__file__).name} --list-models")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
