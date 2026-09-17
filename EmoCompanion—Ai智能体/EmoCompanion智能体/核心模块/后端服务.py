# -*- coding: utf-8 -*-
"""EmoCompanion智能体 V2 —— 后端服务（FastAPI）。

路由路径为中文；请求/响应 JSON 字段保持与 V1 一致的英文契约
（见各路由注释），配置文件（配置文件.json）内部键为中文，
本模块负责两者之间的映射。
"""
import json
import re
import time
import uuid
from pathlib import Path
from typing import List

import fastapi.routing as 路由模块
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.convertors import CONVERTOR_TYPES, PathConvertor

from 环境配置 import 数据目录, 项目根目录, 资源目录
from . import 配置持久化, 音色管理, 智能助手, 语音合成, 实时通话, 多模态识别

前端目录 = 资源目录() / "前端页面"
音频目录 = 数据目录() / "数据缓存" / "音频缓存"
音频目录.mkdir(parents=True, exist_ok=True)

媒体类型表 = {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/pcm",
            "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "mp4": "video/mp4", "webm": "video/webm",
            "mov": "video/quicktime"}

# 静态资源媒体类型：显式指定，避免 Windows / PyInstaller 打包环境下
# Python mimetypes 把 .js 等推断为 text/plain —— AudioWorklet.addModule
# 强制要求 JavaScript MIME，否则报 AbortError: Unable to load a worklet's module。
静态媒体类型表 = {
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".ico": "image/x-icon", ".webp": "image/webp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".mp4": "video/mp4", ".webm": "video/webm",
    ".woff": "font/woff", ".woff2": "font/woff2",
}

# ---------- 中文路径参数支持（绕过 Starlette 仅 ASCII 参数名的限制） ----------
# Starlette 1.3.1 的 PARAM_REGEX 只匹配 [a-zA-Z_][a-zA-Z0-9_]* 的参数名，
# 中文参数名（如 {文件路径:路径}）会被当作路径字面量，导致路由永远无法命中。
# 解决方案：注册中文转换器别名「路径」，并把 fastapi.routing 的 compile_path
# 替换为支持中文参数名/转换器的等价实现（Python re 支持 Unicode 分组名）。
# 该补丁必须在注册任何路由之前生效，属全中文命名的必要技术适配。
CONVERTOR_TYPES["路径"] = PathConvertor()
中文参数正则 = re.compile(r"{([^/{}]+?)(?::([^/{}]+?))?}")


def _编译中文路径(路径):
    """等价于 starlette.routing.compile_path，但支持中文参数名与转换器名。"""
    路径正则 = "^"
    路径格式 = ""
    参数转换器 = {}
    索引 = 0
    for 匹配 in 中文参数正则.finditer(路径):
        参数名, 转换器类型 = 匹配.groups()
        转换器类型 = (转换器类型 or "str").lstrip(":")
        assert 转换器类型 in CONVERTOR_TYPES, f"未知路径转换器 '{转换器类型}'"
        转换器 = CONVERTOR_TYPES[转换器类型]
        路径正则 += re.escape(路径[索引:匹配.start()])
        路径正则 += f"(?P<{参数名}>{转换器.regex})"
        路径格式 += 路径[索引:匹配.start()] + "{%s}" % 参数名
        参数转换器[参数名] = 转换器
        索引 = 匹配.end()
    路径正则 += re.escape(路径[索引:]) + "$"
    路径格式 += 路径[索引:]
    return re.compile(路径正则), 路径格式, 参数转换器


路由模块.compile_path = _编译中文路径  # 必须在注册任何路由之前替换

应用 = FastAPI(title="EmoCompanion智能体", version="2.0.0")


# ---------------- 请求体（字段保持 V1 英文契约） ----------------
class 模型重命名请求(BaseModel):
    model_id: str
    alias: str = ""


class 音色重命名请求(BaseModel):
    voice_id: str
    alias: str = ""


class 风格请求(BaseModel):
    name: str
    instruction: str = ""


class 合成请求(BaseModel):
    model: str = ""   # 空则按账号类型自动选择（业务空间→qwen3-tts-flash，普通→qwen-audio-3.0-tts-plus）
    voice: str = ""
    text: str
    instruction: str = ""
    format: str = "wav"
    sample_rate: int = 48000


class AI标注请求(BaseModel):
    text: str
    hint: str = ""


class AI优化指令请求(BaseModel):
    instruction: str


class 发音纠正请求(BaseModel):
    word: str
    ph: str = ""


# ---------------- 工具函数（配置中文键 <-> 接口英文键映射） ----------------
def 带别名的模型列表():
    输出 = []
    for 模型 in 语音合成.支持的模型列表:
        条目 = dict(模型)
        条目["alias"] = 配置持久化.模型别名(模型["id"])
        条目["name"] = 条目["alias"] if 条目["alias"] != 模型["id"] else 模型["id"]
        输出.append(条目)
    return 输出


def _风格预设转英文(预设们):
    return [{"name": 预设["名称"], "instruction": 预设.get("指令", "")} for 预设 in 预设们]


def _发音纠正转英文(纠正们):
    return [{"word": 条目["词"], "ph": 条目["拼音"]} for 条目 in 纠正们]


def _最近使用转英文(最近):
    return {
        "model": 最近.get("模型", ""),
        "voice": 最近.get("音色", ""),
        "format": 最近.get("格式", ""),
        "sample_rate": 最近.get("采样率", 48000),
    }


# ---------------- 页面与静态 ----------------
@应用.get("/")
def 首页():
    return FileResponse(前端目录 / "页面入口.html")


@应用.get("/静态/{文件路径:路径}")
def 静态文件(文件路径: str):
    """服务 前端页面 目录下任意文件（含 脚本/、资源/、页面样式.css）。

    显式指定媒体类型（静态媒体类型表），保证 .js 以 JavaScript MIME 返回，
    AudioWorklet.addModule 才能加载（打包 EXE / Windows 环境易推断为 text/plain）。
    """
    前端根 = 前端目录.resolve()
    目标 = (前端目录 / 文件路径).resolve()
    if 目标 != 前端根 and 前端根 not in 目标.parents:
        raise HTTPException(404, "文件不存在")
    if not 目标.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(目标, media_type=静态媒体类型表.get(目标.suffix.lower()))


@应用.get("/音频/{文件名}")
def 音频文件(文件名: str):
    """服务 数据缓存/音频缓存 下的音频文件（校验文件名安全）。"""
    安全名 = Path(文件名).name
    目标 = 音频目录 / 安全名
    if not 目标.is_file():
        raise HTTPException(404, "音频不存在或已过期")
    扩展名 = 安全名.rsplit(".", 1)[-1].lower()
    return FileResponse(目标, media_type=媒体类型表.get(扩展名, "application/octet-stream"))


# ---------------- API：模型 ----------------
@应用.get("/api/模型")
def 模型列表():
    """查询支持的模型（含重命名后的别名）。"""
    return {"models": 带别名的模型列表(), "has_env_key": bool(语音合成.环境密钥())}


@应用.post("/api/模型/重命名")
def 模型重命名(请求: 模型重命名请求):
    if 请求.model_id not in 语音合成.模型ID列表:
        raise HTTPException(400, f"未知模型: {请求.model_id}")
    配置持久化.重命名模型(请求.model_id, 请求.alias)
    return {"ok": True, "models": 带别名的模型列表()}


# ---------------- API：音色 ----------------
@应用.get("/api/音色")
def 音色列表(model: str = "qwen-audio-3.0-tts-plus"):
    """查询音色：系统音色 + 本地复刻 + DashScope 在线查询（去重合并）。"""
    音色们 = 音色管理.全部音色(模型ID=model)
    return {"voices": 音色们, "default_voice": 配置持久化.最近使用().get("音色") or ""}


@应用.post("/api/音色/重命名")
def 音色重命名(请求: 音色重命名请求):
    配置持久化.重命名音色(请求.voice_id, 请求.alias)
    return {"ok": True}


# ---------------- API：AI（DeepSeek） ----------------
@应用.post("/api/AI/标注")
def AI标注(请求: AI标注请求):
    """AI 自动识别语气，在原文中插入 [] 标签。"""
    if not (请求.text or "").strip():
        raise HTTPException(400, "请先输入要标注的文本")
    try:
        return 智能助手.标注("", 请求.text, 请求.hint)
    except RuntimeError as 异常:
        raise HTTPException(502, str(异常))


@应用.post("/api/AI/优化指令")
def AI优化指令(请求: AI优化指令请求):
    """AI 优化全局风格指令。"""
    if not (请求.instruction or "").strip():
        raise HTTPException(400, "请先生成或填写指令")
    try:
        return 智能助手.优化指令("", 请求.instruction)
    except RuntimeError as 异常:
        raise HTTPException(502, str(异常))


# ---------------- API：余额 ----------------
@应用.get("/api/余额/深度求索")
def 余额深度求索():
    """查询 DeepSeek 余额（官方接口）。"""
    try:
        return {"ok": True, **智能助手.查询余额("")}
    except RuntimeError as 异常:
        raise HTTPException(502, str(异常))


@应用.get("/api/余额/阿里云")
def 余额阿里云():
    """阿里云余额：官方未开放按 API Key 查询，这里做 Key 有效性与服务可用性探测。"""
    try:
        ok, 消息 = 语音合成.测试密钥("")
        return {"ok": True, "valid": ok, "message": 消息}
    except RuntimeError as 异常:
        raise HTTPException(502, str(异常))


@应用.post("/api/余额/阿里云/探测")
def 余额阿里云探测():
    """试合成 1 个字探测阿里云余额/配额（会消耗极小额度）。"""
    try:
        return {"ok": True, **语音合成.探测余额("")}
    except RuntimeError as 异常:
        raise HTTPException(502, str(异常))


# ---------------- API：风格预设 ----------------
@应用.get("/api/风格")
def 获取风格():
    return {"presets": _风格预设转英文(配置持久化.风格预设())}


@应用.post("/api/风格")
def 新增风格(请求: 风格请求):
    try:
        预设 = 配置持久化.新增风格预设(请求.name, 请求.instruction)
    except ValueError as 异常:
        raise HTTPException(400, str(异常))
    return {"ok": True, "preset": _风格预设转英文([预设])[0],
            "presets": _风格预设转英文(配置持久化.风格预设())}


@应用.delete("/api/风格")
def 删除风格(name: str):
    配置持久化.删除风格预设(name)
    return {"ok": True, "presets": _风格预设转英文(配置持久化.风格预设())}


# ---------------- API：发音纠正（多音字） ----------------
@应用.get("/api/发音纠正")
def 获取发音纠正():
    return {"pronunciations": _发音纠正转英文(配置持久化.发音纠正表())}


@应用.post("/api/发音纠正")
def 新增发音纠正(请求: 发音纠正请求):
    try:
        条目, 是否新增 = 配置持久化.新增发音纠正(请求.word, 请求.ph)
    except ValueError as 异常:
        raise HTTPException(400, str(异常))
    return {"ok": True, "created": 是否新增,
            "pronunciations": _发音纠正转英文(配置持久化.发音纠正表())}


@应用.delete("/api/发音纠正")
def 删除发音纠正(word: str):
    配置持久化.删除发音纠正(word)
    return {"ok": True, "pronunciations": _发音纠正转英文(配置持久化.发音纠正表())}


# ---------------- API：合成 ----------------
@应用.post("/api/合成")
def 合成(请求: 合成请求):
    模型 = (请求.model or "").strip() or 语音合成.推荐模型()
    if 模型 not in 语音合成.模型ID列表:
        raise HTTPException(400, f"未知模型: {模型}")
    格式 = 请求.format if 请求.format in ("wav", "mp3", "pcm") else "wav"
    警告 = ""
    不支持们 = 语音合成.不支持的标签(模型, 请求.text)
    if 不支持们:
        警告 = (f"当前模型 {模型} 不支持标签，以下标签将按原文读出："
                + "、".join(f"[{标签}]" for 标签 in 不支持们))
    try:
        音频数据, 请求ID = 语音合成.合成(
            密钥="", 模型=模型, 音色=请求.voice,
            文本=请求.text, 指令=请求.instruction,
            格式=格式, 采样率=请求.sample_rate,
            发音纠正=配置持久化.发音纠正表())
    except RuntimeError as 异常:
        消息 = str(异常)
        # 业务空间账号（maas 域名）若某模型未开通，语音合成.py 已自动兜底；
        # 仍失败时给出账号相关的可操作提示
        if 语音合成.是否业务空间():
            消息 += ("（提示：当前使用业务空间域名，若提示模型未开通，可在「合成」页选择 "
                     "qwen3-tts-flash / qwen3-tts-instruct-flash 模型（音色 Cherry/Serena/Ethan））")
        raise HTTPException(502, 消息)

    文件名 = f"{uuid.uuid4().hex[:12]}_{int(time.time())}.{格式}"
    (音频目录 / 文件名).write_bytes(音频数据)
    配置持久化.记录最近使用(模型=模型, 音色=请求.voice, 格式=格式,
                             采样率=请求.sample_rate)
    return {
        "ok": True,
        "audio_url": f"/音频/{文件名}",
        "request_id": 请求ID,
        "format": 格式,
        "size": len(音频数据),
        "warning": 警告,
        "model": 请求.model,
    }


# ---------------- API：启动引导数据 ----------------
@应用.get("/api/引导数据")
def 引导数据():
    return {
        "models": 带别名的模型列表(),
        "tags": {"control": 语音合成.控制标签列表, "rich": 语音合成.富标签列表},
        "voice_dimensions": 语音合成.声音维度表,
        "scene_presets": 语音合成.场景预设表,
        "style_presets": _风格预设转英文(配置持久化.风格预设()),
        "last_used": _最近使用转英文(配置持久化.最近使用()),
        "pronunciations": _发音纠正转英文(配置持久化.发音纠正表()),
        "has_dashscope_key": bool(语音合成.环境密钥()),
        "has_deepseek_key": bool(智能助手.环境密钥()),
        "voice_id_files": [str(文件) for 文件 in 语音合成.音色ID文件列表()],
    }


# ---------------- 实时通话 与 多模态识别 路由挂载 ----------------
应用.include_router(实时通话.通话路由)
应用.include_router(多模态识别.识别路由)


# ---------------- API：上传与缓存文件 ----------------
缓存分类目录 = {"图像": "上传图像缓存", "视频": "录制视频缓存", "音频": "音频缓存"}


@应用.post("/api/上传")
async def 上传文件(文件: UploadFile = File(...), 类型: str = Form("图像")):
    """上传媒体文件到 数据缓存 对应目录（图像/视频/音频）。"""
    if 类型 not in 缓存分类目录:
        raise HTTPException(400, f"不支持的缓存类型：{类型}（支持：{'/'.join(缓存分类目录)}）")
    原始名 = (文件.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    扩展名 = 原始名.rsplit(".", 1)[-1].lower() if "." in 原始名 else "bin"
    文件名 = f"{uuid.uuid4().hex}_{int(time.time())}.{扩展名}"
    目录 = 数据目录() / "数据缓存" / 缓存分类目录[类型]
    目录.mkdir(parents=True, exist_ok=True)
    (目录 / 文件名).write_bytes(await 文件.read())
    return {"ok": True, "类型": 类型, "路径": f"/缓存/{类型}/{文件名}", "文件名": 文件名}


@应用.get("/缓存/{分类}/{文件名}")
def 缓存文件(分类: str, 文件名: str):
    """服务 数据缓存 对应目录下的上传文件（校验文件名安全）。"""
    子目录 = 缓存分类目录.get(分类)
    if not 子目录:
        raise HTTPException(404, "分类不存在")
    安全名 = Path(文件名).name
    目标 = (数据目录() / "数据缓存" / 子目录 / 安全名).resolve()
    if not 目标.is_file():
        raise HTTPException(404, "缓存文件不存在或已过期")
    扩展名 = 安全名.rsplit(".", 1)[-1].lower()
    return FileResponse(目标, media_type=媒体类型表.get(扩展名, "application/octet-stream"))


# ---------------- API：定制音色 ----------------
@应用.post("/api/定制音色")
async def 定制音色(文件们: List[UploadFile] = File(...), 名称: str = Form("emocompanion")):
    """上传数据集（可多个音频）→ 处理 → 筛选 → 注册 → 校验 完整流程。

    任一步失败不抛异常，错误写入 流程结果 并返回 200，便于前端展示失败原因。
    """
    from 定制音色 import 音色定制
    名称 = (名称 or "").strip() or "emocompanion"
    if not re.fullmatch(r"[A-Za-z0-9_]+", 名称):
        raise HTTPException(
            400, "定制音色名称仅支持英文、数字与下划线（音色注册接口要求），请重新命名")
    声音库目录 = 音色定制.默认声音库目录()
    数据集目录 = 声音库目录 / "数据集" / 名称
    数据集目录.mkdir(parents=True, exist_ok=True)
    for 序号, 文件 in enumerate(文件们 or [], start=1):
        原始名 = (文件.filename or f"片段{序号}.wav").replace("\\", "/").rsplit("/", 1)[-1]
        (数据集目录 / f"{序号:02d}_{原始名}").write_bytes(await 文件.read())
    流程结果 = 音色定制.执行完整流程(数据集目录, 声音库目录, 前缀=名称)
    音色ID = 流程结果.get("音色ID") or ""
    # 试听：把 最优片段.wav 复制到 音频缓存 用新 uuid 命名
    试听路径 = ""
    if 流程结果.get("最优片段路径"):
        try:
            源 = Path(流程结果["最优片段路径"])
            if 源.is_file():
                新文件名 = f"{uuid.uuid4().hex[:12]}_{int(time.time())}.wav"
                (音频目录 / 新文件名).write_bytes(源.read_bytes())
                试听路径 = f"/音频/{新文件名}"
        except Exception:
            试听路径 = ""
    # 选择报告
    选择报告 = None
    报告路径 = 流程结果.get("选择报告路径")
    if 报告路径 and Path(报告路径).is_file():
        try:
            选择报告 = json.loads(Path(报告路径).read_text(encoding="utf-8"))
        except Exception:
            选择报告 = None
    return {
        "ok": bool(音色ID),
        "流程结果": 流程结果,
        "音色ID": 音色ID,
        "试听路径": 试听路径,
        "选择报告": 选择报告,
    }


@应用.delete("/api/定制音色/删除")
def 删除定制音色(音色ID: str):
    """删除（隐藏）定制音色：不再出现在音色列表。"""
    配置持久化.隐藏音色(音色ID)
    return {"ok": True}


# ---------------- API：/spec 与 /goal 后端 ----------------
def _最新规格文件(文件名):
    """返回 .trae/specs 下最新（按修改时间）的 <子目录>/<文件名>，找不到返回 None。"""
    规格目录 = 项目根目录().parent / ".trae" / "specs"
    if not 规格目录.is_dir():
        return None
    匹配们 = sorted(规格目录.glob(f"*/{文件名}"),
                    key=lambda 路径: 路径.stat().st_mtime, reverse=True)
    return 匹配们[0] if 匹配们 else None


@应用.get("/api/系统/规格")
def 系统规格():
    """读取 .trae/specs 最新 spec.md，返回前 3000 字符摘要。"""
    规格文件 = _最新规格文件("spec.md")
    if not 规格文件:
        raise HTTPException(404, "未找到规格文件（.trae/specs/*/spec.md）")
    内容 = 规格文件.read_text(encoding="utf-8")
    return {"规格路径": str(规格文件), "规格摘要": 内容[:3000]}


@应用.get("/api/系统/目标")
def 系统目标():
    """解析 .trae/specs 最新 tasks.md，统计「任务 N」的完成进度。"""
    任务文件 = _最新规格文件("tasks.md")
    if not 任务文件:
        raise HTTPException(404, "未找到任务清单（.trae/specs/*/tasks.md）")
    任务清单 = []
    for 行 in 任务文件.read_text(encoding="utf-8").splitlines():
        行 = 行.strip()
        完成 = None
        if 行.startswith("- [x]"):
            完成 = True
        elif 行.startswith("- [ ]"):
            完成 = False
        if 完成 is None or not re.search(r"任务\s*\d", 行):
            continue
        任务清单.append({"标题": 行[5:].strip(), "完成": 完成})
    return {
        "规格路径": str(任务文件),
        "任务总数": len(任务清单),
        "已完成": sum(1 for 任务 in 任务清单 if 任务["完成"]),
        "任务清单": 任务清单,
    }
