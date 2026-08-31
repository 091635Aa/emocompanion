# -*- coding: utf-8 -*-
"""DashScope 多模态识别模块：语音/图像/视频/文本 的识别与理解。

请求格式调研结论（WebSearch 阿里云官方文档确认，2026-08）：

1) 多模态模型（qwen3.5-omni / qwen3-omni / qwen-vl-max）：
   - 端点：POST {接入地址}/services/aigc/multimodal-generation/generation
   - 请求体：{"model": 模型, "input": {"messages": [{"role": "user", "content": [
       {"image": "data:image/jpeg;base64,..."},
       {"audio": "data:audio/wav;base64,..."},
       {"video": "data:video/mp4;base64,..."},
       {"text": 问题}
     ]}]}}
   - 响应：output.choices[0].message.content（部分场景为 output.text，本模块同时兼容两种）
   - 注意：Qwen-VL 系列（如 qwen-vl-max）额外要求顶层 parameters 字段
     {"result_format": "message"}，本模块对模型名含 "vl" 时自动补充。

2) 语音识别模型（qwen-audio-3.0-asr 系列）：
   - 端点相同：POST {接入地址}/services/aigc/multimodal-generation/generation
   - 官方文档「非实时语音识别（Qwen-Audio-3.0-ASR-Flash/Fun-ASR-Flash）API 参考」
     （https://help.aliyun.com/zh/model-studio/non-real-time-speech-recognition-for-fun-asr-flash）：
     - input.messages[].content 为数组；音频项
       {"type": "input_audio", "input_audio": "data:audio/wav;base64,{base64}"}（Base64 Data URI）
     - 上下文项 {"type": "input_text", "input_text": {"text": "..."}}
     - 音频格式（wav/mp3/opus）放在 input.parameters.format
   - 响应：output.text（非流式返回完整转写文本）
   注：文档中的模型 ID 为 qwen-audio-3.0-asr-flash 等 flash 变体，
       “qwen-audio-3.0-asr” 为产品侧约定名，若当前账号不识别，
       会自动回退到 兼容模型列表 逐个重试。

3) 鉴权：请求头 Authorization: Bearer <DASHSCOPE_API_KEY>。
"""
import base64
import time
import uuid
from pathlib import Path

import requests
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from 环境配置 import 数据目录
from .语音合成 import 环境密钥, 多模态接口地址

# ---------------- 模型选择 ----------------
模型选择 = {
    "语音": "qwen-audio-3.0-asr",
    "图像": "qwen3.5-omni",
    "视频": "qwen3.5-omni",
    "通用": "qwen3.5-omni",
}
兼容模型列表 = ["qwen3.5-omni", "qwen3-omni", "qwen-vl-max", "qwen-audio-3.0-asr"]

# ---------------- 内容类型映射 ----------------
类型到字段 = {"图像": "image", "音频": "audio", "视频": "video", "文本": "text"}
类型前缀表 = {  # 多模态 content 数组的 data URI 前缀（数据均为 base64）
    "图像": "data:image/jpeg;base64,",
    "音频": "data:audio/wav;base64,",
    "视频": "data:video/mp4;base64,",
}

# ---------------- 上传限制 ----------------
图像大小上限 = 10 * 1024 * 1024   # 10MB
音频大小上限 = 20 * 1024 * 1024   # 20MB
视频大小上限 = 50 * 1024 * 1024   # 50MB
图像扩展名表 = {"jpg", "jpeg", "png", "webp"}
视频扩展名表 = {"mp4", "webm", "mov"}
音频扩展名表 = {"wav", "mp3", "m4a", "ogg"}


def _默认模型(内容列表):
    """按内容类型挑选默认模型：仅音频→语音，含图像→图像，含视频→视频，否则通用。"""
    类型们 = {(条目.get("类型") or "").strip() for 条目 in 内容列表}
    if 类型们 <= {"音频"}:
        return 模型选择["语音"]
    if "图像" in 类型们:
        return 模型选择["图像"]
    if "视频" in 类型们:
        return 模型选择["视频"]
    return 模型选择["通用"]


def _推测音频格式(数据):
    """根据 base64 音频数据头部魔数推测格式（wav/mp3/ogg/m4a），猜不到默认 wav。"""
    try:
        前16字 = (数据 or "")[:16]
        while len(前16字) % 4 != 0:
            前16字 += "="
        头部 = base64.b64decode(前16字)
    except Exception:
        return "wav"
    if 头部[:4] == b"RIFF":
        return "wav"
    if 头部[:4] == b"OggS":
        return "ogg"
    if 头部[:4] == b"ftyp":
        return "m4a"
    if 头部[:3] == b"ID3" or (len(头部) >= 2 and 头部[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        return "mp3"
    return "wav"


def _提取输出文本(输出块):
    """兼容两种响应结构：output.choices[0].message.content 或 output.text。"""
    choices = 输出块.get("choices") or []
    if choices:
        消息 = (choices[0] or {}).get("message") or {}
        内容 = 消息.get("content")
        if isinstance(内容, list):
            return "".join(
                (块.get("text") or "") if isinstance(块, dict) else str(块)
                for 块 in 内容).strip()
        if 内容:
            return str(内容).strip()
    return (输出块.get("text") or "").strip()


def _截断(文本, 长度=200):
    文本 = str(文本)
    return 文本[:长度] + ("…" if len(文本) > 长度 else "")


def _构建多模态载荷(内容列表, 问题):
    """多模态模型：input.messages[].content 数组（image/audio/video/text 块）。"""
    内容块们 = []
    for 条目 in 内容列表:
        类型 = (条目.get("类型") or "").strip()
        数据 = (条目.get("数据") or "").strip()
        if 类型 in 类型到字段 and 数据:
            前缀 = 类型前缀表.get(类型, "")
            内容块们.append({类型到字段[类型]: 前缀 + 数据})
    if (问题 or "").strip():
        内容块们.append({"text": (问题 or "").strip()})
    return {"messages": [{"role": "user", "content": 内容块们}]}


def _构建ASR载荷(内容列表, 问题):
    """语音识别模型（qwen-audio-3.0-asr）：input_audio + input_text 上下文的文档兼容格式。"""
    内容块们 = []
    音频格式 = "wav"
    for 条目 in 内容列表:
        类型 = (条目.get("类型") or "").strip()
        数据 = (条目.get("数据") or "").strip()
        if 类型 == "音频" and 数据:
            音频格式 = _推测音频格式(数据)
            内容块们.append({
                "type": "input_audio",
                "input_audio": f"data:audio/{音频格式};base64,{数据}",
            })
        elif 数据:
            内容块们.append({"type": "input_text", "input_text": {"text": 数据}})
    if (问题 or "").strip():
        内容块们.append({"type": "input_text", "input_text": {"text": (问题 or "").strip()}})
    return {"messages": [{"role": "user", "content": 内容块们}],
            "parameters": {"format": 音频格式}}


def _识别载荷(模型, 内容列表, 问题):
    """按模型类型选择请求体：ASR 模型用 input_audio 格式，其余用多模态 content 数组。"""
    if "asr" in 模型.lower():
        return _构建ASR载荷(内容列表, 问题)
    return _构建多模态载荷(内容列表, 问题)


def 识别(密钥="", 模型="", 内容列表=None, 问题="", 超时=180):
    """多模态识别主入口。

    内容列表元素 dict：{"类型": "图像"|"音频"|"视频"|"文本", "数据": base64字符串 或 文本内容}。
    优先使用指定/默认模型，失败后按 兼容模型列表 逐个重试。
    成功返回 {"文本": 模型输出文本, "请求ID": request_id}；全部模型失败抛 RuntimeError（含各尝试错误摘要）。
    """
    密钥 = (密钥 or "").strip() or 环境密钥()
    if not 密钥:
        raise RuntimeError("缺少 API Key：请检查 密钥配置.env 中的 DASHSCOPE_API_KEY")
    内容列表 = [条目 for 条目 in (内容列表 or []) if 条目 and (条目.get("数据") or "").strip()]
    if not 内容列表:
        raise RuntimeError("内容列表为空，没有可识别的内容")
    模型 = (模型 or "").strip() or _默认模型(内容列表)
    尝试模型们 = []
    for 候选 in [模型] + 兼容模型列表:
        if 候选 and 候选 not in 尝试模型们:
            尝试模型们.append(候选)
    请求头 = {"Authorization": f"Bearer {密钥}", "Content-Type": "application/json"}
    错误们 = []
    for 当前模型 in 尝试模型们:
        try:
            载荷 = {"model": 当前模型, "input": _识别载荷(当前模型, 内容列表, 问题)}
            if "vl" in 当前模型.lower():
                # Qwen-VL 系列原生接口要求顶层 parameters（result_format: message 返回消息格式）
                载荷["parameters"] = {"result_format": "message"}
            响应 = requests.post(多模态接口地址(), json=载荷, headers=请求头, timeout=超时)
        except requests.RequestException as 异常:
            错误们.append(f"{当前模型}: 网络错误 {异常}")
            continue
        if 响应.status_code != 200:
            错误们.append(f"{当前模型}: HTTP {响应.status_code} {_截断(响应.text)}")
            continue
        try:
            结果 = 响应.json()
        except ValueError:
            错误们.append(f"{当前模型}: 响应不是 JSON")
            continue
        输出块 = 结果.get("output") or {}
        文本 = _提取输出文本(输出块)
        if not 文本:
            错误们.append(f"{当前模型}: 响应中无文本内容")
            continue
        请求ID = 结果.get("request_id") or 输出块.get("request_id") or ""
        return {"文本": 文本, "请求ID": 请求ID}
    raise RuntimeError("多模态识别失败（已尝试模型：" + "、".join(尝试模型们) + "）：" + "；".join(错误们))


def _文件转base64(路径):
    文件 = Path(路径)
    if not 文件.is_file():
        raise RuntimeError(f"文件不存在: {文件}")
    return base64.b64encode(文件.read_bytes()).decode("ascii")


def 语音识别(音频路径, 密钥="", 问题="请转写这段语音内容"):
    """识别音频文件，返回 {"文本": ..., "请求ID": ...}。"""
    数据 = _文件转base64(音频路径)
    return 识别(密钥=密钥, 模型=模型选择["语音"],
                内容列表=[{"类型": "音频", "数据": 数据}], 问题=问题)


def 图像识别(图像路径, 密钥="", 问题="请描述这张图片的内容"):
    """识别图像文件，返回 {"文本": ..., "请求ID": ...}。"""
    数据 = _文件转base64(图像路径)
    return 识别(密钥=密钥, 模型=模型选择["图像"],
                内容列表=[{"类型": "图像", "数据": 数据}], 问题=问题)


def 视频识别(视频路径, 密钥="", 问题="请描述这段视频的内容"):
    """识别视频文件（≤50MB，先直接传 base64），返回 {"文本": ..., "请求ID": ...}。

    视频耗时较长，内部调用 识别 时超时放宽到 300 秒。
    """
    文件 = Path(视频路径)
    if not 文件.is_file():
        raise RuntimeError(f"文件不存在: {文件}")
    大小 = 文件.stat().st_size
    if 大小 > 视频大小上限:
        raise RuntimeError(f"视频文件过大（{大小 / 1024 / 1024:.1f}MB），超过 {视频大小上限 // 1024 // 1024}MB 限制")
    数据 = base64.b64encode(文件.read_bytes()).decode("ascii")
    return 识别(密钥=密钥, 模型=模型选择["视频"],
                内容列表=[{"类型": "视频", "数据": 数据}], 问题=问题, 超时=300)


# ---------------- FastAPI 路由 ----------------
识别路由 = APIRouter(tags=["多模态识别"])


def _保存上传文件(文件: UploadFile, 子目录名: str, 允许扩展名: set, 大小上限: int, 类型名: str):
    """校验文件类型与大小，保存到 数据目录()/数据缓存/<子目录名>，返回保存路径。"""
    原始名 = (文件.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    扩展名 = 原始名.rsplit(".", 1)[-1].lower() if "." in 原始名 else ""
    if 扩展名 not in 允许扩展名:
        raise HTTPException(400, f"不支持的{类型名}文件类型：{扩展名 or '未知'}（支持：{'/'.join(sorted(允许扩展名))}）")
    文件.file.seek(0, 2)
    大小 = 文件.file.tell()
    文件.file.seek(0)
    if 大小 > 大小上限:
        raise HTTPException(400, f"{类型名}文件过大（{大小 / 1024 / 1024:.1f}MB），超过 {大小上限 // 1024 // 1024}MB 限制")
    目录 = 数据目录() / "数据缓存" / 子目录名
    目录.mkdir(parents=True, exist_ok=True)
    目标 = 目录 / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{扩展名}"
    目标.write_bytes(文件.file.read())
    return 目标


@识别路由.post("/api/识别/语音")
async def 识别语音(文件: UploadFile = File(...), 问题: str = Form("请转写这段语音内容")):
    """上传音频（wav/mp3/m4a/ogg，≤20MB）→ 语音识别转写。"""
    try:
        目标 = _保存上传文件(文件, "音频缓存", 音频扩展名表, 音频大小上限, "音频")
        结果 = 语音识别(目标, 问题=问题)
        return {"ok": True, "类型": "语音", "结果": 结果["文本"],
                "路径": f"/缓存/音频/{目标.name}"}
    except HTTPException:
        raise
    except Exception as 异常:
        raise HTTPException(502, f"语音识别失败：{异常}")


@识别路由.post("/api/识别/图像")
async def 识别图像(文件: UploadFile = File(...), 问题: str = Form("请描述这张图片的内容")):
    """上传图像（jpg/jpeg/png/webp，≤10MB）→ 图像识别理解。"""
    try:
        目标 = _保存上传文件(文件, "上传图像缓存", 图像扩展名表, 图像大小上限, "图像")
        结果 = 图像识别(目标, 问题=问题)
        return {"ok": True, "类型": "图像", "结果": 结果["文本"]}
    except HTTPException:
        raise
    except Exception as 异常:
        raise HTTPException(502, f"图像识别失败：{异常}")


@识别路由.post("/api/识别/视频")
async def 识别视频(文件: UploadFile = File(...), 问题: str = Form("请描述这段视频的内容")):
    """上传视频（mp4/webm/mov，≤50MB）→ 视频识别理解（超时放宽 300 秒）。"""
    try:
        目标 = _保存上传文件(文件, "录制视频缓存", 视频扩展名表, 视频大小上限, "视频")
        结果 = 视频识别(目标, 问题=问题)
        return {"ok": True, "类型": "视频", "结果": 结果["文本"],
                "路径": f"/缓存/视频/{目标.name}"}
    except HTTPException:
        raise
    except Exception as 异常:
        raise HTTPException(502, f"视频识别失败：{异常}")
