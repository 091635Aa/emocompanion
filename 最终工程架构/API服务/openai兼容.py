# -*- coding: utf-8 -*-
"""
openai兼容 — OpenAI 风格兼容接口（/v1/*，标准英文键）
=====================================================
- GET  /v1/models            → OpenAI 模型列表格式
- POST /v1/chat/completions  → OpenAI 对话补全格式
- POST /v1/completions       → OpenAI 文本补全格式
复用 模型管理器.生成()；模型未加载时自动同步加载（加载并等待）。
"""
import sys
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

from 模型管理 import 管理器
from 开关 import 开关

router = APIRouter(prefix="/v1")


def _API关闭响应():
    return JSONResponse({"error": {"message": "API 接口已关闭（可在控制台开启）",
                                   "type": "api_disabled"}}, status_code=503)


def _检查API开关():
    if not 开关.启用API:
        return _API关闭响应()
    return None


class 消息(BaseModel):
    role: str = "user"
    content: str


class 对话请求(BaseModel):
    model: str = None
    messages: list[消息] = []
    max_tokens: int = 128
    temperature: float = None
    stream: bool = False


class 补全请求(BaseModel):
    model: str = None
    prompt: str = None
    max_tokens: int = 128
    temperature: float = None
    stream: bool = False


def _取模型名(请求模型):
    if 请求模型:
        return 请求模型
    if 管理器.已加载模型名:
        return 管理器.已加载模型名
    已注册 = 管理器.读取模型库()
    if 已注册:
        return 已注册[0]["模型名"]
    raise ValueError("未指定模型且无已注册模型")


def _确保加载(模型名):
    """未加载则同步加载（OpenAI 兼容端点阻塞等待）"""
    if 管理器.加载状态 != "已加载" or 管理器.已加载模型名 != 模型名:
        管理器.加载并等待(模型名)
    return 模型名


def _token统计(模型名, 文本, 步数):
    try:
        if 管理器.框架 is not None and 管理器.框架.tokenizer is not None:
            输入 = 管理器.框架.tokenizer(文本)
            return len(输入.input_ids), 步数 or 0
    except Exception:
        pass
    return max(1, len(文本) // 2), 步数 or 0


@router.get("/models")
def 模型列表():
    关闭 = _检查API开关()
    if 关闭:
        return 关闭
    data = []
    for 描述 in 管理器.读取模型库():
        data.append({
            "id": 描述["模型名"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "semantic-echo",
        })
    return {"object": "list", "data": data}


@router.post("/chat/completions")
def 对话补全(请求: 对话请求):
    关闭 = _检查API开关()
    if 关闭:
        return 关闭
    if not 请求.messages:
        return JSONResponse({"error": {"message": "messages 不能为空",
                                       "type": "invalid_request_error"}}, status_code=400)
    if 请求.stream:
        return JSONResponse({"error": {"message": "当前不支持流式输出",
                                       "type": "invalid_request_error"}}, status_code=400)
    try:
        模型名 = _取模型名(请求.model)
        _确保加载(模型名)
        提示词 = "\n".join(f"{m.role}: {m.content}" for m in 请求.messages if m.content)
        结果 = 管理器.生成(模型名, 提示词, 最大token=请求.max_tokens or 128)
        id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return {
            "id": id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": 模型名,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": 结果["文本"]},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": _token统计(模型名, 提示词, 0)[0],
                "completion_tokens": int(结果["步数"] or 0),
                "total_tokens": _token统计(模型名, 提示词, 结果["步数"])[0]
                                 + int(结果["步数"] or 0),
            },
        }
    except Exception as e:
        return JSONResponse({"error": {"message": str(e), "type": "server_error"}},
                            status_code=500)


@router.post("/completions")
def 文本补全(请求: 补全请求):
    关闭 = _检查API开关()
    if 关闭:
        return 关闭
    if not 请求.prompt:
        return JSONResponse({"error": {"message": "prompt 不能为空",
                                       "type": "invalid_request_error"}}, status_code=400)
    if 请求.stream:
        return JSONResponse({"error": {"message": "当前不支持流式输出",
                                       "type": "invalid_request_error"}}, status_code=400)
    try:
        模型名 = _取模型名(请求.model)
        _确保加载(模型名)
        结果 = 管理器.生成(模型名, 请求.prompt, 最大token=请求.max_tokens or 128)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": 模型名,
            "choices": [{
                "index": 0,
                "text": 结果["文本"],
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": _token统计(模型名, 请求.prompt, 0)[0],
                "completion_tokens": int(结果["步数"] or 0),
                "total_tokens": _token统计(模型名, 请求.prompt, 结果["步数"])[0]
                                 + int(结果["步数"] or 0),
            },
        }
    except Exception as e:
        return JSONResponse({"error": {"message": str(e), "type": "server_error"}},
                            status_code=500)
