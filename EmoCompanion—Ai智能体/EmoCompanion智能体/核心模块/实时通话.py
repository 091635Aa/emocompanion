# -*- coding: utf-8 -*-
"""实时通话：DashScope Qwen-Omni Realtime 实时多模态通话桥接。

- 数据：实时模型列表、模型默认音色。
- 地址：实时接入地址() 按 环境变量 DASHSCOPE_REALTIME_URL → DASHSCOPE_BASE_URL 推导 → 兜底 拼接。
- 配置：配置会话() 构造 session.update JSON。
- 桥接：实时会话 类管理单个浏览器 ↔ DashScope 的 WebSocket 会话；
        通话路由 提供 /api/通话 WebSocket 端点，浏览器二进制音频/JSON 事件透明转发。
"""
import asyncio
import json
import logging
import os

import websockets
from fastapi import APIRouter, WebSocket

from . import 语音合成

日志器 = logging.getLogger("实时通话")

# ---------------- 实时模型列表 ----------------
实时模型列表 = [
    {"id": "qwen3.5-omni-plus-realtime", "名称": "千问3.5全模态", "默认音色": "Tina"},
    {"id": "qwen3-omni-flash-realtime", "名称": "千问3.0全模态", "默认音色": "Cherry"},
]
实时模型ID列表 = [模型["id"] for 模型 in 实时模型列表]
模型默认音色表 = {模型["id"]: 模型["默认音色"] for 模型 in 实时模型列表}


def 模型默认音色(模型ID):
    """返回模型默认音色，未知模型回退 Tina。"""
    return 模型默认音色表.get(模型ID, "Tina")


def 实时接入地址(模型ID="qwen3.5-omni-plus-realtime"):
    """拼接 DashScope Realtime WebSocket 地址，返回 {地址}?model={模型ID}。

    优先级：
    1. 环境变量 DASHSCOPE_REALTIME_URL（存在则原样使用）；
    2. 由 DASHSCOPE_BASE_URL 推导：https→wss、去掉 /api/v1 尾缀并追加 /api-ws/v1/realtime；
    3. 兜底 wss://dashscope.aliyuncs.com/api-ws/v1/realtime。

    业务空间专用域名格式（北京）：
        wss://{业务空间ID}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=...
    """
    地址 = os.environ.get("DASHSCOPE_REALTIME_URL", "").strip()
    if not 地址:
        基址 = 语音合成.接入地址()  # https://{业务空间域名或dashscope.aliyuncs.com}/api/v1
        if 基址.startswith("https://"):
            基址 = "wss://" + 基址[len("https://"):]
        elif 基址.startswith("http://"):
            基址 = "ws://" + 基址[len("http://"):]
        if 基址.endswith("/api/v1"):
            基址 = 基址[: -len("/api/v1")]
        地址 = 基址.rstrip("/") + "/api-ws/v1/realtime"
    分隔符 = "&" if "?" in 地址 else "?"
    return f"{地址}{分隔符}model={模型ID}"


def 候选接入地址们(模型ID="qwen3.5-omni-plus-realtime"):
    """返回实时接入候选地址列表（按优先级），去重。

    1. 环境变量 DASHSCOPE_REALTIME_URL 或由 DASHSCOPE_BASE_URL 推导的业务空间地址；
    2. 兜底官方域名 wss://dashscope.aliyuncs.com/api-ws/v1/realtime
       （业务空间域名瞬时 DNS 解析失败 / 网络波动时自动切换，官方域名仍可用 sk-ws- 密钥）。
    """
    地址们 = [实时接入地址(模型ID)]
    官方地址 = f"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={模型ID}"
    if 官方地址 not in 地址们:
        地址们.append(官方地址)
    return 地址们


def 配置会话(模型ID="qwen3.5-omni-plus-realtime", 音色="", 轮次检测类型="auto",
             系统指令="", 启用轮次检测=None, 回退轮次检测类型="server_vad"):
    """构造 session.update 事件 JSON（遵循 Qwen-Omni-Realtime 官方协议）。

    turn_detection.type 官方支持：
      - "server_vad"    声学 VAD（所有 Omni-Realtime 模型支持）
      - "semantic_vad"  语义 VAD（仅 qwen3.5-omni-realtime 系列支持，过滤语气词/背景音）
    传 "auto"（默认）按模型自动选择；传 "none" 禁用 VAD（手动触发回复）。
    注意："smart_turn" 是 Qwen-Audio Realtime 的取值，Omni-Realtime 不接受，统一按 auto 处理。
    """
    音色 = (音色 or "").strip() or 模型默认音色(模型ID)
    类型 = (轮次检测类型 or "auto").strip().lower()
    回退类型 = (回退轮次检测类型 or "server_vad").strip().lower()
    是35系列 = 模型ID in ("qwen3.5-omni-plus-realtime", "qwen3.5-omni-flash-realtime")
    是否回退 = False
    if 类型 == "smart_turn":   # 兼容旧参数：Omni-Realtime 不支持 smart_turn
        类型 = "auto"
    if 类型 in ("auto", ""):
        类型 = "semantic_vad" if 是35系列 else "server_vad"
    elif 类型 == "semantic_vad" and not 是35系列:
        类型 = 回退类型
        是否回退 = True
    if 启用轮次检测 is None:
        启用轮次检测 = 类型 != "none"
    if 类型 == "none":
        轮次检测配置 = None  # 禁用 VAD，改为手动触发
    else:
        轮次检测配置 = {"type": 类型, "silence_duration_ms": 800}
    会话配置 = {
        "output_modalities": ["text", "audio"],   # 文本 + 音频
        "voice": 音色,
        "input_audio_format": "pcm",              # 输入 16kHz PCM 单声道（官方唯一支持）
        "output_audio_format": "pcm",             # 输出 24kHz PCM 单声道（官方唯一支持）
        "turn_detection": 轮次检测配置,
        "enable_turn_detection": bool(启用轮次检测),
        # 输入音频实时转写：官方要求转写模型为 gummy-realtime-v1，
        # 服务端返回 conversation.item.input_audio_transcription.text/completed 事件
        "input_audio_transcription": {
            "model": "gummy-realtime-v1",
            "language": "zh",
        },
    }
    系统指令 = (系统指令 or "").strip()
    if 系统指令:
        会话配置["instructions"] = 系统指令
    if 是否回退:
        日志器.info("模型 %s 不支持轮次检测类型 %s，回退为 %s",
                    模型ID, 轮次检测类型, 类型)
    return {"type": "session.update", "session": 会话配置}


class 实时会话:
    """管理单个浏览器 ↔ DashScope Realtime 的桥接会话。"""

    def __init__(self, 模型ID="qwen3.5-omni-plus-realtime", 音色="",
                 轮次检测类型="smart_turn", 系统指令=""):
        self.模型ID = 模型ID
        self.音色 = (音色 or "").strip() or 模型默认音色(模型ID)
        self.轮次检测类型 = (轮次检测类型 or "smart_turn").strip()
        self.系统指令 = 系统指令 or ""
        self._连接 = None

    async def _建立连接(self, 地址, 密钥):
        """与 DashScope 建立 WebSocket 连接（兼容 websockets 新旧版本参数）。"""
        # websockets>=11 用 additional_headers；旧版本用 extra_headers
        try:
            return await websockets.connect(
                地址,
                additional_headers={"Authorization": f"Bearer {密钥}"},
                open_timeout=15,
            )
        except TypeError:
            return await websockets.connect(
                地址,
                extra_headers={"Authorization": f"Bearer {密钥}"},
                open_timeout=15,
            )

    async def 连接(self):
        """连接 DashScope Realtime WebSocket，连接成功后发送 session.update。

        依次尝试候选地址（业务空间地址 → 官方域名兜底），每个地址最多重试 2 次，
        抗瞬时 DNS 解析失败（getaddrinfo failed）与网络波动。
        """
        密钥 = 语音合成.环境密钥()
        if not 密钥:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY，请检查 密钥配置.env")
        候选们 = 候选接入地址们(self.模型ID)
        最后异常 = None
        for 地址 in 候选们:
            for 尝试 in range(2):
                try:
                    self._连接 = await self._建立连接(地址, 密钥)
                    break
                except Exception as 异常:
                    self._连接 = None
                    最后异常 = 异常
                    日志器.warning("连接 DashScope %s 失败（第 %s 次）：%s",
                                   地址, 尝试 + 1, 异常)
                    if 尝试 == 0:
                        await asyncio.sleep(1)
            if self._连接:
                break
            if 地址 != 候选们[-1]:
                日志器.warning("切换兜底地址…")
        if not self._连接:
            raise RuntimeError(f"连接 DashScope Realtime 失败：{最后异常}")
        会话事件 = 配置会话(self.模型ID, self.音色, self.轮次检测类型, self.系统指令)
        await self._连接.send(json.dumps(会话事件, ensure_ascii=False))

    async def _发送(self, 事件):
        """发送 JSON 事件到 DashScope。"""
        if not self._连接:
            raise RuntimeError("尚未连接 DashScope")
        await self._连接.send(json.dumps(事件, ensure_ascii=False))

    async def 发送文本(self, 内容):
        """发送用户文本：conversation.item.create（input_text）。"""
        await self._发送({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": 内容}],
            },
        })

    async def 发送音频(self, 字节):
        """发送 PCM16 16kHz 单声道音频。

        DashScope Realtime 服务端不接受裸二进制帧（会回 1003 unsupported data:
        BinaryWebSocketFrame is not supported），输入音频必须通过 JSON 事件
        input_audio_buffer.append 以 base64 编码发送。
        """
        if not 字节:
            return
        if not self._连接:
            raise RuntimeError("尚未连接 DashScope")
        import base64
        await self._发送({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(bytes(字节)).decode("ascii"),
        })

    async def 发送图像(self, 数据):
        """发送图像：接受裸 base64 或 data URL / http(s) URL。

        构造 conversation.item.create 事件，content 含
        {"type": "input_image", "image": "data:image/jpeg;base64,..."}。
        """
        数据 = (数据 or "").strip()
        if not 数据:
            return
        图像值 = 数据
        if not 数据.startswith(("data:", "http://", "https://")):
            图像值 = "data:image/jpeg;base64," + 数据  # 裸 base64 → data URL
        await self._发送({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image": 图像值}],
            },
        })

    async def 取消回复(self):
        """发送 response.cancel，支持手动打断（备用）。"""
        await self._发送({"type": "response.cancel"})

    async def 创建回复(self):
        """触发模型生成回复（response.create）。

        实时通话默认靠 VAD 检测语音自动回复；纯文字对话（无音频输入）时，
        发送文字后需手动触发 response.create 才会生成回复。
        """
        if self._连接:
            await self._发送({"type": "response.create"})

    async def 转发事件(self, 回调):
        """把 DashScope 服务端事件交给回调（异步函数，接收 dict）。

        - 文本事件 → json.loads 后回调；
        - 二进制帧 → 包装为 {"类型": "二进制", "数据": bytes} 后回调；
        - 异常/关闭 → 回调 {"类型": "关闭", "原因": ..., "内容": ...}。
        """
        if not self._连接:
            await 回调({"类型": "关闭", "原因": "未连接", "内容": "DashScope 未连接"})
            return
        try:
            async for 消息 in self._连接:
                if isinstance(消息, str):
                    try:
                        事件 = json.loads(消息)
                    except ValueError:
                        事件 = {"类型": "原始文本", "数据": 消息}
                    await 回调(事件)
                elif isinstance(消息, (bytes, bytearray)):
                    await 回调({"类型": "二进制", "数据": bytes(消息)})
        except Exception as 异常:
            try:
                await 回调({"类型": "关闭", "原因": "异常", "内容": str(异常)})
            except Exception:
                pass
        else:
            try:
                await 回调({"类型": "关闭", "原因": "连接已关闭", "内容": ""})
            except Exception:
                pass

    async def 关闭(self):
        """关闭与 DashScope 的连接。"""
        if self._连接:
            try:
                await self._连接.close()
            except Exception:
                pass
            self._连接 = None


# ---------------- 浏览器 ↔ DashScope 桥接路由 ----------------
通话路由 = APIRouter()


@通话路由.websocket("/api/通话")
async def 通话端点(websocket: WebSocket,
                   model: str = "qwen3.5-omni-plus-realtime",
                   voice: str = "",
                   turn_detection: str = "smart_turn"):
    """实时通话桥接：浏览器连接后，后端立即连接 DashScope。

    浏览器 → 后端：二进制帧 = PCM16 16k 音频；
                   JSON {"类型":"文本","内容":...} / {"类型":"图像","数据":base64} / {"类型":"取消"}
    后端 → 浏览器：DashScope 服务端事件原样转发；连接/错误时发 {"类型":"状态",...}。
    断开任一连接即清理并关闭另一端（try/finally 保证资源释放）。
    """
    await websocket.accept()
    try:
        await websocket.send_json({
            "类型": "状态", "状态": "连接中",
            "内容": f"正在连接模型 {model}…",
        })
    except Exception:
        return

    会话 = 实时会话(模型ID=model, 音色=voice, 轮次检测类型=turn_detection)
    try:
        await 会话.连接()
    except Exception as 异常:
        日志器.warning("连接 DashScope 失败: %s", 异常)
        try:
            await websocket.send_json({
                "类型": "状态", "状态": "错误",
                "内容": f"连接 DashScope 失败: {异常}",
            })
        except Exception:
            pass
        finally:
            await 会话.关闭()
            try:
                await websocket.close()
            except Exception:
                pass
        return

    try:
        await websocket.send_json({
            "类型": "状态", "状态": "已连接",
            "内容": f"已连接 {model}，请说话（建议佩戴耳机避免回声触发打断）",
        })
    except Exception:
        await 会话.关闭()
        return

    async def 转发到浏览器(事件):
        """把 DashScope 事件转发给浏览器；DashScope 断开时通知浏览器并关闭。"""
        try:
            if isinstance(事件, dict):
                if 事件.get("类型") == "关闭":
                    await websocket.send_json({
                        "类型": "状态", "状态": "已关闭",
                        "内容": 事件.get("内容") or "服务端连接已断开",
                    })
                    await websocket.close()
                else:
                    await websocket.send_json(事件)
            elif isinstance(事件, (bytes, bytearray)):
                await websocket.send_bytes(bytes(事件))
        except Exception:
            pass

    转发任务 = asyncio.create_task(会话.转发事件(转发到浏览器))
    try:
        while True:
            消息 = await websocket.receive()
            类型 = 消息.get("type")
            if 类型 == "websocket.disconnect":
                break
            文本 = 消息.get("text")
            if 文本 is not None:
                try:
                    事件 = json.loads(文本)
                except ValueError:
                    continue
                事件类型 = 事件.get("类型")
                if 事件类型 == "文本":
                    await 会话.发送文本(事件.get("内容", ""))
                    await 会话.创建回复()   # 文字对话：手动触发模型回复
                elif 事件类型 == "图像":
                    await 会话.发送图像(事件.get("数据", ""))
                elif 事件类型 == "取消":
                    await 会话.取消回复()
                continue
            字节 = 消息.get("bytes")
            if 字节 is not None:
                await 会话.发送音频(字节)
    except Exception:
        pass
    finally:
        转发任务.cancel()
        try:
            await 转发任务
        except (asyncio.CancelledError, Exception):
            pass  # CancelledError 在 3.8+ 继承 BaseException，需显式捕获
        await 会话.关闭()
        try:
            await websocket.send_json({"类型": "状态", "状态": "已关闭", "内容": "通话已结束"})
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
