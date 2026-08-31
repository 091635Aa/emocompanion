# -*- coding: utf-8 -*-
"""
推理引擎模块（RPG 精灵素材打标）

功能：调用 f:\\打标 已部署的 30B Qwen3-Omni llama-server 服务
（OpenAI 兼容 /v1/chat/completions，地址 http://127.0.0.1:8766）对图片进行智能打标。
本项目不加载模型、不使用 torch，仅通过标准库 urllib 发起 HTTP 请求。

接口（供打标流水线调用）：
    服务可用() -> bool
    初始化() -> dict
    分析图片(图片路径, 素材信息=None) -> {原始文本, 处理耗时秒, 模型名}
    当前模型信息() -> dict
    释放模型() -> None（占位，无操作）
"""

import os
import sys
import time
import json
import base64
import io
import urllib.request
import urllib.error

# 将项目根目录加入模块搜索路径，保证中文模块名可导入
项目根目录 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if 项目根目录 not in sys.path:
    sys.path.append(项目根目录)

# 统一 stdout 编码，避免中文打印乱码
sys.stdout.reconfigure(encoding="utf-8")

# 图片编码相关常量
最大边长 = 1024          # 图片缩放上限：最长边超过该值则等比缩小，节省请求体积
数据URL前缀 = "data:image/png;base64,"

# 模块级当前服务状态（单例）
_全局状态 = {
    "模型名": None,
    "服务地址": None,
    "状态": "未初始化",
    "是否降级": False,
}

# 服务不可用时的统一错误提示（提示先启动 f:\打标 的 30B 服务）
_服务不可用提示 = (
    "推理服务不可用（{}）。请先启动 f:\\打标 的 30B Qwen3-Omni llama-server 服务"
    "（地址 http://127.0.0.1:8766），待 /health 返回 200 后再重试。"
)


def _获取服务配置():
    """
    从系统配置读取模型服务相关配置。

    系统配置"模型"段结构：
        {"服务地址", "模型名", "最大输出长度", "生成温度", "top_p", "请求超时秒"}
    """
    from 业务逻辑层.配置管理 import 获取配置
    配置 = 获取配置()
    模型配置 = 配置.get("模型", {}) or {}
    return {
        "服务地址": str(模型配置.get("服务地址", "http://127.0.0.1:8766")).rstrip("/"),
        "模型名": 模型配置.get("模型名", "Qwen3-Omni-30B-A3B-Instruct（llama-server）"),
        "最大输出长度": int(模型配置.get("最大输出长度", 1024)),
        "生成温度": float(模型配置.get("生成温度", 0.5)),
        "top_p": float(模型配置.get("top_p", 0.9)),
        "请求超时秒": int(模型配置.get("请求超时秒", 300)),
    }


def 服务可用():
    """检测推理服务是否可用：GET {服务地址}/health，2 秒超时，200 返回 True。"""
    服务配置 = _获取服务配置()
    try:
        with urllib.request.urlopen(服务配置["服务地址"] + "/health", timeout=2) as 响应:
            return 响应.status == 200
    except Exception:
        return False


def 初始化():
    """
    初始化推理引擎：检查服务可用性并记录当前模型信息。

    返回 {模型名, 服务地址, 状态, 是否降级:False}；
    服务不可用时状态为"服务不可用"（不抛异常）。
    """
    global _全局状态
    服务配置 = _获取服务配置()
    if 服务可用():
        _全局状态 = {
            "模型名": 服务配置["模型名"],
            "服务地址": 服务配置["服务地址"],
            "状态": "可用",
            "是否降级": False,
        }
    else:
        _全局状态 = {
            "模型名": 服务配置["模型名"],
            "服务地址": 服务配置["服务地址"],
            "状态": "服务不可用",
            "是否降级": False,
        }
    return dict(_全局状态)


def _编码图片为数据URL(图片路径, 最长边限制=最大边长):
    """
    用 Pillow 打开图片：转 RGB，最长边超过限制时等比缩小，
    保存为 PNG bytes 到内存并编码为 data URL（data:image/png;base64,...）。
    """
    from PIL import Image

    with Image.open(图片路径) as 图片:
        # 统一转 RGB：RGBA/P 等模式也转为三通道，避免透明 PNG 解码与尺寸判断歧义
        图片 = 图片.convert("RGB")
        宽, 高 = 图片.size
        最长边实际 = max(宽, 高)
        if 最长边实际 > 最长边限制:
            缩放比例 = 最长边限制 / 最长边实际
            新宽 = max(1, int(宽 * 缩放比例))
            新高 = max(1, int(高 * 缩放比例))
            图片 = 图片.resize((新宽, 新高), Image.LANCZOS)
        缓冲 = io.BytesIO()
        图片.save(缓冲, format="PNG")
        图片字节 = 缓冲.getvalue()
    return 数据URL前缀 + base64.b64encode(图片字节).decode("ascii")


def 分析图片(图片路径, 素材信息=None):
    """
    对单张分割后的精灵图执行 AI 打标（调用 f:\打标 的 30B llama-server 服务）。

    入参：
    - 图片路径：分割后的单格 PNG 图片路径
    - 素材信息：可选 dict（含 类型、名称、来源文件、切割坐标 等，透传给提示词工程）

    返回：{原始文本, 处理耗时秒, 模型名}
    图片不存在抛 FileNotFoundError；服务不可用 / HTTP 错误 / 响应解析失败抛 RuntimeError
    （错误信息包含服务地址提示）。
    """
    素材信息 = 素材信息 if isinstance(素材信息, dict) else {}
    服务配置 = _获取服务配置()
    服务地址 = 服务配置["服务地址"]

    # 1. 校验图片存在
    if not os.path.isfile(图片路径):
        raise FileNotFoundError("图片文件不存在：{}".format(图片路径))

    # 2. 服务不可用 → 明确报错（提示先启动 f:\打标 的 30B 服务）
    if not 服务可用():
        raise RuntimeError(_服务不可用提示.format(服务地址))

    # 3. 构造提示词（系统 + 用户），用户内容为文本 + 图片 data URL
    from 业务逻辑层.提示词工程 import 构造系统提示词, 构造分析提示词
    系统提示词 = 构造系统提示词()
    用户提示词 = 构造分析提示词(素材信息)
    数据URL = _编码图片为数据URL(图片路径)

    消息列表 = [
        {"role": "system", "content": 系统提示词},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": 用户提示词},
                {"type": "image_url", "image_url": {"url": 数据URL}},
            ],
        },
    ]

    # 4. 构造请求体并 POST /v1/chat/completions（非流式）
    请求体字典 = {
        "model": 服务配置["模型名"],
        "messages": 消息列表,
        "max_tokens": 服务配置["最大输出长度"],
        "temperature": 服务配置["生成温度"],
        "top_p": 服务配置["top_p"],
        "stream": False,
    }
    请求体 = json.dumps(请求体字典).encode("utf-8")
    请求 = urllib.request.Request(
        服务地址 + "/v1/chat/completions",
        data=请求体,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    开始时间 = time.time()
    try:
        with urllib.request.urlopen(请求, timeout=服务配置["请求超时秒"]) as 响应:
            响应字节 = 响应.read()
        # 注意编码问题：响应可能含中文，统一 utf-8 解码并容错替换
        响应文本 = 响应字节.decode("utf-8", errors="replace")
        处理耗时秒 = round(time.time() - 开始时间, 2)
    except urllib.error.HTTPError as 异常:
        # HTTP 错误：附带服务地址与响应详情，便于定位
        错误详情 = ""
        try:
            错误详情 = 异常.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        raise RuntimeError(
            "推理服务 HTTP 错误（{}，HTTP {}）：{} {}".format(
                服务地址, 异常.code, 异常.reason, 错误详情
            )
        ) from 异常
    except Exception as 异常:
        raise RuntimeError(
            "请求推理服务失败（{}）：{}。请确认 f:\\打标 的 30B llama-server 服务已启动。".format(
                服务地址, 异常
            )
        ) from 异常

    # 5. 解析响应：choices[0].message.content
    try:
        响应对象 = json.loads(响应文本)
    except (json.JSONDecodeError, ValueError) as 异常:
        raise RuntimeError(
            "解析推理服务响应失败（{}）：{}。响应片段：{}".format(
                服务地址, 异常, 响应文本[:300]
            )
        ) from 异常
    选择列表 = 响应对象.get("choices") or []
    if not 选择列表 or not isinstance(选择列表[0], dict):
        raise RuntimeError(
            "推理服务响应缺少 choices（{}）。响应片段：{}".format(服务地址, 响应文本[:300])
        )
    消息 = 选择列表[0].get("message") or {}
    原始文本 = 消息.get("content")
    if not isinstance(原始文本, str) or not 原始文本.strip():
        raise RuntimeError(
            "推理服务响应缺少文本内容（{}）。响应片段：{}".format(服务地址, 响应文本[:300])
        )
    原始文本 = 原始文本.strip()

    return {
        "原始文本": 原始文本,
        "处理耗时秒": 处理耗时秒,
        "模型名": 服务配置["模型名"],
    }


def 当前模型信息():
    """返回当前推理服务信息 dict（未初始化时为默认状态）。"""
    return dict(_全局状态)


def 释放模型():
    """释放模型资源（本项目为 HTTP 服务模式，无本地加载资源，占位无操作）。"""
    return None


if __name__ == "__main__":
    print("[测试] 推理引擎模块加载成功")
    try:
        print("初始化：", 初始化())
        print("当前模型信息：", 当前模型信息())
        if not 服务可用():
            print("[提示] 30B llama-server 服务当前不可用，请先启动 f:\\打标 的服务后再调用 分析图片()")
    except Exception as 异常:
        print("[测试] 推理引擎自检失败：{}".format(异常))
