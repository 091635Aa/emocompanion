# -*- coding: utf-8 -*-
"""质量校验：用注册成功的定制音色试合成一段语音，验证音色可用。"""
from pathlib import Path

from 核心模块.语音合成 import 合成

校验模型 = "qwen-audio-3.0-tts-plus"       # 与 音色注册.py 的绑定模型一致
校验文本 = "你好，我是缘圆。很高兴在这里遇见你。"
校验指令 = "用温柔、清澈、带一点俏皮的少女语气，自然地说话。"


def 校验(声音库目录, 音色ID):
    """用定制音色试合成，成功写入 声音库目录/合成输出/校验音频.wav。

    返回：
      {"通过": True, 路径, 大小, 请求ID}
      {"通过": False, 错误}
    """
    声音库目录 = Path(声音库目录)
    输出目录 = 声音库目录 / "合成输出"
    输出目录.mkdir(parents=True, exist_ok=True)
    try:
        音频, 请求ID = 合成(模型=校验模型, 音色=音色ID, 文本=校验文本, 指令=校验指令)
    except Exception as 异常:
        return {"通过": False, "错误": str(异常)[:2000]}
    路径 = 输出目录 / "校验音频.wav"
    路径.write_bytes(音频)
    return {"通过": True, "路径": str(路径), "大小": len(音频), "请求ID": 请求ID}
