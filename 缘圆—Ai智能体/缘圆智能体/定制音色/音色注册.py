# -*- coding: utf-8 -*-
"""音色注册：调用 DashScope 声音复刻接口（voice-enrollment / qwen-voice-enrollment）
把筛选出的最优片段注册为定制音色，并把音色 ID 写入 声音库目录/音色ID.txt。

注册策略（详见 克隆方案调研.md）：
  1. 首选（方案 A · base64 直传）：voice-enrollment / create_voice / prefix /
     audio.data（base64 data URI），备选 url 字段同样填 data URI；
     绑定模型 qwen-audio-3.0-tts-plus（与 质量校验.py 的试合成模型一致）。
  2. 次选（方案 B · 官方 base64 直传）：qwen-voice-enrollment / create /
     preferred_name / audio.data，绑定模型 qwen3-tts-vc-2026-01-22。

另提供 克隆()：直接采用 Qwen3.5-Omni 声音复刻（方案 C，复用 create 接口），
上传一段录音即克隆音色，无需训练。
"""
import base64
import json
from pathlib import Path

import requests

from 核心模块.语音合成 import 定制接口地址, 环境密钥

复刻模型A = "voice-enrollment"            # Qwen-Audio-TTS / CosyVoice 复刻模型
绑定模型A = "qwen-audio-3.0-tts-plus"     # 与 质量校验.py 的试合成模型一致
复刻模型B = "qwen-voice-enrollment"       # Qwen-TTS / Qwen3.5-Omni 复刻模型
绑定模型B = "qwen3-tts-vc-2026-01-22"     # Qwen-TTS 复刻绑定模型（检索确认）
欧姆绑定模型默认值 = "qwen3.5-omni-flash"  # Qwen3.5-Omni 复刻绑定模型（非实时）
语言提示默认值 = ["zh"]

请求超时秒 = 120


def _读取数据URI(路径):
    """把音频文件读成 data URI（data:{mime};base64,...）。"""
    路径 = Path(路径)
    if not 路径.is_file():
        raise FileNotFoundError(f"音频文件不存在：{路径}")
    mime映射 = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg",
                ".m4a": "audio/mp4", ".aac": "audio/aac", ".ogg": "audio/ogg"}
    类型 = mime映射.get(路径.suffix.lower(), "audio/wav")
    编码 = base64.b64encode(路径.read_bytes()).decode()
    return f"data:{类型};base64,{编码}"


def _发送(载荷):
    """带鉴权头 POST 到 定制接口地址()。"""
    请求头 = {"Authorization": f"Bearer {环境密钥()}", "Content-Type": "application/json"}
    响应 = requests.post(定制接口地址(), json=载荷, headers=请求头, timeout=请求超时秒)
    return 响应


def _解析音色ID(响应):
    """从响应中提取音色 ID（兼容 voice_id / voice 两种字段），返回 (音色ID, 响应摘要)。"""
    try:
        结果 = 响应.json()
    except ValueError:
        return "", 响应.text[:500]
    输出 = 结果.get("output") or {}
    音色ID = 输出.get("voice_id") or 输出.get("voice") or ""
    摘要 = json.dumps(结果, ensure_ascii=False)[:500]
    return 音色ID, 摘要


def 注册(最优片段路径, 声音库目录, 前缀="yuanyuan"):
    """把最优片段注册为定制音色（绑定 qwen-audio-3.0-tts-plus）。

    返回：
      {"成功": True, 音色ID, 方式, 绑定模型, 响应摘要}   —— 并把音色ID写入 音色ID.txt
      {"成功": False, "错误": ...}
    """
    声音库目录 = Path(声音库目录)
    声音库目录.mkdir(parents=True, exist_ok=True)
    密钥 = 环境密钥()
    if not 密钥:
        return {"成功": False, "错误": "未配置 DashScope API Key（请在 密钥配置.env 中配置 DASHSCOPE_API_KEY）"}

    数据URI = _读取数据URI(最优片段路径)
    尝试们 = [
        # 方案 A-1：voice-enrollment / create_voice / audio.data（base64 直传）
        ("voice-enrollment·audio.data", {
            "model": 复刻模型A,
            "input": {
                "action": "create_voice",
                "target_model": 绑定模型A,
                "prefix": 前缀,
                "language_hints": 语言提示默认值,
                "audio": {"data": 数据URI},
            },
        }),
        # 方案 A-2：voice-enrollment / create_voice / url 字段放 data URI
        ("voice-enrollment·url字段", {
            "model": 复刻模型A,
            "input": {
                "action": "create_voice",
                "target_model": 绑定模型A,
                "prefix": 前缀,
                "language_hints": 语言提示默认值,
                "url": 数据URI,
            },
        }),
        # 方案 B：qwen-voice-enrollment / create / preferred_name（官方 base64 直传）
        ("qwen-voice-enrollment·create", {
            "model": 复刻模型B,
            "input": {
                "action": "create",
                "target_model": 绑定模型B,
                "preferred_name": 前缀,
                "audio": {"data": 数据URI},
            },
        }),
    ]

    错误们 = []
    for 名称, 载荷 in 尝试们:
        try:
            响应 = _发送(载荷)
        except requests.RequestException as 异常:
            错误们.append(f"{名称} 网络错误：{异常}")
            continue
        if 响应.status_code != 200:
            错误们.append(f"{名称} HTTP {响应.status_code}：{响应.text[:200]}")
            continue
        音色ID, 摘要 = _解析音色ID(响应)
        if 音色ID:
            (声音库目录 / "音色ID.txt").write_text(音色ID, encoding="utf-8")
            return {"成功": True, "音色ID": 音色ID, "方式": 名称,
                    "绑定模型": 载荷["input"]["target_model"], "响应摘要": 摘要}
        错误们.append(f"{名称} 响应无音色ID：{摘要}")
    return {"成功": False, "错误": "；".join(错误们)[:2000]}


def 克隆(录音路径, 声音库目录, 名称, 目标模型=None):
    """Qwen3.5-Omni / Qwen-TTS 上传录音即克隆（qwen-voice-enrollment，base64 直传）。

    参数：
      录音路径   —— 10~20 秒的录音文件（wav/mp3/m4a 均可）
      声音库目录 —— 音色ID写入目录
      名称       —— 音色名称（preferred_name，仅字母数字，≤10 字符）
      目标模型   —— 绑定模型，默认 qwen3.5-omni-flash（可传 qwen3.5-omni-plus /
                     qwen3.5-omni-plus-realtime / qwen3.5-omni-flash-realtime /
                     qwen3-tts-vc-2026-01-22）

    返回：{"成功": True, 音色ID, 绑定模型, 响应摘要} 或 {"成功": False, "错误": ...}
    """
    目标模型 = 目标模型 or 欧姆绑定模型默认值
    声音库目录 = Path(声音库目录)
    声音库目录.mkdir(parents=True, exist_ok=True)
    密钥 = 环境密钥()
    if not 密钥:
        return {"成功": False, "错误": "未配置 DashScope API Key（请在 密钥配置.env 中配置 DASHSCOPE_API_KEY）"}

    数据URI = _读取数据URI(录音路径)
    载荷 = {
        "model": 复刻模型B,
        "input": {
            "action": "create",
            "target_model": 目标模型,
            "preferred_name": 名称,
            "audio": {"data": 数据URI},
        },
    }
    try:
        响应 = _发送(载荷)
    except requests.RequestException as 异常:
        return {"成功": False, "错误": f"网络错误：{异常}"}
    if 响应.status_code != 200:
        return {"成功": False, "错误": f"HTTP {响应.status_code}：{响应.text[:200]}"}
    音色ID, 摘要 = _解析音色ID(响应)
    if not 音色ID:
        return {"成功": False, "错误": f"响应无音色ID：{摘要}"}
    (声音库目录 / "音色ID.txt").write_text(音色ID, encoding="utf-8")
    return {"成功": True, "音色ID": 音色ID, "绑定模型": 目标模型, "响应摘要": 摘要}
