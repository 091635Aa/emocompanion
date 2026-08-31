# -*- coding: utf-8 -*-
"""DeepSeek 客户端：AI 文本标注、指令优化、余额查询。"""
import os

import requests

接口地址 = "https://api.deepseek.com"
对话地址 = 接口地址 + "/chat/completions"
余额地址 = 接口地址 + "/user/balance"


def 环境密钥():
    return os.environ.get("DEEPSEEK_API_KEY", "").strip()


def _智能对话(密钥, 消息列表, 温度=0.5, 超时=120):
    """私有的 DeepSeek 对话请求，返回回答文本。"""
    密钥 = (密钥 or "").strip() or 环境密钥()
    if not 密钥:
        raise RuntimeError("缺少 DeepSeek API Key（请在 密钥配置.env 中配置 DEEPSEEK_API_KEY）")
    请求头 = {"Authorization": f"Bearer {密钥}", "Content-Type": "application/json"}
    载荷 = {
        "model": "deepseek-chat",
        "messages": 消息列表,
        "temperature": 温度,
        "stream": False,
    }
    响应 = requests.post(对话地址, json=载荷, headers=请求头, timeout=超时)
    if 响应.status_code != 200:
        try:
            消息 = 响应.json().get("error", {}).get("message") or 响应.text[:200]
        except Exception:
            消息 = (响应.text or "")[:200]
        raise RuntimeError(f"DeepSeek API HTTP {响应.status_code}: {消息}")
    结果 = 响应.json()
    try:
        return 结果["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"DeepSeek 响应异常: {str(结果)[:300]}")


标签规格说明 = """可用标签（仅限以下，[] 内必须是英文，不能自造）：
一、情绪控制标签（作用于其后文本，直到下一个控制标签）：
[sad]悲伤 [bored]无聊 [amazed]惊叹 [tired]疲惫 [deep and loud shouting]深沉大声呐喊 [scornful]轻蔑 [trembling]颤抖 [shouting]大喊 [angry]愤怒 [asmr]ASMR轻柔耳语 [excited]兴奋 [panicked]恐慌 [sarcastic]讽刺 [mischievously]调皮 [curious]好奇 [empathetic]共情 [like dracula]德古拉风格（低沉阴森） [whispers]耳语 [serious]严肃 [reluctantly]不情愿 [very slowly]非常缓慢 [crying]哭泣 [very fast]非常快速
二、富语言标签（在当前位置插入拟声效果，不影响前后情感）：
[gasp]倒吸一口气 [cough]咳嗽 [sighing]叹息 [giggles]咯咯笑 [clears throat]清嗓 [laughing]大笑 [snorts]哼声/嗤笑"""

标注系统提示词 = (
    "你是专业的中文语音合成（TTS）文本标注助手。用户会给你一段待朗读的文字，请你：\n"
    "1. 通读全文，理解整体情绪与每句话的语气；\n"
    "2. 只在情绪明显变化或拟声发生的位置插入标签，宁缺毋滥，不要每个句子都加；\n"
    "3. 情绪控制标签放在句子开头或情绪转折处；拟声标签放在动作对应的句子内部；\n"
    "4. 严禁改写、增删、润色用户的原文，只能插入 [] 标签；\n"
    "5. 只输出标注后的文本，不要任何解释、前言、后记或代码块标记。\n\n"
    + 标签规格说明
)

优化系统提示词 = (
    "你是语音合成指令（instruction）优化专家。用户会给出一条指令，请改写得更具体、多维、自然：\n"
    "- 遵循原则：具体而非模糊、多维而非单一、客观而非主观；简洁而非冗余；不模仿名人/特定演员；\n"
    "- 可参考维度：性别/年龄、音调、语速、情感、声音特点、用途场景，必要时可指定方言；\n"
    "- 只输出优化后的指令本身，不要解释、不要列表、不要引号。"
)


def 标注(密钥, 文本, 提示=""):
    """AI 自动识别语气，在原文中插入 [] 情绪/拟声标签。"""
    用户消息 = 文本
    if 提示 and 提示.strip():
        用户消息 += "\n\n额外语气要求（请按此风格标注）：" + 提示.strip()
    内容 = _智能对话(密钥, [
        {"role": "system", "content": 标注系统提示词},
        {"role": "user", "content": 用户消息},
    ], 温度=0.4)
    # 去掉可能的多余代码块包裹
    内容 = 内容.strip()
    if 内容.startswith("```"):
        行们 = 内容.splitlines()
        if 行们 and 行们[0].startswith("```"):
            行们 = 行们[1:]
        if 行们 and 行们[-1].strip() == "```":
            行们 = 行们[:-1]
        内容 = "\n".join(行们).strip()
    return {"text": 内容}


def 优化指令(密钥, 指令):
    """AI 优化全局风格指令。"""
    内容 = _智能对话(密钥, [
        {"role": "system", "content": 优化系统提示词},
        {"role": "user", "content": 指令},
    ], 温度=0.5)
    内容 = 内容.strip().strip('"').strip("`")
    return {"instruction": 内容}


def 查询余额(密钥):
    """查询 DeepSeek 账户余额（官方接口）。"""
    密钥 = (密钥 or "").strip() or 环境密钥()
    if not 密钥:
        raise RuntimeError("缺少 DeepSeek API Key（请在 密钥配置.env 中配置 DEEPSEEK_API_KEY）")
    请求头 = {"Authorization": f"Bearer {密钥}"}
    响应 = requests.get(余额地址, headers=请求头, timeout=30)
    if 响应.status_code != 200:
        try:
            消息 = 响应.json().get("error", {}).get("message") or 响应.text[:200]
        except Exception:
            消息 = (响应.text or "")[:200]
        raise RuntimeError(f"DeepSeek 余额查询失败 HTTP {响应.status_code}: {消息}")
    结果 = 响应.json()
    余额们 = []
    for 条目 in 结果.get("balance_infos") or []:
        余额们.append({
            "currency": 条目.get("currency", "CNY"),
            "total": 条目.get("total_balance"),
            "granted": 条目.get("granted_balance"),
            "topped_up": 条目.get("topped_up_balance"),
        })
    return {"is_available": bool(结果.get("is_available", False)), "balances": 余额们}
