# -*- coding: utf-8 -*-
"""DeepSeek 客户端：AI 文本标注、指令优化、余额查询。"""
import os
import requests

BASE = "https://api.deepseek.com"
CHAT_URL = BASE + "/chat/completions"
BAL_URL = BASE + "/user/balance"


def env_key():
    return os.environ.get("DEEPSEEK_API_KEY", "").strip()


def _chat(api_key, messages, temperature=0.5, timeout=120):
    key = (api_key or "").strip() or env_key()
    if not key:
        raise RuntimeError("缺少 DeepSeek API Key（请在 .env 中配置 DEEPSEEK_API_KEY）")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    r = requests.post(CHAT_URL, json=payload, headers=headers, timeout=timeout)
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message") or r.text[:200]
        except Exception:
            msg = (r.text or "")[:200]
        raise RuntimeError(f"DeepSeek API HTTP {r.status_code}: {msg}")
    j = r.json()
    try:
        return j["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"DeepSeek 响应异常: {str(j)[:300]}")


TAG_SPEC = """可用标签（仅限以下，[] 内必须是英文，不能自造）：
一、情绪控制标签（作用于其后文本，直到下一个控制标签）：
[sad]悲伤 [bored]无聊 [amazed]惊叹 [tired]疲惫 [deep and loud shouting]深沉大声呐喊 [scornful]轻蔑 [trembling]颤抖 [shouting]大喊 [angry]愤怒 [asmr]ASMR轻柔耳语 [excited]兴奋 [panicked]恐慌 [sarcastic]讽刺 [mischievously]调皮 [curious]好奇 [empathetic]共情 [like dracula]德古拉风格（低沉阴森） [whispers]耳语 [serious]严肃 [reluctantly]不情愿 [very slowly]非常缓慢 [crying]哭泣 [very fast]非常快速
二、富语言标签（在当前位置插入拟声效果，不影响前后情感）：
[gasp]倒吸一口气 [cough]咳嗽 [sighing]叹息 [giggles]咯咯笑 [clears throat]清嗓 [laughing]大笑 [snorts]哼声/嗤笑"""

ANNOTATE_SYSTEM = (
    "你是专业的中文语音合成（TTS）文本标注助手。用户会给你一段待朗读的文字，请你：\n"
    "1. 通读全文，理解整体情绪与每句话的语气；\n"
    "2. 只在情绪明显变化或拟声发生的位置插入标签，宁缺毋滥，不要每个句子都加；\n"
    "3. 情绪控制标签放在句子开头或情绪转折处；拟声标签放在动作对应的句子内部；\n"
    "4. 严禁改写、增删、润色用户的原文，只能插入 [] 标签；\n"
    "5. 只输出标注后的文本，不要任何解释、前言、后记或代码块标记。\n\n"
    + TAG_SPEC
)

OPTIMIZE_SYSTEM = (
    "你是语音合成指令（instruction）优化专家。用户会给出一条指令，请改写得更具体、多维、自然：\n"
    "- 遵循原则：具体而非模糊、多维而非单一、客观而非主观；简洁而非冗余；不模仿名人/特定演员；\n"
    "- 可参考维度：性别/年龄、音调、语速、情感、声音特点、用途场景，必要时可指定方言；\n"
    "- 只输出优化后的指令本身，不要解释、不要列表、不要引号。"
)


def annotate(api_key, text, hint=""):
    """AI 自动识别语气，在原文中插入 [] 情绪/拟声标签。"""
    user = text
    if hint and hint.strip():
        user += "\n\n额外语气要求（请按此风格标注）：" + hint.strip()
    content = _chat(api_key, [
        {"role": "system", "content": ANNOTATE_SYSTEM},
        {"role": "user", "content": user},
    ], temperature=0.4)
    # 去掉可能的多余代码块包裹
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return {"text": content}


def optimize_instruction(api_key, instruction):
    """AI 优化全局风格指令。"""
    content = _chat(api_key, [
        {"role": "system", "content": OPTIMIZE_SYSTEM},
        {"role": "user", "content": instruction},
    ], temperature=0.5)
    content = content.strip().strip('"').strip("`")
    return {"instruction": content}


def query_balance(api_key):
    """查询 DeepSeek 账户余额（官方接口）。"""
    key = (api_key or "").strip() or env_key()
    if not key:
        raise RuntimeError("缺少 DeepSeek API Key（请在 .env 中配置 DEEPSEEK_API_KEY）")
    headers = {"Authorization": f"Bearer {key}"}
    r = requests.get(BAL_URL, headers=headers, timeout=30)
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message") or r.text[:200]
        except Exception:
            msg = (r.text or "")[:200]
        raise RuntimeError(f"DeepSeek 余额查询失败 HTTP {r.status_code}: {msg}")
    j = r.json()
    balances = []
    for it in j.get("balance_infos") or []:
        balances.append({
            "currency": it.get("currency", "CNY"),
            "total": it.get("total_balance"),
            "granted": it.get("granted_balance"),
            "topped_up": it.get("topped_up_balance"),
        })
    return {"is_available": bool(j.get("is_available", False)), "balances": balances}
