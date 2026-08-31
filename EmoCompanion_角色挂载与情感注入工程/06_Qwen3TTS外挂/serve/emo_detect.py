# -*- coding: utf-8 -*-
"""EmoCompanion 情感自动识别（双路）：对话→TTS 时高精度自动插入情感

- 精度优先：LLM（复用 04 文本引擎运行中的 Qwen3-4B）严格结构化分类
- 兜底：领域词典关键词打分（文本引擎离线也可用）
- 情感词表与 tts_engine 对齐，保证自动识别结果可直接喂给合成器

对外:
  detect_emotion(reply_text, text_chat_fn=None) -> EmotionResult(label, confidence, source)
"""
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from tts_engine import EMOTION_VOCAB  # 对齐词表，避免标签不一致

EMOTIONS = list(EMOTION_VOCAB.keys())  # 开心/俏皮/悲伤/平静/兴奋/撒娇
DEFAULT = "平静"

# 情感 → 触发关键词（领域词典；命中越多/词越强得分越高）
_LEXICON = {
    "开心": ["开心", "高兴", "哈哈", "哈哈哈", "太棒", "好耶", "耶", "爽", "笑死", "爱死", "喜欢", "太爽", "棒呆",
             "哈喽", "欢迎", "欢迎来到", "家人们", "直播间", "陪大家", "好好", "开心"],
    "俏皮": ["嘻嘻", "嘿嘿", "调皮", "卖萌", "么么", "亲亲", "嘤嘤", "傲娇", "人家", "啾咪", "mua"],
    "悲伤": ["难过", "伤心", "想哭", "呜呜", "哭", "委屈", "心痛", "失落", "蓝瘦", "唉", "好难过", "伤心死"],
    "兴奋": ["太激动", "好兴奋", "冲啊", "起飞", "燃", "热血", "哇塞", "惊艳", "炸裂", "疯了"],
    "撒娇": ["撒娇", "抱抱", "要抱抱", "别这样", "讨厌啦", "人家不要", "哼", "不嘛", "小哥哥", "亲亲我"],
}
# 单字符弱词（避免误伤），单独处理
_WEAK = {"哼", "唉", "耶", "爽"}


@dataclass
class EmotionResult:
    label: str = DEFAULT
    confidence: float = 1.0
    source: str = "default"      # llm | lexicon | default
    raw: str = ""                # LLM 原始输出（诊断用）
    detail: dict = field(default_factory=dict)


# -------------------- 词典兜底 --------------------
def _lexicon_detect(text: str) -> EmotionResult:
    scores = {}
    for emo, words in _LEXICON.items():
        s = 0
        for w in words:
            if w in text:
                s += 1 if w not in _WEAK else 0.5
        scores[emo] = s
    # 得意 max + 覆盖比例
    top = max(scores, key=scores.get)
    total = sum(scores.values())
    if total <= 0:
        return EmotionResult(DEFAULT, 0.0, "default", "", scores)
    conf = round(min(0.9, 0.5 + scores[top] / total), 3)
    return EmotionResult(top, conf, "lexicon", "", scores)


# -------------------- LLM 严格分类（复用文本引擎） --------------------
_PROMPT_USER = (
    "你是情绪分类器。仅从集合 [{emotions}] 中选出一个最能表达下面句子情绪的标签，"
    "只输出标签本身，不要输出任何解释、标点或换行。\n句子：{reply}"
)
_PATTERNS = [
    re.compile(r"\[(开心|俏皮|悲伤|平静|兴奋|撒娇)\]"),
    re.compile(r"(开心|俏皮|悲伤|平静|兴奋|撒娇)"),
]


def _llm_detect(reply: str, text_chat_fn: Callable, user_msg: str = "") -> EmotionResult:
    """调用文本引擎做情绪分类。text_chat_fn 负责向 /chat 发请求并返回响应 JSON。
    优先结合用户消息(user_msg)判定用户真实情绪；无用户消息则仅按回复语气。"""
    try:
        if user_msg:
            user = _PROMPT_USER.format(emotions="/".join(EMOTIONS),
                                       user=user_msg[:1000], reply=reply[:1000])
        else:
            user = _PROMPT_REPLY_ONLY.format(emotions="/".join(EMOTIONS), reply=reply[:2000])
        resp = text_chat_fn({
            "messages": [
                {"role": "system", "content": "你是情绪分类器，只输出一个情感标签。"},
                {"role": "user", "content": user},
            ],
            "max_new": 8, "temperature": 0.1, "top_p": 0.9, "top_k": 8,
        })
        raw = _extract_text(resp) or ""
        raw = raw.strip()
        for pat in _PATTERNS:
            m = pat.search(raw)
            if m and m.group(1) in EMOTIONS:
                return EmotionResult(m.group(1), 0.99, "llm", raw)
        # LLM 未命中也给个低置信词典结果兜底
        fb = _lexicon_detect(reply)
        fb.source = "llm_fallback"
        fb.confidence = max(fb.confidence, 0.5)
        fb.raw = raw
        return fb
    except Exception:
        fb = _lexicon_detect(reply)
        fb.source = "lexicon"
        fb.raw = "llm_unavailable"
        return fb


def _extract_text(resp) -> str:
    """从文本引擎响应中稳健取回复文本。兼容 04 引擎 `reply` 与 OpenAI `choices/text` 两套协议。"""
    if isinstance(resp, dict):
        # 04 文本引擎返回: {"role": ..., "reply": "...", "usage": ...}
        if isinstance(resp.get("reply"), str) and resp["reply"]:
            return resp["reply"].strip()
        if isinstance(resp.get("text"), str) and resp["text"]:
            return resp["text"]
        choices = resp.get("choices")
        if choices:
            c = choices[0]
            if isinstance(c, dict) and "message" in c and isinstance(c["message"], dict):
                return c["message"].get("content", "")
    return ""


# -------------------- 对外入口 --------------------
def detect_emotion(reply: str, text_chat_fn: Optional[Callable] = None,
                   user_msg: str = "") -> EmotionResult:
    """对要求输出的情绪做判定。

    reply   : 助手回复（口播音文本）
    user_msg: 用户最近一条消息原义（语境，代表用户真实情绪）。优先据此判定，
              兼顾助手回复语气，避免助手 persona 掩盖用户心情。
    """
    reply = (reply or "").strip()
    uniq = (user_msg or "").strip()
    if not reply:
        if uniq:
            return _lexicon_detect(uniq)
        return EmotionResult(DEFAULT, 1.0, "default", "", {})
    if text_chat_fn is not None:
        return _llm_detect(reply, text_chat_fn, uniq)
    # 离线词典：合并用户语境(前置)与回复语气，两者情感词共同计分
    combined = (uniq + " " + reply) if uniq else reply
    return _lexicon_detect(combined)