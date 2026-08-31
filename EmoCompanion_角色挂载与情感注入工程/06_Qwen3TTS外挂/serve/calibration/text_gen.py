# -*- coding: utf-8 -*-
"""calibration/text_gen —— 模型外挂·文本生成环节（新句泛化）

闭环里补上"模型生成文本"这一步：golden 复述校准之外，用 04 文本引擎(Qwen3-4B)
把 golden 句改写为同情感、同口吻、语意近的新句，交给 TTS 外挂在"没见过的句子"上
也被检验(泛化)，避免只对训练过的句子复述得高分(overfit)。

情感/语速注入口径与 tts_engine 一致：
  - 情感：结构化字段传入 synthesize(emotion=...)，不往文本里塞标签(tokenizer 会口播)；
  - 语速：由进化权重 cfg['rate'] 统一 time-stretch 注入，文本层不重复控制。

可插拔：提供 --text-base 且引擎在线才启用；否则 augment() 自动回退原句(纯 golden 校准)。
协议与 unified_server._call_text_engine 一致：POST {text_base}/chat, {"messages":[...], 采样参数}
"""
import json
import urllib.error
import urllib.request

import re

_CLEAN = re.compile(r"[「」『』“”\"'《》【】]")


def call_llm(text_base: str, payload: dict, timeout: int = 300) -> dict:
    """向 04 文本引擎 /chat 发请求并返回响应 JSON。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(text_base.rstrip("/") + "/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_text(resp) -> str:
    """从引擎响应稳健取回复文本(兼容 text / choices[0].message.content)。"""
    if isinstance(resp, dict):
        if isinstance(resp.get("text"), str):
            return resp["text"].strip()
        ch = resp.get("choices")
        if ch and isinstance(ch, list) and isinstance(ch[0], dict):
            m = ch[0].get("message")
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                return m["content"].strip()
    return ""


def is_online(text_base: str, timeout: int = 5) -> bool:
    try:
        call_llm(text_base, {"messages": [{"role": "user", "content": "hi"}],
                             "max_new": 1}, timeout=timeout)
        return True
    except Exception:
        return False


def augment(text_base: str, text: str, emotion: str,
            max_len: int = 60, seed: int = 0) -> str:
    """把 golden 句改写为同情感新句。引擎不可用/输出不合规则回退原句(golden 复述)。"""
    sys_p = (
        "你是直播女主播。把下面的原句改写成语义相近、口吻相同的一句话直播话术，"
        f"整体表达情感【{emotion}】，口语自然、像萌系女主播现场说话，"
        f"控制在{max_len}字以内。只输出改写后的文本本身，不要任何标签、解释、引号或标点说明。"
    )
    try:
        resp = call_llm(text_base, {
            "messages": [{"role": "system", "content": sys_p},
                         {"role": "user", "content": (text or "").strip()}],
            "max_new": max_len + 32, "temperature": 0.85,
            "top_p": 0.92, "top_k": 40, "seed": seed,
        })
        s = _CLEAN.sub("", extract_text(resp)).strip()
        if 4 <= len(s) <= max_len + 10:
            return s
    except Exception:
        pass
    return text.strip()  # 退化：无引擎或失败直接复述 golden 句


def make_aug_set(text_base, by_emo_order, max_chars, aug_k, seed=0):
    """为整池构建"模型外挂"泛化样本，返回 (aug_pool, is_online_bool)。

    by_emo_order: 已按情感桶数量降序的对象(需反映每情感 local 最短 golden 样本)。
    每个 aug 样本带 '_wav' 指向对应情感最短 golden 音频 —— precompute 会用同一音频
    复用 _pros/_emb 作为韵律/音色标的，但 _text_clean 用 LLM 新句 —— 因此既检验新句
    泛化，又与原始样本可比(像这种情感、这种语速说话)。
    """
    from calibration import text_gen as tg
    if not (text_base and aug_k and aug_k > 0):
        return [], False
    if not tg.is_online(text_base):
        return [], False
    aug, emo_seen = [], set()
    for e, items in by_emo_order:
        if not items:
            continue
        ref = items[0]  # 每情感最短 golden 音频作为标的
        for k in range(aug_k):
            newt = tg.augment(text_base, ref["text"], e,
                              max_len=max_chars, seed=seed + k)
            aug.append({"text": newt, "emotion": e, "_wav": ref["_wav"]})
            emo_seen.add(e)
    return aug, True