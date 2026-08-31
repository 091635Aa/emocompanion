# -*- coding: utf-8 -*-
"""DashScope Qwen-Audio TTS 客户端：支持的模型、标签表、音色列表、语音合成。"""
import base64
import os
import pathlib
import re
import time

import requests

BASE_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


# ---------------- 接入地址（业务空间专用域名或普通百炼域名） ----------------
def dashscope_base():
    """DashScope 原生 API 根地址，可在 .env 的 DASHSCOPE_BASE_URL 覆盖。"""
    return os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/api/v1").rstrip("/")


def compatible_base():
    """OpenAI 兼容地址（用于探测 Key 有效性）。"""
    b = dashscope_base()
    if b.endswith("/api/v1"):
        return b[: -len("/api/v1")] + "/compatible-mode/v1"
    return b.rstrip("/") + "/compatible-mode/v1"


def _synth_url():
    return dashscope_base() + "/services/audio/tts/SpeechSynthesizer"


def _mm_url():
    return dashscope_base() + "/services/aigc/multimodal-generation/generation"


def _custom_url():
    return dashscope_base() + "/services/audio/tts/customization"


# ---------------- 支持的模型（可查询、可重命名别名） ----------------
SUPPORTED_MODELS = [
    {"id": "qwen-audio-3.0-tts-plus", "tags": True, "instruction": True,
     "note": "推荐 · 支持情绪/富语言标签与指令（复刻音色正用这个）"},
    {"id": "qwen-audio-3.0-tts-flash", "tags": True, "instruction": True,
     "note": "旧版快版 · 同样支持标签与指令"},
    {"id": "qwen3-tts-instruct-flash", "tags": True, "instruction": True,
     "note": "千问3 TTS · 支持自然语言指令（配 Cherry/Serena/Ethan 系统音色）"},
    {"id": "qwen3-tts-flash", "tags": True, "instruction": True,
     "note": "千问3 TTS 快版 · 配 Cherry/Serena/Ethan 系统音色"},
    {"id": "qwen-tts", "tags": False, "instruction": False,
     "note": "标准版 · 不支持标签/指令"},
    {"id": "qwen-tts-flash", "tags": False, "instruction": False,
     "note": "标准快版 · 不支持标签/指令"},
    {"id": "cosyvoice-v2", "tags": False, "instruction": False,
     "note": "CosyVoice 系列 · 不支持标签/指令"},
    {"id": "cosyvoice-v1", "tags": False, "instruction": False,
     "note": "CosyVoice 旧版 · 不支持标签/指令"},
]
MODEL_IDS = [m["id"] for m in SUPPORTED_MODELS]

# ---------------- 完整标签表（官方文档 30 个） ----------------
CONTROL_TAGS = [  # 控制类（23）：设定情绪/风格，作用于其后文本，直到下一个控制标签
    ("sad", "悲伤"), ("bored", "无聊"), ("amazed", "惊叹"), ("tired", "疲惫"),
    ("deep and loud shouting", "深沉大声呐喊"), ("scornful", "轻蔑"),
    ("trembling", "颤抖"), ("shouting", "大喊"), ("angry", "愤怒"),
    ("asmr", "ASMR 轻柔耳语"), ("excited", "兴奋"), ("panicked", "恐慌"),
    ("sarcastic", "讽刺"), ("mischievously", "调皮"), ("curious", "好奇"),
    ("empathetic", "共情"), ("like dracula", "德古拉风格（低沉阴森）"),
    ("whispers", "耳语"), ("serious", "严肃"), ("reluctantly", "不情愿"),
    ("very slowly", "非常缓慢"), ("crying", "哭泣"), ("very fast", "非常快速"),
]
RICH_TAGS = [  # 富语言类（7）：在当前位置插入拟声效果，不影响前后情感
    ("gasp", "倒吸一口气"), ("cough", "咳嗽"), ("sighing", "叹息"),
    ("giggles", "咯咯笑"), ("clears throat", "清嗓"), ("laughing", "大笑"),
    ("snorts", "哼声/嗤笑"),
]
ALL_TAG_NAMES = [t[0] for t in CONTROL_TAGS] + [t[0] for t in RICH_TAGS]
TAG_RE = re.compile(r"\[([^\[\]]+)\]")

# 中文标签含义 -> 英文标签（用于把用户手写的 [悲伤] 转成 [sad]）
ZH_TO_EN = {zh: en for en, zh in CONTROL_TAGS}
ZH_TO_EN.update({zh: en for en, zh in RICH_TAGS})


def normalize_tags(text):
    """把用户手写的中文标签（如 [悲伤]）转成 API 识别的英文标签（[sad]）。"""
    out = text or ""
    for zh, en in ZH_TO_EN.items():
        out = out.replace("[" + zh + "]", "[" + en + "]")
    return out


def apply_pronunciations(text, prons):
    """给指定词自动加上注音标签，纠正多音字读音。

    prons: [{"word": "缘圆", "ph": "yuan3 yuan4"}, ...]
    输出如：<phoneme alphabet="py" ph="yuan3 yuan4">缘圆</phoneme>
    """
    out = text or ""
    if not prons:
        return out
    # 长的词优先替换，避免子串抢先被替换
    for p in sorted(prons, key=lambda x: len(x.get("word", "")), reverse=True):
        word = (p.get("word") or "").strip()
        ph = (p.get("ph") or "").strip()
        if not word or not ph:
            continue
        out = out.replace(word, f'<phoneme alphabet="py" ph="{ph}">{word}</phoneme>')
    return out


# ---------------- 声音维度表（用于「声音风格助手」一键生成指令） ----------------
VOICE_DIMENSIONS = {
    "gender": {"label": "性别", "options": ["男性", "女性", "中性"]},
    "age": {"label": "年龄",
            "options": ["儿童（5-12岁）", "青少年（13-18岁）", "青年（19-35岁）",
                        "中年（36-55岁）", "老年（55岁以上）"]},
    "pitch": {"label": "音调", "options": ["高音", "中音", "低音", "偏高", "偏低"]},
    "speed": {"label": "语速", "options": ["快速", "中速", "缓慢", "偏快", "偏慢"]},
    "emotion": {"label": "情感",
                "options": ["开朗", "沉稳", "温柔", "严肃", "活泼", "冷静", "治愈"]},
    "trait": {"label": "特点",
              "options": ["有磁性", "清脆", "沙哑", "圆润", "甜美", "浑厚", "有力"]},
    "scene": {"label": "用途",
              "options": ["新闻播报", "广告配音", "有声书", "动画角色", "语音助手", "纪录片解说"]},
}

SCENE_PRESETS = [  # 场景预设：一键填充各维度
    {"name": "语音助手",
     "dims": {"gender": "女性", "age": "青年（19-35岁）", "pitch": "偏高",
              "speed": "中速", "emotion": "温柔", "trait": "圆润", "scene": "语音助手"}},
    {"name": "动画角色",
     "dims": {"gender": "女性", "age": "儿童（5-12岁）", "pitch": "高音",
              "speed": "快速", "emotion": "活泼", "trait": "清脆", "scene": "动画角色"}},
    {"name": "新闻播报",
     "dims": {"gender": "中性", "age": "中年（36-55岁）", "pitch": "中音",
              "speed": "中速", "emotion": "沉稳", "trait": "浑厚", "scene": "新闻播报"}},
    {"name": "广告配音",
     "dims": {"gender": "女性", "age": "青年（19-35岁）", "pitch": "偏高",
              "speed": "快速", "emotion": "开朗", "trait": "甜美", "scene": "广告配音"}},
    {"name": "有声书",
     "dims": {"gender": "中性", "age": "中年（36-55岁）", "pitch": "中音",
              "speed": "缓慢", "emotion": "治愈", "trait": "圆润", "scene": "有声书"}},
    {"name": "纪录片解说",
     "dims": {"gender": "男性", "age": "中年（36-55岁）", "pitch": "低音",
              "speed": "缓慢", "emotion": "沉稳", "trait": "浑厚", "scene": "纪录片解说"}},
]

# ---------------- 系统音色 ----------------
# 前几组为 qwen3-tts 系列（业务空间账号用）；后面为 qwen-audio 系列（旧版模型用）。
SYSTEM_VOICES = [
    {"id": "Cherry", "name": "Cherry · 中文女声（推荐）", "lang": "中文"},
    {"id": "Serena", "name": "Serena · 中文女声", "lang": "中文"},
    {"id": "Ethan", "name": "Ethan · 中文男声", "lang": "中文"},
    {"id": "longxia_v3.6", "name": "小夏 · 中文女声（qwen-audio 系列）", "lang": "中文"},
    {"id": "longxiaochun_v3.6", "name": "小晓春 · 中文女声（qwen-audio 系列）", "lang": "中文"},
    {"id": "longanyu_v3.6", "name": "小安雨 · 中文女声（qwen-audio 系列）", "lang": "中文"},
    {"id": "longanyi_v3.6", "name": "小安忆 · 中文女声（qwen-audio 系列）", "lang": "中文"},
    {"id": "longhua_v3.6", "name": "小华 · 中文男声（qwen-audio 系列）", "lang": "中文"},
    {"id": "longhaonan_v3.6", "name": "小浩男 · 中文男声（qwen-audio 系列）", "lang": "中文"},
    {"id": "longjoel_v3.6", "name": "乔尔 · 英文男声（qwen-audio 系列）", "lang": "英文"},
    {"id": "longerin_v3.6", "name": "艾琳 · 英文女声（qwen-audio 系列）", "lang": "英文"},
    {"id": "longanna_v3.6", "name": "安娜 · 英文女声（qwen-audio 系列）", "lang": "英文"},
]

# ---------------- 本地复刻音色 ----------------
VOICE_ID_NAMES = ["voice_id.txt", "voice_id_v1.txt", "voice_id_v2.txt"]


def voice_id_files():
    """复刻音色 ID 文件：优先 EXE 旁（用户可改），其次打包资源。"""
    import config  # 局部导入，避免循环依赖
    out = []
    for name in VOICE_ID_NAMES:
        for base in (config.app_dir(), config.res_dir()):
            p = base / "voice_clone_work" / name
            if p.exists():
                out.append(p)
                break
    return out


def env_api_key():
    return os.environ.get("DASHSCOPE_API_KEY", "").strip()


def model_supports(model_id, feature):
    for m in SUPPORTED_MODELS:
        if m["id"] == model_id:
            return bool(m.get(feature))
    return True


def local_voices():
    """从本地 voice_id*.txt 读取复刻音色。"""
    out = []
    seen = set()
    for f in voice_id_files():
        vid = f.read_text(encoding="utf-8").strip()
        if vid and vid not in seen:
            seen.add(vid)
            out.append({
                "id": vid,
                "kind": "local_clone",
                "name": f"缘圆复刻 · {f.stem}",
                "lang": "中文",
                "source": str(f),
            })
    return out


def query_api_voices(api_key):
    """尽力从 DashScope 查询已注册的复刻音色（失败时返回空列表，不影响使用）。"""
    api_key = (api_key or "").strip() or env_api_key()
    if not api_key:
        return []
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    attempts = [
        {"model": "qwen-voice-enrollment", "input": {"action": "list", "page_index": 0, "page_size": 50}},
        {"model": "voice-enrollment", "input": {"action": "list_voices"}},
        {"model": "voice-enrollment", "input": {"action": "list", "page_size": 50}},
    ]
    for payload in attempts:
        try:
            r = requests.post(_custom_url(), json=payload, headers=headers, timeout=30)
            if r.status_code != 200:
                continue
            j = r.json()
            output = j.get("output") or {}
            items = (output.get("voices") or output.get("voice_list")
                     or output.get("list") or output.get("items") or [])
            if not isinstance(items, list) or not items:
                continue
            out = []
            for it in items:
                vid = it.get("voice_id") or it.get("voiceID") or it.get("voice")
                if not vid:
                    continue
                out.append({
                    "id": vid,
                    "kind": "clone",
                    "name": f"复刻音色 · {it.get('voice_name') or it.get('name') or vid[:12]}",
                    "lang": it.get("language") or "未知",
                    "source": "DashScope 查询",
                })
            if out:
                return out
        except Exception:
            continue
    return []


def _clean_key(k):
    """清理用户输入的 Key：去掉首尾空白和误带的引号。"""
    return (k or "").strip().strip('"').strip("'").strip()


def test_api_key(api_key):
    """用免费接口探测 API Key 是否有效。优先用 compatible-mode 模型列表。返回 (ok, 提示)。"""
    key = _clean_key(api_key) or env_api_key()
    if not key:
        return False, "未配置 DashScope API Key（请在 .env 中配置 DASHSCOPE_API_KEY）"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        r = requests.get(compatible_base() + "/models",
                         headers=headers, timeout=30)
    except requests.RequestException as e:
        return False, f"网络错误: {e}"
    if r.status_code == 200:
        return True, "连接正常，API Key 有效"
    if r.status_code == 401:
        return False, "API Key 无效：请检查是否复制完整、是否已过期、是否在百炼控制台创建"
    if r.status_code == 403:
        return False, "API Key 无权限：当前账号可能未开通该服务"
    return False, f"HTTP {r.status_code}"


def aliyun_probe(api_key=""):
    """试合成 1 个字，探测阿里云余额/配额是否充足（会消耗极小额度）。"""
    api_key = _clean_key(api_key) or env_api_key()
    if not api_key:
        raise RuntimeError("缺少 DashScope API Key（请在 .env 中配置 DASHSCOPE_API_KEY）")
    try:
        data, _rid = synthesize(api_key=api_key, model="qwen-audio-3.0-tts-plus",
                                voice="longxia_v3.6", text="测", fmt="mp3",
                                sample_rate=24000, max_retries=1, timeout=60)
    except RuntimeError as e:
        m = str(e)
        low = any(k in m for k in ("余额", "欠费", "quota", "Quota", "OutOfQuota",
                                   "insufficient", "Arrearage", "arrearage",
                                   "no balance", "overdraft"))
        return {"ok": False, "insufficient": low, "message": m}
    return {"ok": True, "insufficient": False,
            "message": f"试合成成功（{len(data)} 字节），阿里云账户可用、余额/配额充足"}


def tags_in_text(text):
    """提取文本中出现的标签名（去重、按出现顺序）。"""
    found = []
    for m in TAG_RE.finditer(text or ""):
        name = m.group(1)
        if name in ALL_TAG_NAMES and name not in found:
            found.append(name)
    return found


def unsupported_tags(model_id, text):
    """若模型不支持标签而文本里又出现了标签，返回这些标签名。"""
    if model_supports(model_id, "tags"):
        return []
    return tags_in_text(text)


def _safe_slice(s, n=800):
    s = str(s)
    return s[:n] + ("…" if len(s) > n else "")


def synthesize(api_key, model, voice, text, instruction="", fmt="wav",
               sample_rate=48000, max_retries=4, timeout=180, pronunciations=None):
    """合成语音，返回 (音频bytes, request_id)。

    自动尝试两种接口，兼容不同账号配置：
    1) 多模态生成接口（新版 qwen3-tts，业务空间账号用，指令字段 instructions）
    2) 原生 TTS 接口（旧版 qwen-audio-3.0-tts 等，普通百炼账号用，指令字段 instruction）
    """
    api_key = _clean_key(api_key) or env_api_key()
    if not api_key:
        raise RuntimeError("缺少 API Key：请检查 .env 中的 DASHSCOPE_API_KEY")
    if not (text or "").strip():
        raise RuntimeError("文本不能为空")
    text = normalize_tags(text)   # 中文标签 [悲伤] -> 英文标签 [sad]
    text = apply_pronunciations(text, pronunciations or [])  # 多音字注音纠正
    voice = voice or "Cherry"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base_input = {"text": text, "voice": voice, "format": fmt,
                  "sample_rate": int(sample_rate)}
    variants = []
    if model_supports(model, "instruction") and instruction:
        mm = {"model": model, "input": dict(base_input)}
        mm["input"]["instructions"] = instruction  # 新版字段
        variants.append(("多模态接口", _mm_url(), mm))
        nat = {"model": model, "input": dict(base_input)}
        nat["input"]["instruction"] = instruction  # 旧版字段
        variants.append(("原生TTS接口", _synth_url(), nat))
    else:
        variants.append(("多模态接口", _mm_url(), {"model": model, "input": dict(base_input)}))
        variants.append(("原生TTS接口", _synth_url(), {"model": model, "input": dict(base_input)}))

    errors = []
    for name, url, payload in variants:
        for attempt in range(max_retries):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            except requests.RequestException as e:
                errors.append(f"{name} 网络错误: {e}")
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                errors.append(f"{name} 服务端繁忙(HTTP {r.status_code})")
                time.sleep(6 * (attempt + 1))
                continue
            if r.status_code != 200:
                errors.append(f"{name}: {_safe_slice(r.text, 160)}")
                break  # 参数类错误，换下一个接口
            try:
                j = r.json()
            except ValueError:
                errors.append(f"{name} 响应不是 JSON")
                break
            output = j.get("output") or {}
            request_id = j.get("request_id") or output.get("request_id") or ""
            audio = output.get("audio") or {}
            url_ = audio.get("url") or ""
            data = audio.get("data") or ""
            if url_:
                dr = requests.get(url_, timeout=timeout)
                if dr.status_code == 200:
                    return dr.content, request_id
                errors.append(f"{name} 下载音频失败 HTTP {dr.status_code}")
            elif data:
                return base64.b64decode(data), request_id
            else:
                errors.append(f"{name} 响应中没有音频")
            break
    raise RuntimeError("合成失败：" + "；".join(errors[-4:]) or "合成失败")
