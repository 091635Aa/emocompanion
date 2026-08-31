# -*- coding: utf-8 -*-
"""DashScope Qwen-Audio TTS 客户端：支持的模型、标签表、音色列表、语音合成。"""
import base64
import os
import re
import time

import requests

from 环境配置 import 数据目录, 资源目录

# ---------------- 接入地址（业务空间专用域名或普通百炼域名） ----------------
def 接入地址():
    """DashScope 原生 API 根地址，可在 密钥配置.env 的 DASHSCOPE_BASE_URL 覆盖。"""
    return os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/api/v1").rstrip("/")


def 是否业务空间():
    """是否使用百炼业务空间专用域名（maas.aliyuncs.com，配合 sk-ws- 密钥）。

    业务空间账号通常只开通 qwen3-tts 系列模型（走多模态生成接口），
    qwen-audio-3.0-tts / cosyvoice 等旧系列可能未开通（返回 411/418）。
    """
    return ".maas.aliyuncs.com" in 接入地址()


def 推荐模型():
    """按账号类型返回默认合成模型：业务空间用 qwen3-tts-flash，普通账号用 qwen-audio-3.0-tts-plus。"""
    return "qwen3-tts-flash" if 是否业务空间() else "qwen-audio-3.0-tts-plus"


def 兼容地址():
    """OpenAI 兼容地址（用于探测 Key 有效性）。"""
    基址 = 接入地址()
    if 基址.endswith("/api/v1"):
        return 基址[: -len("/api/v1")] + "/compatible-mode/v1"
    return 基址.rstrip("/") + "/compatible-mode/v1"


def 合成接口地址():
    return 接入地址() + "/services/audio/tts/SpeechSynthesizer"


def 多模态接口地址():
    return 接入地址() + "/services/aigc/multimodal-generation/generation"


def 定制接口地址():
    return 接入地址() + "/services/audio/tts/customization"


# ---------------- 支持的模型（可查询、可重命名别名） ----------------
支持的模型列表 = [
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
模型ID列表 = [模型["id"] for 模型 in 支持的模型列表]

# ---------------- 完整标签表（官方文档 30 个） ----------------
控制标签列表 = [  # 控制类（23）：设定情绪/风格，作用于其后文本，直到下一个控制标签
    ("sad", "悲伤"), ("bored", "无聊"), ("amazed", "惊叹"), ("tired", "疲惫"),
    ("deep and loud shouting", "深沉大声呐喊"), ("scornful", "轻蔑"),
    ("trembling", "颤抖"), ("shouting", "大喊"), ("angry", "愤怒"),
    ("asmr", "ASMR 轻柔耳语"), ("excited", "兴奋"), ("panicked", "恐慌"),
    ("sarcastic", "讽刺"), ("mischievously", "调皮"), ("curious", "好奇"),
    ("empathetic", "共情"), ("like dracula", "德古拉风格（低沉阴森）"),
    ("whispers", "耳语"), ("serious", "严肃"), ("reluctantly", "不情愿"),
    ("very slowly", "非常缓慢"), ("crying", "哭泣"), ("very fast", "非常快速"),
]
富标签列表 = [  # 富语言类（7）：在当前位置插入拟声效果，不影响前后情感
    ("gasp", "倒吸一口气"), ("cough", "咳嗽"), ("sighing", "叹息"),
    ("giggles", "咯咯笑"), ("clears throat", "清嗓"), ("laughing", "大笑"),
    ("snorts", "哼声/嗤笑"),
]
全部标签名 = [标签[0] for 标签 in 控制标签列表] + [标签[0] for 标签 in 富标签列表]
标签正则 = re.compile(r"\[([^\[\]]+)\]")

# 中文标签含义 -> 英文标签（用于把用户手写的 [悲伤] 转成 [sad]）
中文标签映射 = {中文: 英文 for 英文, 中文 in 控制标签列表}
中文标签映射.update({中文: 英文 for 英文, 中文 in 富标签列表})


def 标准化标签(文本):
    """把用户手写的中文标签（如 [悲伤]）转成 API 识别的英文标签（[sad]）。"""
    输出 = 文本 or ""
    for 中文, 英文 in 中文标签映射.items():
        输出 = 输出.replace("[" + 中文 + "]", "[" + 英文 + "]")
    return 输出


def 应用发音纠正(文本, 纠正表):
    """给指定词自动加上注音标签，纠正多音字读音。

    纠正表: [{"词": "缘圆", "拼音": "yuan3 yuan4"}, ...]
    输出如：<phoneme alphabet="py" ph="yuan3 yuan4">缘圆</phoneme>
    """
    输出 = 文本 or ""
    if not 纠正表:
        return 输出
    # 长的词优先替换，避免子串抢先被替换
    for 条目 in sorted(纠正表, key=lambda x: len(x.get("词", "")), reverse=True):
        词 = (条目.get("词") or "").strip()
        拼音 = (条目.get("拼音") or "").strip()
        if not 词 or not 拼音:
            continue
        输出 = 输出.replace(词, f'<phoneme alphabet="py" ph="{拼音}">{词}</phoneme>')
    return 输出


# ---------------- 声音维度表（用于「声音风格助手」一键生成指令） ----------------
声音维度表 = {
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

场景预设表 = [  # 场景预设：一键填充各维度
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
系统音色列表 = [
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
音色ID文件名列表 = ["voice_id.txt", "voice_id_v1.txt", "voice_id_v2.txt"]


def 音色ID文件列表():
    """复刻音色 ID 文件：优先数据目录（用户可改），其次资源目录。

    兼容新结构（定制音色/声音库/音色ID.txt 等）与旧结构（voice_clone_work/voice_id*.txt）。
    """
    输出 = []
    已见 = set()
    搜索根们 = []
    for 根 in (数据目录(), 资源目录()):
        for 子目录 in ("定制音色/声音库", "voice_clone_work"):
            搜索根们.append(根 / 子目录)
    for 目录 in 搜索根们:
        if not 目录.is_dir():
            continue
        for 文件名 in 音色ID文件名列表:
            文件 = 目录 / 文件名
            if 文件.is_file() and str(文件) not in 已见:
                已见.add(str(文件))
                输出.append(文件)
        # 兼容任意命名的 voice_id*.txt / 音色ID*.txt
        for 模式 in ("voice_id*.txt", "音色ID*.txt"):
            for 文件 in sorted(目录.glob(模式)):
                if 文件.is_file() and str(文件) not in 已见:
                    已见.add(str(文件))
                    输出.append(文件)
    return 输出


def 环境密钥():
    return os.environ.get("DASHSCOPE_API_KEY", "").strip()


def 模型支持(模型ID, 特性):
    """查询模型是否支持某特性（如 tags / instruction）。"""
    for 模型 in 支持的模型列表:
        if 模型["id"] == 模型ID:
            return bool(模型.get(特性))
    return True


def 本地复刻音色():
    """从本地 voice_id*.txt / 音色ID*.txt 读取复刻音色。

    id 以 "qwen-omni-vc" 开头的为 Omni 复刻音色（绑定 qwen3.5-omni-plus-realtime 等，
    可用于实时通话/对话，标记 通话可用=True）；其余为 TTS 复刻（仅合成可用）。
    """
    输出 = []
    已见 = set()
    for 文件 in 音色ID文件列表():
        音色ID = 文件.read_text(encoding="utf-8").strip()
        if 音色ID and 音色ID not in 已见:
            已见.add(音色ID)
            通话可用 = 音色ID.startswith("qwen-omni-vc")
            输出.append({
                "id": 音色ID,
                "kind": "local_clone",
                "name": f"缘圆复刻 · {文件.stem}",
                "lang": "中文",
                "source": str(文件),
                "通话可用": 通话可用,
            })
    return 输出


def 查询在线音色(密钥):
    """尽力从 DashScope 查询已注册的复刻音色（失败时返回空列表，不影响使用）。"""
    密钥 = (密钥 or "").strip() or 环境密钥()
    if not 密钥:
        return []
    请求头 = {"Authorization": f"Bearer {密钥}", "Content-Type": "application/json"}
    尝试们 = [
        {"model": "qwen-voice-enrollment", "input": {"action": "list", "page_index": 0, "page_size": 50}},
        {"model": "voice-enrollment", "input": {"action": "list_voices"}},
        {"model": "voice-enrollment", "input": {"action": "list", "page_size": 50}},
    ]
    for 载荷 in 尝试们:
        try:
            响应 = requests.post(定制接口地址(), json=载荷, headers=请求头, timeout=30)
            if 响应.status_code != 200:
                continue
            结果 = 响应.json()
            输出块 = 结果.get("output") or {}
            条目们 = (输出块.get("voices") or 输出块.get("voice_list")
                      or 输出块.get("list") or 输出块.get("items") or [])
            if not isinstance(条目们, list) or not 条目们:
                continue
            在线音色们 = []
            for 条目 in 条目们:
                音色ID = 条目.get("voice_id") or 条目.get("voiceID") or 条目.get("voice")
                if not 音色ID:
                    continue
                在线音色们.append({
                    "id": 音色ID,
                    "kind": "clone",
                    "name": f"复刻音色 · {条目.get('voice_name') or 条目.get('name') or 音色ID[:12]}",
                    "lang": 条目.get("language") or "未知",
                    "source": "DashScope 查询",
                    "target_model": 条目.get("target_model") or "",
                })
            if 在线音色们:
                return 在线音色们
        except Exception:
            continue
    return []


def _清理密钥(密钥):
    """清理用户输入的 Key：去掉首尾空白和误带的引号。"""
    return (密钥 or "").strip().strip('"').strip("'").strip()


def 测试密钥(密钥):
    """用免费接口探测 API Key 是否有效。优先用 compatible-mode 模型列表。返回 (ok, 提示)。"""
    密钥 = _清理密钥(密钥) or 环境密钥()
    if not 密钥:
        return False, "未配置 DashScope API Key（请在 密钥配置.env 中配置 DASHSCOPE_API_KEY）"
    请求头 = {"Authorization": f"Bearer {密钥}"}
    try:
        响应 = requests.get(兼容地址() + "/models",
                         headers=请求头, timeout=30)
    except requests.RequestException as 异常:
        return False, f"网络错误: {异常}"
    if 响应.status_code == 200:
        return True, "连接正常，API Key 有效"
    if 响应.status_code == 401:
        return False, "API Key 无效：请检查是否复制完整、是否已过期、是否在百炼控制台创建"
    if 响应.status_code == 403:
        return False, "API Key 无权限：当前账号可能未开通该服务"
    return False, f"HTTP {响应.status_code}"


def 探测余额(密钥=""):
    """试合成 1 个字，探测阿里云余额/配额是否充足（会消耗极小额度）。"""
    密钥 = _清理密钥(密钥) or 环境密钥()
    if not 密钥:
        raise RuntimeError("缺少 DashScope API Key（请在 密钥配置.env 中配置 DASHSCOPE_API_KEY）")
    try:
        数据, _请求ID = 合成(密钥=密钥, 模型="qwen-audio-3.0-tts-plus",
                             音色="longxia_v3.6", 文本="测", 格式="mp3",
                             采样率=24000, 最大重试=1, 超时=60)
    except RuntimeError as 异常:
        消息 = str(异常)
        是否不足 = any(关键词 in 消息 for 关键词 in (
            "余额", "欠费", "quota", "Quota", "OutOfQuota",
            "insufficient", "Arrearage", "arrearage",
            "no balance", "overdraft"))
        return {"ok": False, "insufficient": 是否不足, "message": 消息}
    return {"ok": True, "insufficient": False,
            "message": f"试合成成功（{len(数据)} 字节），阿里云账户可用、余额/配额充足"}


def 文本中的标签(文本):
    """提取文本中出现的标签名（去重、按出现顺序）。"""
    找到 = []
    for 匹配 in 标签正则.finditer(文本 or ""):
        名称 = 匹配.group(1)
        if 名称 in 全部标签名 and 名称 not in 找到:
            找到.append(名称)
    return 找到


def 不支持的标签(模型ID, 文本):
    """若模型不支持标签而文本里又出现了标签，返回这些标签名。"""
    if 模型支持(模型ID, "tags"):
        return []
    return 文本中的标签(文本)


def _截断(文本, 长度=800):
    文本 = str(文本)
    return 文本[:长度] + ("…" if len(文本) > 长度 else "")


def _尝试接口(名称, 地址, 载荷, 请求头, 超时, 最大重试, 错误们):
    """调用一个合成接口，成功返回 (音频bytes, request_id)，失败返回 None。"""
    for 尝试 in range(最大重试):
        try:
            响应 = requests.post(地址, json=载荷, headers=请求头, timeout=超时)
        except requests.RequestException as 异常:
            错误们.append(f"{名称} 网络错误: {异常}")
            time.sleep(5 * (尝试 + 1))
            continue
        if 响应.status_code in (429, 500, 502, 503, 504):
            错误们.append(f"{名称} 服务端繁忙(HTTP {响应.status_code})")
            time.sleep(6 * (尝试 + 1))
            continue
        if 响应.status_code != 200:
            错误们.append(f"{名称}: {_截断(响应.text, 160)}")
            return None  # 参数类错误，不重试
        try:
            结果 = 响应.json()
        except ValueError:
            错误们.append(f"{名称} 响应不是 JSON")
            return None
        输出块 = 结果.get("output") or {}
        请求ID = 结果.get("request_id") or 输出块.get("request_id") or ""
        音频 = 输出块.get("audio") or {}
        音频地址 = 音频.get("url") or ""
        音频数据 = 音频.get("data") or ""
        if 音频地址:
            try:
                下载响应 = requests.get(音频地址, timeout=超时)
            except requests.RequestException as 异常:
                错误们.append(f"{名称} 下载音频失败: {异常}")
                return None
            if 下载响应.status_code == 200:
                return 下载响应.content, 请求ID
            错误们.append(f"{名称} 下载音频失败 HTTP {下载响应.status_code}")
            return None
        if 音频数据:
            return base64.b64decode(音频数据), 请求ID
        错误们.append(f"{名称} 响应中没有音频")
        return None
    return None


def 合成(密钥="", *, 模型, 音色, 文本, 指令="", 格式="wav",
         采样率=48000, 最大重试=4, 超时=180, 发音纠正=None):
    """合成语音，返回 (音频bytes, request_id)。

    按模型选择正确接口（避免"接口与模型不匹配"导致 url error / 411）：
    - qwen3-tts 系列 → 多模态生成接口（业务空间账号用，指令字段 instructions）
    - qwen-audio-3.0-tts / cosyvoice → 原生 TTS 接口（普通百炼账号用，指令字段 instruction）
    业务空间账号若当前模型未开通（411/418），自动兜底用 qwen3-tts-flash + Cherry 重试。
    """
    密钥 = _清理密钥(密钥) or 环境密钥()
    if not 密钥:
        raise RuntimeError("缺少 API Key：请检查 密钥配置.env 中的 DASHSCOPE_API_KEY")
    if not (文本 or "").strip():
        raise RuntimeError("文本不能为空")
    文本 = 标准化标签(文本)      # 中文标签 [悲伤] -> 英文标签 [sad]
    文本 = 应用发音纠正(文本, 发音纠正 or [])  # 多音字注音纠正
    音色 = 音色 or "Cherry"

    请求头 = {"Authorization": f"Bearer {密钥}", "Content-Type": "application/json"}
    基础输入 = {"text": 文本, "voice": 音色, "format": 格式,
                "sample_rate": int(采样率)}

    # 按模型分流接口
    变体们 = []
    if 模型.startswith("qwen3-tts"):
        if 模型支持(模型, "instruction") and 指令:
            多模态 = {"model": 模型, "input": dict(基础输入)}
            多模态["input"]["instructions"] = 指令  # 新版指令字段
            变体们.append(("多模态接口", 多模态接口地址(), 多模态))
        else:
            变体们.append(("多模态接口", 多模态接口地址(), {"model": 模型, "input": dict(基础输入)}))
    else:
        if 模型支持(模型, "instruction") and 指令:
            原生 = {"model": 模型, "input": dict(基础输入)}
            原生["input"]["instruction"] = 指令  # 旧版指令字段
            变体们.append(("原生TTS接口", 合成接口地址(), 原生))
        else:
            变体们.append(("原生TTS接口", 合成接口地址(), {"model": 模型, "input": dict(基础输入)}))

    错误们 = []
    for 名称, 地址, 载荷 in 变体们:
        结果 = _尝试接口(名称, 地址, 载荷, 请求头, 超时, 最大重试, 错误们)
        if 结果:
            return 结果

    # 业务空间账号兜底：多数业务空间账号仅开通 qwen3-tts 系列（多模态接口），
    # 当前模型未开通（如 qwen-audio / cosyvoice 返回 411/418）时，用
    # qwen3-tts-flash + Cherry 再试一次，保证合成可用。
    if 是否业务空间() and 模型 != "qwen3-tts-flash":
        兜底载荷 = {"model": "qwen3-tts-flash", "input": dict(基础输入)}
        兜底载荷["input"]["voice"] = "Cherry"
        if 指令:
            兜底载荷["input"]["instructions"] = 指令
        结果 = _尝试接口("业务空间兜底", 多模态接口地址(), 兜底载荷,
                     请求头, 超时, 最大重试, 错误们)
        if 结果:
            return 结果

    raise RuntimeError("合成失败：" + "；".join(错误们[-4:]) or "合成失败")
