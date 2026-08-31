# -*- coding: utf-8 -*-
"""缘圆 · 一体化对话台（文本生成 + 本地 TTS + 会话历史 + 挂载切换 + 角色设定）

集成架构:
  - 文本生成: 代理 04 文本引擎（llama.cpp + Qwen3-4B，默认 http://127.0.0.1:8000/chat）
  - TTS: 单一路径 llama-tts.exe（GGUF INT4，CUDA 加速，RTF≈0.4）外挂方案
      情感靠 voice/emotion LoRA adapter + 参考音频(serve/refs/<情感>.wav)音色锚点 + 采样预设驱动
  - 会话历史: serve/out/sessions/*.json 持久化，支持历史回顾/问答回放
  - 挂载切换: adapter / 情感模式 / 角色 全局热切换
  - 角色设定: 基于打标数据(微调训练集)聚合的 persona/traits/情感关键词，可编辑
  - 详细日志: serve/out/logs/integrated_app.log（加载/合成/语速逐条明细）

启动:
  python integrated_app.py [--host 127.0.0.1] [--port 8071] [--text-base http://127.0.0.1:8000]
"""
import argparse
import io
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tts_engine import get_engine, list_adapter_names, TTSUnavailable
from emo_detect import detect_emotion, _extract_text
from tts_gguf import GGUFTTS, STYLE_PRESETS as _STYLE_PRESETS, DEFAULT_STYLE as _DEFAULT_STYLE

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "webapp"
OUT_DIR = HERE.parent / "out"
SESS_DIR = OUT_DIR / "sessions"
AUDIO_DIR = OUT_DIR / "chat_audio"
SESS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="缘圆 · 一体化对话台", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---------------- 运行参数（main() 注入） ----------------
_TEXT_BASE = "http://127.0.0.1:8000"
_GGUF = None

# ---------------- 详细日志（stdout + file） ----------------
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "integrated_app.log"

_LOGGER = logging.getLogger("yy-integrated")
_LOGGER.setLevel(logging.DEBUG)
if not _LOGGER.handlers:
    _fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    _h_console = logging.StreamHandler()
    _h_console.setLevel(logging.INFO)
    _h_console.setFormatter(_fmt)
    _h_file = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
    _h_file.setLevel(logging.DEBUG)
    _h_file.setFormatter(_fmt)
    _LOGGER.addHandler(_h_console)
    _LOGGER.addHandler(_h_file)


def log(level: str, msg: str, **kw):
    """带额外字段的明细日志。level∈{debug,info,warn,error}；字段会拼到行尾。"""
    extra = "  ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
    getattr(_LOGGER, level)(f"{msg}" + (f"  {extra}" if extra else ""))


def _t(s):
    """把耗时秒数格式化成 ms / s，便于阅读。"""
    try:
        s = float(s)
    except Exception:
        return "?"
    return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


def _get_gguf():
    global _GGUF
    if _GGUF is None:
        _GGUF = GGUFTTS.get()
    return _GGUF


# ======================================================================
# 角色设定（源自打标数据聚合 + role_pack persona）
# ======================================================================
ROLE_PACK_PERSONA = (
    "你是'缘圆'，一个温柔、爱撒娇、爱黏人、口语化的直播情感主播。"
    "说话习惯：常带'呀''嘛''啦''呗'；口头禅'我去''哎呀''家人们''有感觉吗'；"
    "热情招呼点关注、加粉丝灯牌；感谢用'谢谢、爱你、想你了'；自称'新人主播'。"
    "像真人主播一样自然回应：日常寒暄说几句简短的话就行；观众难过、需要安慰或想深聊时，"
    "要多写几句、温柔地展开（可以写一小段），但始终口语化，绝不书面化/翻译腔/AI腔/透露是AI。\n"
    "【示例】观众'今天紧张吗'→缘圆'哎呀~有点紧张呢，我才开播第五天，但看到你来我就开心啦！'\n"
    "【示例】观众'为什么要加灯牌呀'→缘圆'点个关注加个灯牌，就能一直找到我啦，好嘛？'\n"
    "【语音标记·必须】为让语音有情绪起伏，回复正文每一句都必须以 [情感] 开头、句末按情绪加 (语速词)，"
    "格式：[情感]正文(语速词)。情感从 开心/俏皮/悲伤/平静/兴奋/撒娇/温柔/激动 中选；"
    "语速词从 最快/飞快/快速/中速/舒缓/慢速 中选。相邻句尽量换不同的情感，让整段语音有起伏。"
    "示例：[俏皮]嘿嘿~你终于来啦(快速)[温柔]今天辛苦啦，我陪你(舒缓)。"
    "⚠方括号[ ]和圆括号( )是语音指令，绝不允许出现在纯文本里、也不许念出来；"
    "不要用引号\"\"、星号**、或其它符号代替；指令后必须紧跟正文、不得多加空格。\n"
    "【动作提示】想表现神态/动作时（如微笑、眨眼、挥手），单独用〔动作：……〕写在回复末尾另起一行，"
    "如〔动作：温柔地笑了笑〕；该动作属表演指令，绝不混入正文，也不要用引号、星号或其它括号包动作。"
)

# 角色动作提示剥离：〔动作：…〕/（…）/(…)【…】 均不向用户展示、也不喂给 TTS
_ACT_RE = re.compile(r"〔(?P<a>[^〔〕]*)〕|【(?P<d>[^【】]*)】|（(?P<b>[^（）]*)）|\((?P<c>[^()]*)\)")


def _strip_actions(text: str):
    """剥离括号动作提示。返回 (干净文本, 动作列表[供未来 VTS 等绑定])。"""
    actions = []

    def _repl(m):
        seg = (m.group("a") or m.group("d") or m.group("b") or m.group("c") or "").strip()
        if seg:
            actions.append(seg)
        return ""

    clean = _ACT_RE.sub(_repl, text or "")
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, actions


# 逐句切分（按中文句号类结尾切，保留句末标点；逗号不切，避免碎句）
_SENT_SPLIT_RE = re.compile(r"([^。！？!?…～~]+[。！？!?…～~]*)")


def _split_sentences(text: str, max_n: int = 6):
    """把回复切成句子列表。超过 max_n 句则把剩余合并进最后一句，避免 TTS 调用过多。"""
    segs = [s.strip() for s in _SENT_SPLIT_RE.findall(text or "") if s and s.strip()]
    if not segs:
        segs = [text.strip()] if (text or "").strip() else []
    if len(segs) > max_n:
        segs = segs[:max_n - 1] + ["".join(segs[max_n - 1:])]
    return segs


# 子句切分（逗号/顿号/~ 等，制造句内情感/语速波浪）
_SUB_RE = re.compile(r"([^，,、~…]+[，,、~…]?|[^，,、~…]+$)")

# 情感/语速波浪循环（相邻错开，制造非直线曲线）
_EMO_CYCLE = ["俏皮", "温柔", "兴奋", "悲伤", "平静", "开心", "撒娇", "激动"]
_RATE_CYCLE = [1.10, 0.95, 1.05, 0.92, 1.12, 0.98, 1.15, 1.0]


def _expand_emotion_curve(segments, base_emotion: str = "平静", max_clauses: int = 6):
    """把 [(文本, 情感, 语速)] 展开成更细的子句序列，增强句内情感/语速波浪。

    即使引擎只回 1 句，也按逗号/顿号切成子句；子句情感在主情感与相邻情感间
    错开、语速快慢交替，形成波浪曲线而非直线。未标注情感的句子回退到整句情感。
    """
    out = []
    for seg_text, style, rate in segments:
        t = (seg_text or "").strip()
        if not t:
            continue
        # 足够长才子句化
        if len(t) > 5 and t.count("，") + t.count(",") + t.count("、") + t.count("~") > 0:
            parts = [p.strip() for p in _SUB_RE.findall(t) if p and p.strip()]
            if len(parts) >= 2:
                base_emo = style or base_emotion
                idx = _EMO_CYCLE.index(base_emo) if base_emo in _EMO_CYCLE else 0
                for i, p in enumerate(parts):
                    if i >= max_clauses:
                        break
                    emo = _EMO_CYCLE[(idx + i) % len(_EMO_CYCLE)]
                    r = _RATE_CYCLE[(idx + i) % len(_RATE_CYCLE)]
                    # 保留每段主情感倾向：首子句用主情感，其后错开
                    if i == 0:
                        emo = base_emo
                    out.append((p, emo, r))
                continue
        # 无逗号的短句：情感用标注或整句情感；语速 1.0 时交给合成层按情感预设原生变速
        out.append((t, style or base_emotion,
                    float(rate) if (rate is not None and rate > 0) else 1.0))
    return out


def _merge_clauses(expanded, max_calls: int = 4):
    """合并过细的子句，把合成调用数压到 max_calls 以内。

    每个 llama-tts 调用都要重载模型(约5s)，过多调用会让整段合成极慢；
    合并后情感取首子句、语速取首子句，保证调用数受限但曲线仍在。
    """
    if len(expanded) <= max_calls:
        return expanded
    n = len(expanded)
    per = max(1, int(round(n / max_calls)))
    out = []
    for i in range(0, n, per):
        chunk = expanded[i:i + per]
        txt = "".join(c[0] for c in chunk)
        out.append((txt, chunk[0][1] or "平静",
                    float(chunk[0][2]) if chunk[0][2] and chunk[0][2] > 0 else 1.0))
    return out[:max_calls]


# ======================================================================
# 自然语言风格标记（AI 逐句控 TTS 风格/语速）
#   格式: [风格]文本内容(语速)   —— 括号内容用户不可见，仅喂给 TTS
#   例  : [俏皮]嘿嘿你终于来啦(快速) [温柔]今天辛苦啦(慢速)
#   无标记的句子风格缺省(用整句情感)，语速默认 1.0。
# ======================================================================
_KNOWN_STYLES = ["开心", "俏皮", "悲伤", "平静", "兴奋", "撒娇", "温柔", "激动"]
_RATE_WORDS = {
    "最快": 1.30, "飞快": 1.25, "急促": 1.22, "快速": 1.18, "快": 1.12,
    "中速": 1.00, "正常": 1.00, "适中": 1.00, "平稳": 1.00,
    "舒缓": 0.92, "慢": 0.90, "慢速": 0.88, "缓慢": 0.85, "温柔": 0.92,
    "轻声": 0.95, "低语": 0.90,
}


def _norm_style(style: str):
    """把任意风格词归一化到情感词表；不认识返回 None(走自动/整句情感)。"""
    s = (style or "").strip()
    if not s:
        return None
    for k in _KNOWN_STYLES:
        if k in s:
            return k
    return None


def _parse_rate(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 1.0
    if s in _RATE_WORDS:
        return _RATE_WORDS[s]
    # 支持 "x1.2" / "1.2x" / "1.2" 纯数字
    m = re.fullmatch(r"[xX]?(\d+(?:\.\d+)?)[xX倍]?", s)
    if m:
        try:
            v = float(m.group(1))
            return min(max(v, 0.6), 1.6)
        except Exception:
            return 1.0
    return 1.0


def _strip_markup(raw: str) -> str:
    """去掉 [风格] 与一切括号内容（(语速)/（说明）〔动作〕【】），仅留用户可见文本。"""
    s = re.sub(r"\[[^\[\]]*\]", "", raw or "")
    s = _strip_actions(s)[0]
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s


def _parse_style_markup(text: str):
    """解析自然语言风格标记，返回 (segments, display_text)。

    segments: [(可见文本, 风格或None, 语速), ...]；无标记时按句切分、风格 None。
    display_text: 剥掉所有 [风格]/(语速) 后的用户可见文本。
    """
    raw = (text or "").strip()
    segments: list = []
    pos = 0
    n = len(raw)
    while True:
        m = re.search(r"\[([^\[\]]+)\]", raw[pos:])
        if m is None:
            rest = raw[pos:].strip()
            if rest:
                for s in _split_sentences(rest, max_n=8):
                    s = _strip_actions(s)[0]
                    if s.strip():
                        segments.append((s.strip(), None, 1.0))
            break
        head = raw[pos:pos + m.start()].strip()
        if head:  # 方括号前的纯文本 → 逐句、风格缺省
            for s in _split_sentences(head, max_n=8):
                s = _strip_actions(s)[0]
                if s.strip():
                    segments.append((s.strip(), None, 1.0))
        style = m.group(1).strip()
        # 方括号之后到下一个 [ 或结尾 为标记段正文（可含 (语速) 后缀）
        body_start = pos + m.end()
        nxt = raw.find("[", body_start)
        body_end = n if nxt < 0 else nxt
        body = raw[body_start:body_end].strip()
        rate = 1.0
        rm = re.search(r"[（(]([^（）()]*)[）)]\s*$", body)
        if rm:
            rate = _parse_rate(rm.group(1))
            body = body[:rm.start()].strip()
        body = _strip_actions(body)[0]  # 剥离剩余动作/说明括号，勿朗读
        if body:
            segments.append((body, _norm_style(style), rate))
        pos = body_end if nxt < 0 else nxt
    if not segments:
        segments = [(raw or " ", None, 1.0)]
    return segments, _strip_markup(raw)

# 打标数据聚合结果（6951 条样本）
_LABELED_TRAITS = {
    "warmth": 0.83, "playfulness": 0.75, "sassiness": 0.52,
    "energy_baseline": 0.76, "formality": 0.31,
}

# 情感 → 高频关键词（打标数据 Top + 领域词典合并，供自动情感识别与 TTS 参考音频选择）
_LABELED_EMO_KEYWORDS = {
    "开心": "欢迎 谢谢 关注 灯牌 喜欢 太棒 好耶 开心 高兴 哈哈 家人们 欢迎来到",
    "俏皮": "嘻嘻 嘿嘿 调皮 卖萌 么么 亲亲 嘤嘤 傲娇 人家 啾咪 mua",
    "悲伤": "难过 伤心 想哭 呜呜 委屈 心痛 失落 蓝瘦 唉 伤心死",
    "平静": "今天 然后 感觉 知道 可以 嗯 平平淡淡 深呼吸",
    "兴奋": "太激动 好兴奋 冲啊 起飞 燃 热血 哇塞 惊艳 炸裂 疯了",
    "撒娇": "撒娇 抱抱 要抱抱 别这样 讨厌啦 不要 哼 不嘛 小哥哥 亲亲 愿意 喜欢你",
    "温柔": "温柔 想你 喜欢你 爱你 愿意 牵手 呵护 慢慢 甜甜的",
    "激动": "太激动 哭了 第一次 终于 感动 破防",
}

ROLES = {
    "缘圆": {
        "name": "缘圆",
        "desc": "温柔爱撒娇的口语化直播情感主播",
        "persona": ROLE_PACK_PERSONA,
        "traits": dict(_LABELED_TRAITS),
        "catchphrases": ["呀", "嘛", "啦", "呗", "我去", "哎呀", "家人们", "有感觉吗"],
        "emotion_keywords": {k: v for k, v in _LABELED_EMO_KEYWORDS.items()},
    }
}
_active_role = "缘圆"

# 全局挂载状态（前端热切换）
_mount = {
    "tts_backend": "gguf",    # 单一外挂方案（llama-tts.exe GGUF INT4，CUDA 加速 RTF≈0.4）
    "adapter": "emotion",     # voice(音色) | emotion(情感律动)
    "emotion_mode": "auto",   # auto(自动识别) | manual(手动指定)
    "emotion": "平静",
    "style": "自然",          # 说话风格（StylePlug）：自然/甜美/元气/活力/娇俏/温柔/慵懒/坚定
    "role": _active_role,
    "want_tts": True,
    "max_new": 160, "temperature": 0.9, "top_p": 0.9, "top_k": 50,
    "tone_variation": 0.35,   # 随机语气强度 0~1
}

# 主播化行为层：情感向量路由 → 长度自适应
# 按"用户消息"识别出的路由情感动态给 max_new：安慰/深聊类给足长度，日常寒暄类短促
_MAX_NEW_BY_EMO = {
    "悲伤": 320, "温柔": 320, "激动": 300, "平静": 260,
    "开心": 200, "俏皮": 200, "兴奋": 200, "撒娇": 220,
}
_MAX_NEW_DEFAULT = 200

# 句句有回应兜底：文本引擎空回复时附在 persona 后的后缀（只重试一次）
_FALLBACK_SUFFIX = "\n无论如何都要回一句实在的，别空着。"

# ======================================================================
# 会话历史（SQLite 动态数据库；文本+音频路径均持久化，历史音频可回放）
# ======================================================================
DB_PATH = OUT_DIR / "yuanchat.db"


def _db_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions("
        " id TEXT PRIMARY KEY, title TEXT, created REAL, updated REAL, messages TEXT)")
    conn.commit()
    return conn


def _load_sessions():
    out = {}
    try:
        conn = _db_conn()
        rows = conn.execute("SELECT * FROM sessions").fetchall()
        conn.close()
        for r in rows:
            try:
                out[r[0]] = {"id": r[0], "title": r[1], "created": r[2],
                             "updated": r[3],
                             "messages": json.loads(r[4] or "[]")}
            except Exception:
                continue
    except Exception:
        out = {}
    # 迁移旧版 JSON 文件（serve/out/sessions/*.json）到 SQLite，保证不丢历史
    try:
        if not out and SESS_DIR.is_dir():
            for f in SESS_DIR.glob("*.json"):
                try:
                    s = json.loads(f.read_text(encoding="utf-8"))
                    if s.get("id"):
                        out[s["id"]] = s
                        _save_session(s)
                except Exception:
                    continue
    except Exception:
        pass
    return out


_sessions = _load_sessions()


def _save_session(s):
    try:
        conn = _db_conn()
        conn.execute(
            "INSERT OR REPLACE INTO sessions(id,title,created,updated,messages) "
            "VALUES(?,?,?,?,?)",
            (s["id"], s.get("title", "新对话"), float(s.get("created", 0.0)),
             float(s.get("updated", 0.0)),
             json.dumps(s.get("messages", []), ensure_ascii=False)))
        conn.commit()
        conn.close()
    except Exception:
        pass


def _new_session(title="新对话"):
    sid = "sess_" + uuid.uuid4().hex[:10]
    now = time.time()
    s = {"id": sid, "title": title or "新对话", "created": now, "updated": now,
         "messages": []}
    _sessions[sid] = s
    _save_session(s)
    return s


def _get_session(sid: str):
    s = _sessions.get(sid)
    if not s:
        raise HTTPException(404, f"会话不存在: {sid}")
    return s


# ======================================================================
# 文本引擎代理（04：llama.cpp + Qwen3-4B）
# ======================================================================
def call_text_engine(payload: dict) -> dict:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(_TEXT_BASE.rstrip("/") + "/chat", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code or 502,
                            f"文本引擎错误: {e.read().decode('utf-8', 'ignore')}")
    except Exception as e:
        raise HTTPException(502, f"无法连接文本引擎 {_TEXT_BASE}: {e}")


def text_engine_online() -> bool:
    try:
        call_text_engine({"messages": [{"role": "user", "content": "ping"}], "max_new": 1})
        return True
    except HTTPException:
        return False


# ---------------- 文本引擎自动拉起 ----------------
# 一体化启动时若 04 文本引擎未运行，则在本目录进程中拉起 server.py 子进程，
# 避免用户手动启动、对话时被端口占用(WinError 10061)拒绝。
_TEXT_PROC = None


def _text_engine_env():
    env = dict(os.environ)
    from pathlib import Path as _P
    _f = _P(__file__).resolve()
    root = _f.parents[3]                      # d:\AI情感
    engine_root = _f.parents[2]               # 缘圆_角色挂载与情感注入工程
    # llama.cpp CUDA 由 pykits 提供；torch lib 一并加入 PATH
    llamacpp_pkg = root / "pykits" / "llamacpp"
    llamacpp_lib = llamacpp_pkg / "llama_cpp" / "lib"
    torch_lib = _P(
        r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib")
    backend_dir = engine_root / "04_源码与原型" / "backend"
    env["PATH"] = str(llamacpp_lib) + os.pathsep + str(torch_lib) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(backend_dir) + os.pathsep + str(llamacpp_pkg) + os.pathsep + env.get("PYTHONPATH", "")
    return env, backend_dir


def _ensure_text_engine(log_cb=None):
    """若文本引擎离线则拉起 04 server.py（后台），等待上线。返回是否可用。"""
    global _TEXT_PROC
    if text_engine_online():
        log("debug", "文本引擎已在线，无需拉起")
        return True
    if _TEXT_PROC is not None and _TEXT_PROC.poll() is None:
        log("debug", "文本引擎进程已存在，等待上线")
    else:
        import subprocess as _sp
        try:
            env, backend_dir = _text_engine_env()
            py = backend_dir / "../.venv/Scripts/python.exe"
            py = py if py.exists() else _sp.sys.executable
            port = int(_TEXT_BASE.rsplit(":", 1)[-1])
            cmd = [str(py), str(backend_dir / "server.py"), "--host",
                   _TEXT_BASE.split("://")[1].split(":")[0], "--port", str(port)]
            log("info", "文本引擎离线，自动拉起 04 server.py", cmd=" ".join(str(c) for c in cmd))
            if log_cb:
                log_cb("文本引擎离线，正在自动启动… 首次加载模型需约 20-40s")
            _TEXT_PROC = _sp.Popen(cmd, cwd=str(backend_dir), env=env,
                                   stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                                   creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0))
        except Exception as e:
            log("error", "文本引擎自动拉起失败", err=f"{type(e).__name__}: {e}")
            return False
    # 等待上线（最多 180s）
    deadline = time.time() + 180
    while time.time() < deadline:
        if text_engine_online():
            log("info", "文本引擎自动拉起成功")
            if log_cb:
                log_cb("文本引擎已就绪")
            return True
        time.sleep(3)
    log("error", "文本引擎拉起超时")
    return False


# ======================================================================
# 本地 TTS 合成（双后端）
# ======================================================================
def _ref_for_emotion(emotion: str):
    """情感 → serve/refs/<情感>.wav（llama-tts 实测可读中文路径）。不存在返回 None。"""
    p = HERE / "refs" / f"{emotion}.wav"
    return str(p) if os.path.isfile(p) else None


def _refs_for_emotion(emotion: str, anchor: str = "平静") -> list:
    """多段情感参考片段 ICL：每情感一段，合成时动态选段。

    返回 list[dict{audio,text,emotion}]：
      - 先放入音色锚点(平静)段做角色基准（若存在且与目标不同）
      - 再放入目标情感段（ref 文本带 `[情感]` 标签，只进 ICL prompt 不会被朗读）
    供 tf 后端一次传入 generate_voice_clone 实现多段 ICL。
    """
    refs_dir = HERE / "refs"
    out = []

    def _seg(emo):
        wav = refs_dir / f"{emo}.wav"
        if not wav.is_file():
            return None
        txt = refs_dir / f"{emo}.txt"
        text = None
        if txt.is_file():
            raw = txt.read_text(encoding="utf-8").strip()
            if raw:
                text = f"[{emo}]{raw}"
        return {"audio": str(wav), "text": text, "emotion": emo}

    if anchor and anchor != emotion:
        a = _seg(anchor)
        if a:
            out.append(a)
    t = _seg(emotion)
    if t:
        out.append(t)
    return out


def synth_tts(text: str, emotion: str, backend: str | None = None,
              adapter: str = "emotion", tone_variation: float = 0.35,
              rate: float = 1.0, ref: str | None = None,
              refs: list | None = None):
    """合成单句语音（外挂 llama-tts GGUF）。语速靠内嵌指令由模型原生控制，不再程序变速。"""
    text, _ = _strip_actions(text)   # 动作提示永不进入语音
    start = time.time()
    try:
        wav, sr, meta = _get_gguf().synthesize(text, emotion, ref=ref, rate=rate)
    except Exception as e:
        log("error", "gguf 外挂合成失败", text=text[:40], emotion=emotion,
            rate=rate, err=f"{type(e).__name__}: {e}")
        raise
    meta = dict(meta or {})
    meta["backend"] = "gguf"
    meta["rate"] = float(rate)
    log("debug", "gguf 外挂合成完成", text=text[:40], emotion=emotion,
        rate=rate, dur=_t(time.time() - start), sr=sr)
    return np.asarray(wav, dtype="float32"), int(sr), meta


# 语速词 -> 模型原生指令（放回文本，由 TTS 演绎，非程序 time_stretch）
_RATE_INSTR = {
    "最快": "（语速飞快）", "飞快": "（语速飞快）", "急促": "（语速加快）",
    "快速": "（语速加快）", "快": "（语速稍快）",
    "中速": "（语速正常）", "正常": "（语速正常）", "适中": "（语速正常）", "平稳": "（语速正常）",
    "舒缓": "（语速放慢）", "慢": "（语速放慢）", "慢速": "（语速放慢）",
    "缓慢": "（语速放慢）", "温柔": "（轻声温柔）",
    "轻声": "（轻声）", "低语": "（低声细语）",
}
# 情感词 -> 模型原生 [情感] 指令（放在句首，由模型切换演绎）
_EMO_INSTR = {
    "开心": "[开心]", "俏皮": "[俏皮]", "悲伤": "[悲伤]", "平静": "[平静]",
    "兴奋": "[兴奋]", "撒娇": "[撒娇]", "温柔": "[温柔]", "激动": "[激动]",
}


def build_instruction_text(segments):
    """把逐句 [(文本, 情感orNone, 语速词)] 重组成一段带原生指令的文本。

    由文本引擎强绑定 TTS：情感用 [情感] 句首标记、语速用 (语速说明) 句尾标记，
    全部由 llama-tts 模型原生演绎，保证整段音色一致 + 情感/语速曲线连续。
    """
    out = []
    for seg_text, style, rate_word in segments:
        t = (seg_text or "").strip()
        emo = style or None
        inst = ""
        if emo and emo in _EMO_INSTR:
            inst += _EMO_INSTR[emo]
        body = _ACT_RE.sub("", t).strip()
        if rate_word and rate_word in _RATE_INSTR:
            body = body + " " + _RATE_INSTR[rate_word]
        if not body:
            continue
        out.append(inst + body)
    return " ".join(out)


def synth_paragraph_tts(segments, backend: str = "gguf",
                        emotion: str = "平静", ref: str | None = None):
    """整段一次合成（核心）：把文本引擎给出的逐句(文本/情感/语速)重组成一段带原生指令的
    文本，一次性交给 llama-tts，模型原生演绎整段音色一致 + 情感/语速曲线。

    解决：逐句独立+随机seed → 音色漂移成多音色；程序变速 → 语速生硬。
    返回 (wav, sr, meta)。
    """
    full_text = build_instruction_text(segments)
    return synth_paragraph_tts_segments(full_text, emotion=emotion, ref=ref)


def synth_paragraph_tts_segments(full_text: str, backend: str = "gguf",
                                 emotion: str = "平静", ref: str | None = None):
    """整段一次合成（文本版）：把已含 [情感]/(语速) 指令的文本直接整段喂模型。

    full_text 由文本引擎产出（PERSONA 强制每句带 [风格] + 句末 (语速)），
    llama-tts 模型原生演绎：整段音色一致 + 情感/语速曲线连续，不再逐句随机漂移。
    """
    full_text = (full_text or "").strip()
    if not full_text:
        raise ValueError("[gguf] 合成文本为空")
    start = time.time()
    g = _get_gguf()
    # 固定稳定 seed（同一段文本可复现，音色稳定），而非随机 -1
    wav, sr, meta = g.synthesize_paragraph(full_text, emotion=emotion, ref=ref,
                                           seed=g.stable_seed)
    meta = dict(meta or {})
    meta["backend"] = "gguf"
    meta["mode"] = meta.get("mode", "paragraph_label_control")  # gguf 层已标注标签控制模式
    log("info", "段落标签控制 TTS 合成", seed=meta.get("seed"),
        dur=_t(time.time() - start), audio_s=meta.get("audio_seconds"))
    return np.asarray(wav, dtype="float32"), int(sr), meta


def synth_flow_tts(segments, backend: str = "gguf",
                   emotion: str = "平静", ref: str | None = None,
                   style: str | None = None):
    """混合方案：整段一次合成为主，长回复才拆关键子句。

    segments: [(文本, 情感orNone, 语速float), ...]（来自 _parse_style_markup）
    style: 说话风格（StylePlug），作为整段另一条正交轴叠加（采样系数 + 句间停顿）。
    - 默认整段一次合成（1 次 llama-tts 调用≈5s）：正文拼接，情感用整句情感采样参数
      + 表情文本 + LoRA 驱动；语速用模型自然语速（不做程序变速 → 无电音、不拖慢）。
    - 仅当回复较长（>24字）且含逗号且存在≥2种情感标记时，才拆关键子句（最多3次调用）
      让每子句用各自情感采样参数，形成句内情感起伏。
    - 音色一致：稳定 seed + speaker=503 锚定，不漂移。
    """
    if not segments:
        raise ValueError("[gguf] 合成段落为空")
    g = _get_gguf()
    total_len = sum(len((s[0] or "").strip()) for s in segments)
    n_emo = len({(s[1] or emotion) for s in segments if s[1]})
    has_comma = any((("，" in s[0]) or ("," in s[0])) for s in segments if s[0])
    do_split = len(segments) >= 2 and total_len > 24 and has_comma and n_emo >= 2

    start = time.time()
    if do_split:
        expanded = _expand_emotion_curve(segments, base_emotion=emotion or "平静")
        expanded = _merge_clauses(expanded, max_calls=3)
        wav, sr, meta = g.synthesize_flow(expanded, ref=ref, seed=g.stable_seed,
                                          style=style)
        meta = dict(meta or {})
        meta["mode"] = "split_clauses"
        log("info", "关键子句 TTS 合成", n_calls=len(expanded), n_seg=len(segments),
            style=style, dur=_t(time.time() - start))
    else:
        # 整段一次合成：正文拼接（剥离所有标记），自然语速，无电音
        from tts_gguf import strip_control_tokens as _sct
        parts = []
        for seg_text, style, rate in segments:
            t = _sct(seg_text or "")
            if not t:
                continue
            parts.append(t)
        full = "，".join(parts)
        if not full:
            raise ValueError("[gguf] 合成正文为空")
        wav, sr, meta = g.synthesize(full, emotion=emotion or "平静", ref=ref,
                                     rate=1.0, style=style)
        meta = dict(meta or {})
        meta["mode"] = "paragraph_once"
        log("info", "整段一次 TTS 合成", text=full[:40], style=style,
            dur=_t(time.time() - start), audio_s=meta.get("audio_seconds"))
    meta["backend"] = "gguf"
    return np.asarray(wav, dtype="float32"), int(sr), meta


def _trim_silence(arr, sr, thresh=0.005, keep_lead=0.02, keep_tail=0.06):
    """裁掉首尾数字静音，保留极短头尾，避免逐句拼接出现长静音拖沓(不连贯)。"""
    a = np.asarray(arr, dtype="float32")
    idx = np.where(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a
    start = max(0, idx[0] - int(sr * keep_lead))
    end = min(len(a), idx[-1] + int(sr * keep_tail))
    return a[start:end]


def _concat_wavs(wavs, sr, gap_s: float = 0.08):
    """把多段 float32 wav 拼接为整段（先裁静音，句间加小停顿，听感连贯）。"""
    gap = np.zeros(int(sr * gap_s), dtype="float32")
    pieces = []
    for i, w in enumerate(wavs):
        pieces.append(_trim_silence(w, sr))
        if i < len(wavs) - 1:
            pieces.append(gap)
    return np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")


def _wav_bytes(arr, sr) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        pcm = (np.clip(arr.astype("float32"), -1.0, 1.0) * 32767).astype("int16")
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ======================================================================
# API：健康 / 挂载
# ======================================================================
@app.get("/api/health")
def health():
    gguf = _get_gguf().availability()
    tts = {"tts": "ready" if gguf["available"] else "missing",
           "backend": "gguf", "gguf": gguf}
    return {
        "text_engine": "online" if text_engine_online() else "offline",
        "tts": tts, "gguf": gguf, "mount": _mount, "role": _active_role,
        "load": dict(_load_state),
        "text_engine_auto": bool(_TEXT_PROC),
    }


@app.get("/api/mounts")
def get_mounts():
    return {"mount": _mount,
            "tts_backends": ["gguf"],
            "adapters": list(list_adapter_names().get("adapters", {}).keys()) or ["voice", "emotion"],
            "emotions": list(_LABELED_EMO_KEYWORDS.keys()),
            "speaking_styles": list(_STYLE_PRESETS),   # 说话风格（StylePlug）
            "default_speaking_style": _DEFAULT_STYLE,
            "roles": list(ROLES.keys()),
            "text_online": text_engine_online()}


# 快速问答（动态下发，不硬编码在前端）
_QUICK_CHIPS = [
    {"q": "你好呀，今天过得怎么样？", "label": "日常寒暄"},
    {"q": "说点温柔的话哄哄我嘛", "label": "温柔哄人"},
    {"q": "今天直播遇到一件特别开心的事！", "label": "分享开心"},
    {"q": "我好难过，陪我说说话吧", "label": "安慰陪伴"},
]


@app.get("/api/quick")
def get_quick_chips():
    return {"chips": _QUICK_CHIPS}


class MountReq(BaseModel):
    tts_backend: str | None = None
    adapter: str | None = None
    emotion_mode: str | None = None
    emotion: str | None = None
    style: str | None = None          # 说话风格（StylePlug）
    role: str | None = None
    want_tts: bool | None = None
    tone_variation: float | None = None


@app.get("/api/debug")
def debug_info():
    """调试模式详情：相关标签(角色/情感/去AI腔) + 生成速度 + 显卡资源 + TTS 挂载"""
    gpu = None
    try:
        import subprocess
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if raw:
            name, used, total, util = map(str.strip, raw.split(","))
            gpu = {"name": name, "used_mb": int(used), "total_mb": int(total),
                   "util_gpu": util}
    except Exception:
        gpu = None
    import psutil
    proc_ram = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    # 文本引擎（04）调试信息：角色锚点/去AI腔标签 + 生成速度统计
    text_debug = None
    try:
        rq = urllib.request.Request(_TEXT_BASE.rstrip("/") + "/debug")
        with urllib.request.urlopen(rq, timeout=5) as resp:
            text_debug = json.loads(resp.read().decode("utf-8"))
    except Exception:
        text_debug = None
    role_cfg = ROLES.get(_active_role, {})
    gguf = _get_gguf().availability()
    return {
        "gpu": gpu,
        "process_ram_mb": proc_ram,
        "text_debug": text_debug,
        "mount": dict(_mount),
        "role": {"active": _active_role,
                 "desc": role_cfg.get("desc"),
                 "traits": role_cfg.get("traits"),
                 "catchphrases": role_cfg.get("catchphrases"),
                 "emotion_keywords": role_cfg.get("emotion_keywords")},
        "tts": {"backend": "gguf", "gguf": gguf,
                "adapters": ["voice", "emotion"],
                "style_lora_loaded": bool(gguf.get("style_lora_loaded"))},
        "styles": list(_KNOWN_STYLES),
        "speaking_styles": list(_STYLE_PRESETS),   # 说话风格（StylePlug），默认
        "default_speaking_style": _DEFAULT_STYLE,
        "emotions": list(_LABELED_EMO_KEYWORDS.keys()),
        "load": dict(_load_state),
    }


# ======================================================================
# 模型加载（前端选择/手动加载对应模型；启动时可预加载默认后端）
# ======================================================================
_load_state = {"state": "idle", "backend": None, "message": "尚未加载 TTS 模型",
               "elapsed": 0.0}
_load_lock = threading.Lock()


def _set_load(**kw):
    with _load_lock:
        _load_state.update(kw)


def load_tts_backend(backend: str | None = None):
    """加载外挂 TTS 模型（llama-tts GGUF，单一方案）。
    重复加载时若正在加载则直接返回当前进度，不重复起。"""
    backend = "gguf"
    with _load_lock:
        if _load_state["state"] == "loading":
            return dict(_load_state)
        _load_state.update({"state": "loading", "backend": backend,
                            "message": f"正在加载 {backend} 模型…", "elapsed": 0.0})
    t0 = time.time()
    log("info", "开始加载外挂 TTS 模型", backend=backend)
    try:
        g = _get_gguf()
        a = g.availability()
        if not a["available"]:
            raise RuntimeError(a["reason"])
        g.synthesize("嗯，我在呢。", emotion="平静", ref=None, rate=1.0)  # 预热确认可出音频
        msg = "gguf 外挂已就绪 (llama-tts CUDA INT4)"
        _set_load(state="ready", message=msg, elapsed=round(time.time() - t0, 1))
        log("info", "外挂 TTS 加载完成", detail=msg, elapsed=_t(time.time() - t0))
    except Exception as e:
        msg = f"{backend} 加载失败: {str(e)[:200]}"
        _set_load(state="error", message=msg, elapsed=round(time.time() - t0, 1))
        log("error", "外挂 TTS 加载失败", elapsed=_t(time.time() - t0), err=str(e)[:300])
    return dict(_load_state)


def _preload_gguf_async():
    try:
        if _load_state["state"] == "idle":
            load_tts_backend("gguf")
    except Exception:
        pass


class LoadReq(BaseModel):
    backend: str | None = None


@app.post("/api/tts/load")
def api_tts_load(req: LoadReq | None = None):
    backend = (req.backend if req else None) or _mount["tts_backend"]
    return load_tts_backend(backend)


@app.get("/api/tts/load")
def api_tts_load_status():
    return dict(_load_state)


@app.post("/api/mounts")
def set_mount(req: MountReq):
    global _active_role
    for k in ("tts_backend", "adapter", "emotion_mode", "emotion", "style", "role",
              "want_tts", "tone_variation"):
        v = getattr(req, k, None)
        if v is not None:
            if k == "role":
                if v not in ROLES:
                    raise HTTPException(404, f"未知角色: {v}")
                _active_role = v
                _mount["role"] = v
            elif k == "style":
                # 说话风格归一化到合法风格表；非法回退默认
                _mount[k] = v if v in _STYLE_PRESETS else _DEFAULT_STYLE
            else:
                if k == "tts_backend":
                    _mount[k] = "gguf"   # 单一外挂方案，强制 gguf
                else:
                    _mount[k] = v
    return {"mount": _mount}


# ======================================================================
# API：角色设定
# ======================================================================
@app.get("/api/roles")
def list_roles():
    return {"active": _active_role, "roles": {k: v for k, v in ROLES.items()}}


class RoleReq(BaseModel):
    name: str = "缘圆"
    persona: str | None = None
    desc: str | None = None
    traits: dict | None = None
    catchphrases: list | None = None
    emotion_keywords: dict | None = None


@app.put("/api/roles")
def update_role(req: RoleReq):
    if req.name not in ROLES:
        raise HTTPException(404, f"未知角色: {req.name}")
    r = ROLES[req.name]
    if req.persona is not None:
        r["persona"] = req.persona.strip()
    if req.desc is not None:
        r["desc"] = req.desc.strip()
    if req.traits:
        r["traits"].update({k: float(v) for k, v in req.traits.items() if v is not None})
    if req.catchphrases:
        r["catchphrases"] = [str(x) for x in req.catchphrases]
    if req.emotion_keywords:
        r["emotion_keywords"].update(req.emotion_keywords)
    _save_roles()
    return {"ok": True, "role": r}


_ROLES_FILE = SESS_DIR.parent / "roles.json"


def _save_roles():
    try:
        _ROLES_FILE.write_text(json.dumps(ROLES, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    except Exception:
        pass


def _load_roles_file():
    global ROLES, _active_role
    try:
        if _ROLES_FILE.exists():
            data = json.loads(_ROLES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                ROLES = data
                if _active_role not in ROLES:
                    _active_role = next(iter(ROLES), "缘圆")
    except Exception:
        pass


# ======================================================================
# API：会话
# ======================================================================
@app.get("/api/sessions")
def list_sessions():
    items = []
    for s in _sessions.values():
        msgs = s.get("messages", [])
        last = msgs[-1]["content"][:40] if msgs else ""
        n_audio = sum(1 for m in msgs
                      if m.get("role") == "assistant" and m.get("audio"))
        words = sum(len(str(m.get("content") or ""))
                    for m in msgs if m.get("role") == "assistant")
        items.append({
            "id": s["id"], "title": s.get("title") or "新对话",
            "created": s["created"], "updated": s["updated"],
            "n_msgs": len(msgs), "n_audio": n_audio, "words": words,
            "last": last,
        })
    items.sort(key=lambda x: x["updated"], reverse=True)
    return {"sessions": items}


class NewSessionReq(BaseModel):
    title: str = "新对话"


@app.post("/api/sessions")
def create_session(req: NewSessionReq = None):
    s = _new_session(req.title if req else "新对话")
    return {"id": s["id"], "title": s["title"]}


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    s = _get_session(sid)
    return {"id": s["id"], "title": s["title"], "messages": s["messages"]}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    if sid in _sessions:
        del _sessions[sid]
        try:
            conn = _db_conn()
            conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
            conn.commit()
            conn.close()
        except Exception:
            pass
    return {"ok": True}


class TalkReq(BaseModel):
    content: str
    want_tts: bool | None = None
    backend: str | None = None
    emotion_mode: str | None = None
    emotion: str | None = None
    style: str | None = None          # 说话风格（StylePlug）
    role: str | None = None
    adapter: str | None = None
    max_new: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tone_variation: float | None = None


def detect_emotion_fast(reply: str, user_msg: str = ""):
    """情感识别加速：先走离线词典(不占 LLM)，置信不足才回退 LLM，省掉整轮文本引擎调用。"""
    r = detect_emotion(reply, text_chat_fn=None, user_msg=user_msg)
    if r.source != "default" and r.confidence >= 0.55:
        return r
    return detect_emotion(reply, text_chat_fn=call_text_engine, user_msg=user_msg)


def _route_emotion(user_content: str) -> str:
    """情感向量路由：生成文本前先对'用户消息'做情感识别，得路由标签 route_emo。

    路由标签会随 payload 传给 04 文本引擎（emotion=route_emo, scale_emo=1.0），
    驱动其 logits 情感偏置（70% 角色情感 + 30% 引擎优化情感）。失败回退"平静"。
    """
    try:
        r = detect_emotion_fast(user_content or "", user_content or "")
        emo = (r.label if r else "") or "平静"
        log("info", "情感向量路由", route_emo=emo, confidence=getattr(r, "confidence", None),
            source=getattr(r, "source", None), target="04引擎emotion偏置")
    except Exception as e:
        emo = "平静"
        log("warn", "情感向量路由失败，回退", fallback=emo,
            err=f"{type(e).__name__}: {e}")
    return emo


def _adaptive_max_new(route_emo: str, req_max_new: int | None) -> int:
    """长度自适应：请求显式传 max_new 以请求为准；否则按路由情感给长度。

    安慰类（悲伤/温柔/激动/平静深聊）给足长度；日常类（开心/俏皮/兴奋/撒娇）短促。
    """
    if req_max_new is not None:
        return req_max_new
    return _MAX_NEW_BY_EMO.get(route_emo or "", _MAX_NEW_DEFAULT)


@app.post("/api/sessions/{sid}/talk")
def talk(sid: str, req: TalkReq):
    """问答/对话：追加用户消息 → 文本生成 →（可选）自动情感 → 本地 TTS。"""
    s = _get_session(sid)
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(422, "消息不能为空")

    backend = req.backend or _mount["tts_backend"]
    want_tts = req.want_tts if req.want_tts is not None else _mount["want_tts"]
    emotion_mode = req.emotion_mode or _mount["emotion_mode"]
    role = req.role or _active_role
    adapter = req.adapter or _mount["adapter"]
    if role not in ROLES:
        raise HTTPException(404, f"未知角色: {role}")

    # 1) 追加用户消息
    s["messages"].append({"role": "user", "content": content, "ts": time.time()})

    # 2) 情感向量路由 + 长度自适应：生成文本前先对"用户消息"做情感识别
    route_emo = _route_emotion(content)
    max_new = _adaptive_max_new(route_emo, req.max_new)

    # 3) 文本生成（角色 persona 作为 system 覆盖 04 角色包；送最近 20 条上下文）
    role_cfg = ROLES[role]
    msgs = [{"role": "system", "content": role_cfg["persona"]}]
    for m in s["messages"][-20:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    payload = {
        "messages": msgs,
        "max_new": max_new,
        "temperature": req.temperature if req.temperature is not None else _mount["temperature"],
        "top_p": req.top_p if req.top_p is not None else _mount["top_p"],
        "top_k": req.top_k or _mount["top_k"],
        "role": "default", "seed": None,   # 04 引擎只注册 default 角色；角色设定经 system 消息注入
        "emotion": route_emo, "scale_emo": 1.0,   # 04 引擎情感向量路由：70%角色情感+30%引擎优化情感
    }
    chat_resp = call_text_engine(payload)
    raw_reply = (_extract_text(chat_resp) or "").strip()
    # 解析自然语言风格标记 [风格]文本(语速)，括号内容用户不可见；剩余动作提示剥离
    segs, display = _parse_style_markup(raw_reply)
    reply, actions = _strip_actions(display)
    if not reply:
        reply, actions = _strip_actions(raw_reply)
    # 4) 句句有回应兜底：空回复或剥完括号后 <2 字 → 附兜底 persona 后缀重试一次；仍空则结构化报错
    if not raw_reply or len(reply or "") < 2:
        log("warn", "文本引擎回复为空/过短，附兜底 persona 重试", sid=sid,
            raw_len=len(raw_reply), stripped_len=len(reply or ""), route_emo=route_emo)
        payload_retry = dict(payload)
        payload_retry["messages"] = (
            [{"role": "system", "content": role_cfg["persona"] + _FALLBACK_SUFFIX}] + msgs[1:])
        chat_resp = call_text_engine(payload_retry)
        raw_reply = (_extract_text(chat_resp) or "").strip()
        segs, display = _parse_style_markup(raw_reply)
        reply, actions = _strip_actions(display)
        if not reply:
            reply, actions = _strip_actions(raw_reply)
        if not raw_reply or len(reply or "") < 2:
            raise HTTPException(502, "文本引擎连续两次返回空回复（已附兜底 persona 重试）")

    # 5) 情感：auto 词典优先(快) → LLM 兜底；manual 用指定
    if emotion_mode == "auto":
        r = detect_emotion_fast(reply, content)
        emotion = r.label
        emo_info = {"mode": "auto", "label": r.label, "confidence": r.confidence,
                    "source": r.source}
    else:
        emotion = req.emotion or _mount["emotion"]
        emo_info = {"mode": "manual", "label": emotion, "confidence": 1.0, "source": "manual"}

    # 6) 本地 TTS（可选）：整段一次合成（引擎已带 [情感]/(语速) 指令，模型原生演绎）
    #    由文本引擎强绑定 TTS → 音色一致、情感/语速曲线连续，不再逐句随机漂移。
    audio_url, tts_meta = None, None
    _speaking_style = req.style or _mount.get("style", _DEFAULT_STYLE)
    if want_tts:
        log("info", "talk 整段一次 TTS", sid=sid, n_seg=len(segs), emotion=emotion,
            style=_speaking_style)
        try:
            # 混合方案：逐段合成（segs 是 [(文本, 情感, 语速)]），真实情感幅度+语速波浪
            wav, sr, m = synth_flow_tts(segs, emotion=emotion, style=_speaking_style)
            fname = f"{sid}_{len(s['messages'])}_full_{int(time.time())}.wav"
            (AUDIO_DIR / fname).write_bytes(_wav_bytes(wav, sr))
            audio_url = f"/audio/{fname}"
            seg_infos = [{"index": i, "text": t, "style": s or emotion,
                          "rate": r, "audio": audio_url}
                         for i, (t, s, r) in enumerate(segs)]
            log("debug", "talk 整段音频已落盘", fname=fname, dur=_t(len(wav) / sr))
            tts_meta = {"backend": "gguf", "mode": "paragraph_once",
                        "n_sentences": len(segs), "meta": m,
                        "styles": [i["style"] for i in seg_infos],
                        "rates": [i["rate"] for i in seg_infos],
                        "segments": seg_infos}
            log("info", "talk 整段 TTS 完成", n_seg=len(segs), audio=audio_url)
        except Exception as ex:
            tts_meta = {"error": str(ex)[:400]}
            log("error", "talk 整段 TTS 合成中断", sid=sid, err=str(ex)[:400])

    # 7) 追加助手消息
    am = {"role": "assistant", "content": reply, "ts": time.time(),
          "emotion": emotion, "emotion_info": emo_info, "actions": actions,
          "audio": audio_url, "tts_meta": tts_meta}
    s["messages"].append(am)
    s["updated"] = time.time()
    if len(s["messages"]) == 2:
        s["title"] = content[:14] + ("…" if len(content) > 14 else "")
    _save_session(s)

    return {
        "reply": reply, "actions": actions, "emotion": emotion, "emotion_info": emo_info,
        "audio": audio_url, "tts_meta": tts_meta,
        "text_stats": {"latency_s": chat_resp.get("latency_s"),
                       "tok_s": chat_resp.get("tok_s")},
    }


@app.post("/api/sessions/{sid}/stream")
def talk_stream(sid: str, req: TalkReq):
    """SSE 流式对话：文本逐字回流 → 末尾 done 事件带 reply/emotion/audio。
    前端用 fetch 流式读取，实现'对方正在输入'体验。"""
    from fastapi.responses import StreamingResponse
    s = _get_session(sid)
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(422, "消息不能为空")

    backend = req.backend or _mount["tts_backend"]
    want_tts = req.want_tts if req.want_tts is not None else _mount["want_tts"]
    emotion_mode = req.emotion_mode or _mount["emotion_mode"]
    role = req.role or _active_role
    adapter = req.adapter or _mount["adapter"]
    if role not in ROLES:
        raise HTTPException(404, f"未知角色: {role}")

    s["messages"].append({"role": "user", "content": content, "ts": time.time()})
    # 情感向量路由 + 长度自适应：生成文本前先对"用户消息"做情感识别
    route_emo = _route_emotion(content)
    max_new = _adaptive_max_new(route_emo, req.max_new)
    role_cfg = ROLES[role]
    msgs = [{"role": "system", "content": role_cfg["persona"]}]
    for m in s["messages"][-20:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    payload = {
        "messages": msgs,
        "max_new": max_new,
        "temperature": req.temperature if req.temperature is not None else _mount["temperature"],
        "top_p": req.top_p if req.top_p is not None else _mount["top_p"],
        "top_k": req.top_k or _mount["top_k"],
        "role": "default", "seed": None,
        "emotion": route_emo, "scale_emo": 1.0,   # 04 引擎情感向量路由：70%角色情感+30%引擎优化情感
    }

    def gen():
        import urllib.request as ur
        t0 = time.time()
        pl = payload
        attempt = 0
        yielded_any = False
        while True:
            attempt += 1
            buf, done_reply = [], None
            try:
                data = json.dumps(pl).encode("utf-8")
                rq = ur.Request(_TEXT_BASE.rstrip("/") + "/chat/stream", data=data,
                                headers={"Content-Type": "application/json"})
                with ur.urlopen(rq, timeout=300) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data: "):
                            continue
                        evt = json.loads(line[6:])
                        if "delta" in evt and evt["delta"]:
                            buf.append(evt["delta"])
                            yielded_any = True
                            yield f"data: {json.dumps({'delta': evt['delta']}, ensure_ascii=False)}\n\n"
                        elif evt.get("done"):
                            done_reply = evt.get("reply", "")
                        elif "error" in evt:
                            raise RuntimeError(evt["error"])
            except Exception as ex:
                yield f"data: {json.dumps({'error': str(ex)[:300]}, ensure_ascii=False)}\n\n"
                return
            if done_reply is None:
                done_reply = "".join(buf)
            # 解析自然语言风格标记 [风格]文本(语速)：括号内容用户不可见，喂给 TTS 逐句控风格/语速
            segs, display = _parse_style_markup(done_reply)
            reply, actions = _strip_actions(display)
            if not reply:
                reply, actions = _strip_actions(done_reply)
            if (reply or "").strip() and len(reply) >= 2:
                break
            # 句句有回应兜底：空/剥完括号后 <2 字 → 附兜底 persona 后缀重试一次；仍空则结构化报错
            if attempt >= 2 or yielded_any:
                yield f"data: {json.dumps({'error': '文本引擎连续两次返回空回复（已附兜底 persona 重试）'}, ensure_ascii=False)}\n\n"
                return
            log("warn", "流式回复为空/过短，附兜底 persona 重试", sid=sid,
                attempt=attempt, route_emo=route_emo, stripped_len=len(reply or ""))
            pl = dict(pl)
            pl["messages"] = (
                [dict(pl["messages"][0], content=role_cfg["persona"] + _FALLBACK_SUFFIX)]
                + pl["messages"][1:])
        # 情感：auto 词典优先(快) → LLM 兜底；manual 用指定
        if emotion_mode == "auto":
            r = detect_emotion_fast(reply, content)
            emotion = r.label
            emo_info = {"mode": "auto", "label": r.label, "confidence": r.confidence,
                        "source": r.source}
        else:
            emotion = req.emotion or _mount["emotion"]
            emo_info = {"mode": "manual", "label": emotion, "confidence": 1.0, "source": "manual"}
        # 先让前端看到完整文本（不再逐字流式，改"对方正在输入"后整段展示）
        yield f"data: {json.dumps({'text_done': True, 'reply': reply}, ensure_ascii=False)}\n\n"
        # 整段一次 TTS：引擎已带 [情感]/(语速) 指令 → 模型原生整段演绎（音色一致+曲线连续）
        audio_url, tts_meta = None, None
        seg_infos = []
        _speaking_style = req.style or _mount.get("style", _DEFAULT_STYLE)
        if want_tts:
            log("info", "stream 整段一次 TTS", sid=sid, n_seg=len(segs), emotion=emotion,
                style=_speaking_style)
            try:
                # 混合方案：逐段合成（segs 是 [(文本, 情感, 语速)]）
                wav, sr, m = synth_flow_tts(segs, emotion=emotion, style=_speaking_style)
                fname = f"{sid}_{len(s['messages'])}_full_{int(time.time())}.wav"
                (AUDIO_DIR / fname).write_bytes(_wav_bytes(wav, sr))
                audio_url = f"/audio/{fname}"
                seg_infos = [{"index": i, "text": t, "style": s or emotion,
                              "rate": r, "audio": audio_url}
                             for i, (t, s, r) in enumerate(segs)]
                log("debug", "stream 整段音频已落盘", fname=fname, dur=_t(len(wav) / sr))
                # 前端逐句按钮仍用统一整段音频；style/rate 用于标注情感/语速曲线
                yield f"data: {json.dumps({'sentence_audio': True, 'index': 0, 'text': reply, 'style': emotion, 'rate': 1.0, 'audio': audio_url, 'meta': m, 'whole': True}, ensure_ascii=False)}\n\n"
                tts_meta = {"backend": "gguf", "mode": "paragraph_once",
                            "n_sentences": len(segs), "meta": m,
                            "styles": [i["style"] for i in seg_infos],
                            "rates": [i["rate"] for i in seg_infos],
                            "segments": seg_infos}
                log("info", "stream 整段 TTS 完成", sid=sid, n_seg=len(segs), audio=audio_url)
            except Exception as ex:
                tts_meta = {"error": str(ex)[:400]}
                log("error", "stream 整段 TTS 合成中断", sid=sid, err=str(ex)[:400])
        # 追加助手消息 + 持久化
        am = {"role": "assistant", "content": reply, "ts": time.time(),
              "emotion": emotion, "emotion_info": emo_info, "actions": actions,
              "audio": audio_url, "tts_meta": tts_meta}
        s["messages"].append(am)
        s["updated"] = time.time()
        if len(s["messages"]) == 2:
            s["title"] = content[:14] + ("…" if len(content) > 14 else "")
        _save_session(s)
        final = {"done": True, "reply": reply, "actions": actions, "emotion": emotion,
                 "emotion_info": emo_info, "audio": audio_url, "tts_meta": tts_meta,
                 "text_stats": {"latency_s": round(time.time() - t0, 2), "tok_s": None}}
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ======================================================================
# API：直接 TTS / 情感识别（供挂载测试）
# ======================================================================
class TTSReq(BaseModel):
    text: str
    emotion: str = "平静"
    backend: str = "gguf"
    adapter: str = "emotion"
    rate: float = 1.0


@app.post("/api/tts/synthesize")
def tts_synthesize(req: TTSReq):
    try:
        wav, sr, meta = synth_tts(req.text, req.emotion, backend=req.backend,
                                  adapter=req.adapter, rate=req.rate)
    except (TTSUnavailable, RuntimeError) as ex:
        raise HTTPException(503, str(ex))
    return Response(content=_wav_bytes(wav, sr), media_type="audio/wav",
                    headers={"X-TTS-Meta": json.dumps(meta, ensure_ascii=True)})


class DetectReq(BaseModel):
    text: str
    user_msg: str = ""


@app.post("/api/tts/detect")
def tts_detect(req: DetectReq):
    r = detect_emotion(req.text or "", text_chat_fn=call_text_engine,
                       user_msg=req.user_msg or "")
    return {"label": r.label, "confidence": r.confidence, "source": r.source}


# ======================================================================
# 音频回放 / 静态资源
# ======================================================================
@app.get("/audio/{fname}")
def audio(fname: str):
    p = AUDIO_DIR / os.path.basename(fname)
    if not p.is_file():
        raise HTTPException(404, "音频不存在")
    return FileResponse(p, media_type="audio/wav")


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="webapp")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


def main():
    global _TEXT_BASE
    import uvicorn
    _load_roles_file()
    ap = argparse.ArgumentParser(description="缘圆一体化对话台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8071)
    ap.add_argument("--text-base", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    _TEXT_BASE = a.text_base
    print(f"[yy-app] 缘圆一体化对话台: http://{a.host}:{a.port}  文本引擎={_TEXT_BASE}")
    print(f"[yy-app] 角色: {list(ROLES.keys())} | 挂载: {_mount}")
    print("[yy-app] TTS: 单一外挂方案 llama-tts GGUF(CUDA INT4, RTF≈0.4)", flush=True)
    log("info", "缘圆一体化对话台启动", host=a.host, port=a.port, text_base=_TEXT_BASE)
    # 文本引擎：若离线则自动拉起（后台线程，等待上线，不阻塞服务）
    threading.Thread(target=lambda: _ensure_text_engine(), daemon=True).start()
    # 程序启动预加载默认 TTS 模型（后台线程，不阻塞服务；环境变量 YY_PRELOAD=0 关闭）
    if os.environ.get("YY_PRELOAD", "1") != "0":
        threading.Thread(target=_preload_gguf_async, daemon=True).start()
        print("[yy-app] 已在后台预加载默认 TTS 模型 (gguf)...", flush=True)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
