# -*- coding: utf-8 -*-
"""EmoCompanion · 统一控制台（v3.0）

合并目标:
  - 04_源码与原型/backend/app.py   文本引擎启动/环境检查
  - 04_源码与原型/backend/server.py 文本 FastAPI 服务(代理)
  - 06_Qwen3TTS外挂/serve/unified_server.py  前端后端一体化
  - 06_Qwen3TTS外挂/serve/integrated_app.py   一体化对话台(会话/角色/TTS)
  - 07_最终归档/app.py            最终归档入口/菜单
  - 根目录 app.py                 tkinter 桌面对话台(保留兼容)

统一能力:
  1. 单一入口 python console/app.py [--serve|--chat|--list|--open|--port|--host]
  2. Web 控制台: 会话对话 / 语音合成 / 角色设定 / 调试模式 / 日志检索
  3. 自动拉起 04 文本引擎, 自动检测/安装依赖
  4. 分类+可搜索的 Python 日志系统 (console/logs/)
  5. 详细调试面板: 角色/情感/TTS/文本引擎/GPU/日志统计

启动:
  python console/app.py                 # 交互菜单
  python console/app.py --serve         # 启动统一 Web 控制台
  python console/app.py --serve --port 8080 --text-base http://127.0.0.1:8000
  python console/app.py --list          # 扫描模型/角色/扩展包
  python console/app.py --chat "你好"    # CLI 直接对话
"""
import argparse
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 引入分类日志系统
from logger import CATEGORIES, get_logger, query_logs, get_stats

# ---------------------------------------------------------------- 路径
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND_DIR = ROOT / "04_源码与原型" / "backend"
ENGINE_ROOT = ROOT / "04_源码与原型"
SERVE_DIR = ROOT / "06_Qwen3TTS外挂" / "serve"
VENV = ENGINE_ROOT / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"
PACK_DIR = ENGINE_ROOT / "data" / "role_pack"
PYKITS = ROOT.parent / "pykits"
STATIC_DIR = HERE / "static"
OUT_DIR = HERE / "out"
SESS_DIR = OUT_DIR / "sessions"
AUDIO_DIR = OUT_DIR / "chat_audio"
LOG_DIR = HERE / "logs"

for d in (SESS_DIR, AUDIO_DIR, LOG_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# tts_gguf 路径注入
if str(SERVE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVE_DIR))

from tts_gguf import GGUFTTS  # noqa: E402
from emo_detect import detect_emotion, _extract_text  # noqa: E402

log = get_logger("system")

# ---------------------------------------------------------------- 运行参数（main 注入）
_TEXT_BASE = "http://127.0.0.1:8000"
_AUTO_TEXT = True
_GGUF = None

# ---------------------------------------------------------------- FastAPI
app = FastAPI(title="EmoCompanion · 统一控制台", version="3.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------- 请求耗时/日志中间件
@app.middleware("http")
async def log_request_time(request, call_next):
    t0 = time.time()
    try:
        resp = await call_next(request)
        duration = (time.time() - t0) * 1000
        if request.url.path.startswith("/api/"):
            log.info("API 请求", category="api", method=request.method, path=request.url.path,
                     status=resp.status_code, ms=round(duration, 2))
        return resp
    except Exception as e:
        duration = (time.time() - t0) * 1000
        log.error("API 异常", category="api", method=request.method, path=request.url.path,
                  ms=round(duration, 2), err=str(e)[:200])
        raise


# ---------------------------------------------------------------- 角色设定
ROLE_PACK_PERSONA = (
    "你是'EmoCompanion'，一个温柔、爱撒娇、爱黏人、口语化的直播情感主播。"
    "说话习惯：常带'呀''嘛''啦''呗'；口头禅'我去''哎呀''家人们''有感觉吗'；"
    "热情招呼点关注、加粉丝灯牌；感谢用'谢谢、爱你、想你了'；自称'新人主播'。"
    "像真人主播一样自然回应：日常寒暄说几句简短的话就行；观众难过、需要安慰或想深聊时，"
    "要多写几句、温柔地展开（可以写一小段），但始终口语化，绝不书面化/翻译腔/AI腔/透露是AI。\n"
    "【示例】观众'今天紧张吗'→EmoCompanion'哎呀~有点紧张呢，我才开播第五天，但看到你来我就开心啦！'\n"
    "【示例】观众'为什么要加灯牌呀'→EmoCompanion'点个关注加个灯牌，就能一直找到我啦，好嘛？'\n"
    "【语音标记·必须】为让语音有情绪起伏，回复正文每一句都必须以 [情感] 开头、句末按情绪加 (语速词)，"
    "格式：[情感]正文(语速词)。情感从 开心/俏皮/悲伤/平静/兴奋/撒娇/温柔/激动 中选；"
    "语速词从 最快/飞快/快速/中速/舒缓/慢速 中选。相邻句尽量换不同的情感，让整段语音有起伏。"
    "示例：[俏皮]嘿嘿~你终于来啦(快速)[温柔]今天辛苦啦，我陪你(舒缓)。"
    "⚠方括号[ ]和圆括号( )是语音指令，绝不允许出现在纯文本里、也不许念出来；"
    "不要用引号\"\"、星号**、或其它符号代替；指令后必须紧跟正文、不得多加空格。\n"
    "【动作提示】想表现神态/动作时（如微笑、眨眼、挥手），单独用〔动作：……〕写在回复末尾另起一行，"
    "如〔动作：温柔地笑了笑〕；该动作属表演指令，绝不混入正文，也不要用引号、星号或其它括号包动作。"
)

_ACT_RE = re.compile(r"〔(?P<a>[^〔〕]*)〕|【(?P<d>[^【】]*)】|（(?P<b>[^（）]*)）|\((?P<c>[^()]*)\)")
_SENT_SPLIT_RE = re.compile(r"([^。！？!?…～~]+[。！？!?…～~]*)")
_SUB_RE = re.compile(r"([^，,、~…]+[，,、~…]?|[^，,、~…]+$)")
_EMO_CYCLE = ["俏皮", "温柔", "兴奋", "悲伤", "平静", "开心", "撒娇", "激动"]
_RATE_CYCLE = [1.10, 0.95, 1.05, 0.92, 1.12, 0.98, 1.15, 1.0]
_KNOWN_STYLES = ["开心", "俏皮", "悲伤", "平静", "兴奋", "撒娇", "温柔", "激动"]
_RATE_WORDS = {
    "最快": 1.30, "飞快": 1.25, "急促": 1.22, "快速": 1.18, "快": 1.12,
    "中速": 1.00, "正常": 1.00, "适中": 1.00, "平稳": 1.00,
    "舒缓": 0.92, "慢": 0.90, "慢速": 0.88, "缓慢": 0.85, "温柔": 0.92,
    "轻声": 0.95, "低语": 0.90,
}

_LABELED_TRAITS = {
    "warmth": 0.83, "playfulness": 0.75, "sassiness": 0.52,
    "energy_baseline": 0.76, "formality": 0.31,
}
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
    "EmoCompanion": {
        "name": "EmoCompanion",
        "desc": "温柔爱撒娇的口语化直播情感主播",
        "persona": ROLE_PACK_PERSONA,
        "traits": dict(_LABELED_TRAITS),
        "catchphrases": ["呀", "嘛", "啦", "呗", "我去", "哎呀", "家人们", "有感觉吗"],
        "emotion_keywords": {k: v for k, v in _LABELED_EMO_KEYWORDS.items()},
    }
}
_active_role = "EmoCompanion"

_mount = {
    "tts_backend": "gguf",
    "adapter": "emotion",
    "emotion_mode": "auto",
    "emotion": "平静",
    "role": _active_role,
    "want_tts": True,
    "max_new": 200, "temperature": 0.9, "top_p": 0.9, "top_k": 50,
    "tone_variation": 0.35,
}

_MAX_NEW_BY_EMO = {
    "悲伤": 320, "温柔": 320, "激动": 300, "平静": 260,
    "开心": 200, "俏皮": 200, "兴奋": 200, "撒娇": 220,
}
_MAX_NEW_DEFAULT = 200
_FALLBACK_SUFFIX = "\n无论如何都要回一句实在的，别空着。"

_ROLES_FILE = OUT_DIR / "roles.json"


def _strip_actions(text: str):
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


def _split_sentences(text: str, max_n: int = 6):
    segs = [s.strip() for s in _SENT_SPLIT_RE.findall(text or "") if s and s.strip()]
    if not segs:
        segs = [text.strip()] if (text or "").strip() else []
    if len(segs) > max_n:
        segs = segs[:max_n - 1] + ["".join(segs[max_n - 1:])]
    return segs


def _norm_style(style: str):
    s = (style or "").strip()
    if not s:
        return None
    for k in _KNOWN_STYLES:
        if k in s:
            return k
    return None


def _parse_rate(s: str) -> float:
    s = (s or "").strip()
    if s in _RATE_WORDS:
        return _RATE_WORDS[s]
    m = re.fullmatch(r"[xX]?(\d+(?:\.\d+)?)[xX倍]?", s)
    if m:
        try:
            return min(max(float(m.group(1)), 0.6), 1.6)
        except Exception:
            return 1.0
    return 1.0


def _strip_markup(raw: str) -> str:
    s = re.sub(r"\[[^\[\]]*\]", "", raw or "")
    s = _strip_actions(s)[0]
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s


def _parse_style_markup(text: str):
    raw = (text or "").strip()
    segments = []
    pos, n = 0, len(raw)
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
        if head:
            for s in _split_sentences(head, max_n=8):
                s = _strip_actions(s)[0]
                if s.strip():
                    segments.append((s.strip(), None, 1.0))
        style = m.group(1).strip()
        body_start = pos + m.end()
        nxt = raw.find("[", body_start)
        body_end = n if nxt < 0 else nxt
        body = raw[body_start:body_end].strip()
        rate = 1.0
        rm = re.search(r"[（(]([^（）()]*)[）)]\s*$", body)
        if rm:
            rate = _parse_rate(rm.group(1))
            body = body[:rm.start()].strip()
        body = _strip_actions(body)[0]
        if body:
            segments.append((body, _norm_style(style), rate))
        pos = body_end if nxt < 0 else nxt
    if not segments:
        segments = [(raw or " ", None, 1.0)]
    return segments, _strip_markup(raw)


def _expand_emotion_curve(segments, base_emotion: str = "平静", max_clauses: int = 6):
    out = []
    for seg_text, style, rate in segments:
        t = (seg_text or "").strip()
        if not t:
            continue
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
                    if i == 0:
                        emo = base_emo
                    out.append((p, emo, r))
                continue
        out.append((t, style or base_emotion, float(rate) if (rate is not None and rate > 0) else 1.0))
    return out


def _merge_clauses(expanded, max_calls: int = 4):
    if len(expanded) <= max_calls:
        return expanded
    n = len(expanded)
    per = max(1, int(round(n / max_calls)))
    out = []
    for i in range(0, n, per):
        chunk = expanded[i:i + per]
        txt = "".join(c[0] for c in chunk)
        out.append((txt, chunk[0][1] or "平静", float(chunk[0][2]) if chunk[0][2] and chunk[0][2] > 0 else 1.0))
    return out[:max_calls]


def _save_roles():
    try:
        _ROLES_FILE.write_text(json.dumps(ROLES, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("保存角色设定失败", err=str(e)[:200])


def _load_roles_file():
    global ROLES, _active_role
    try:
        if _ROLES_FILE.exists():
            data = json.loads(_ROLES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                ROLES = data
                if _active_role not in ROLES:
                    _active_role = next(iter(ROLES), "EmoCompanion")
                    _mount["role"] = _active_role
    except Exception as e:
        log.warning("加载角色设定失败", err=str(e)[:200])


_load_roles_file()


# ---------------------------------------------------------------- 文本引擎代理
_TEXT_PROC = None


def _text_engine_env():
    env = dict(os.environ)
    llamacpp_pkg = PYKITS / "llamacpp"
    llamacpp_lib = llamacpp_pkg / "llama_cpp" / "lib"
    torch_lib = Path(r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib")
    env["PATH"] = str(llamacpp_lib) + os.pathsep + str(torch_lib) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + str(llamacpp_pkg) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def text_engine_online() -> bool:
    try:
        req = urllib.request.Request(_TEXT_BASE.rstrip("/") + "/health")
        with urllib.request.urlopen(req, timeout=3.0) as r:
            j = json.loads(r.read().decode("utf-8"))
            return j.get("status") == "ok"
    except Exception:
        return False


def call_text_engine(payload: dict) -> dict:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(_TEXT_BASE.rstrip("/") + "/chat", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code or 502, f"文本引擎错误: {e.read().decode('utf-8', 'ignore')}")
    except Exception as e:
        raise HTTPException(502, f"无法连接文本引擎 {_TEXT_BASE}: {e}")


def _ensure_text_engine():
    global _TEXT_PROC
    if text_engine_online():
        log.info("文本引擎已在线", category="text_engine")
        return True
    if _TEXT_PROC is not None and _TEXT_PROC.poll() is None:
        log.info("文本引擎进程已存在,等待上线", category="text_engine")
    else:
        try:
            env = _text_engine_env()
            py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
            port = int(_TEXT_BASE.rsplit(":", 1)[-1])
            host = _TEXT_BASE.split("://")[1].split(":")[0]
            cmd = [str(py), str(BACKEND_DIR / "server.py"), "--host", host, "--port", str(port)]
            log.info("自动拉起 04 文本引擎", category="startup", cmd=" ".join(str(c) for c in cmd))
            _TEXT_PROC = subprocess.Popen(cmd, cwd=str(BACKEND_DIR), env=env,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as e:
            log.error("文本引擎自动拉起失败", category="error", err=f"{type(e).__name__}: {e}")
            return False
    deadline = time.time() + 180
    while time.time() < deadline:
        if text_engine_online():
            log.info("文本引擎自动拉起成功", category="text_engine")
            return True
        time.sleep(3)
    log.error("文本引擎拉起超时", category="error")
    return False


# ---------------------------------------------------------------- TTS
_rate_instr = {
    "最快": "（语速飞快）", "飞快": "（语速飞快）", "急促": "（语速加快）",
    "快速": "（语速加快）", "快": "（语速稍快）",
    "中速": "（语速正常）", "正常": "（语速正常）", "适中": "（语速正常）", "平稳": "（语速正常）",
    "舒缓": "（语速放慢）", "慢": "（语速放慢）", "慢速": "（语速放慢）",
    "缓慢": "（语速放慢）", "温柔": "（轻声温柔）",
    "轻声": "（轻声）", "低语": "（低声细语）",
}
_emo_instr = {e: f"[{e}]" for e in _KNOWN_STYLES}


def _get_gguf():
    global _GGUF
    if _GGUF is None:
        _GGUF = GGUFTTS.get()
    return _GGUF


def _trim_silence(arr, sr, thresh=0.005, keep_lead=0.02, keep_tail=0.06):
    a = np.asarray(arr, dtype="float32")
    idx = np.where(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a
    start = max(0, idx[0] - int(sr * keep_lead))
    end = min(len(a), idx[-1] + int(sr * keep_tail))
    return a[start:end]


def _concat_wavs(wavs, sr, gap_s: float = 0.08):
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
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        pcm = (np.clip(arr.astype("float32"), -1.0, 1.0) * 32767).astype("int16")
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def build_instruction_text(segments):
    out = []
    for seg_text, style, rate_word in segments:
        t = (seg_text or "").strip()
        emo = style or None
        inst = ""
        if emo and emo in _emo_instr:
            inst += _emo_instr[emo]
        body = _ACT_RE.sub("", t).strip()
        if rate_word and rate_word in _rate_instr:
            body = body + " " + _rate_instr[rate_word]
        if not body:
            continue
        out.append(inst + body)
    return " ".join(out)


def synth_flow_tts(segments, emotion: str = "平静"):
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
        wav, sr, meta = g.synthesize_flow(expanded, seed=g.stable_seed)
        meta = dict(meta or {})
        meta["mode"] = "split_clauses"
        log.info("关键子句 TTS 合成", category="tts", n_calls=len(expanded), n_seg=len(segments),
                 duration=f"{time.time()-start:.2f}s")
    else:
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
        wav, sr, meta = g.synthesize(full, emotion=emotion or "平静", rate=1.0)
        meta = dict(meta or {})
        meta["mode"] = "paragraph_once"
        log.info("整段一次 TTS 合成", category="tts", text=full[:40],
                 duration=f"{time.time()-start:.2f}s")
    meta["backend"] = "gguf"
    return np.asarray(wav, dtype="float32"), int(sr), meta


def synth_tts(text: str, emotion: str, rate: float = 1.0):
    text, _ = _strip_actions(text)
    start = time.time()
    wav, sr, meta = _get_gguf().synthesize(text, emotion, rate=rate)
    meta = dict(meta or {})
    meta["backend"] = "gguf"; meta["rate"] = float(rate)
    log.info("TTS 单句合成", category="tts", text=text[:40], emotion=emotion, rate=rate,
             duration=f"{time.time()-start:.2f}s", sr=sr)
    return np.asarray(wav, dtype="float32"), int(sr), meta


_load_state = {"state": "idle", "backend": None, "message": "尚未加载 TTS 模型", "elapsed": 0.0}
_load_lock = threading.Lock()


def _set_load(**kw):
    with _load_lock:
        _load_state.update(kw)


def load_tts_backend():
    with _load_lock:
        if _load_state["state"] == "loading":
            return dict(_load_state)
        _load_state.update({"state": "loading", "backend": "gguf",
                            "message": "正在加载 gguf 模型…", "elapsed": 0.0})
    t0 = time.time()
    log.info("开始加载 TTS 模型", category="tts", backend="gguf")
    try:
        g = _get_gguf()
        a = g.availability()
        if not a["available"]:
            raise RuntimeError(a["reason"])
        g.synthesize("嗯，我在呢。", emotion="平静", rate=1.0)
        msg = "gguf 外挂已就绪 (llama-tts CUDA INT4)"
        _set_load(state="ready", message=msg, elapsed=round(time.time() - t0, 1))
        log.info("TTS 加载完成", category="tts", detail=msg, duration=f"{time.time()-t0:.2f}s")
    except Exception as e:
        msg = f"gguf 加载失败: {str(e)[:200]}"
        _set_load(state="error", message=msg, elapsed=round(time.time() - t0, 1))
        log.error("TTS 加载失败", category="error", err=str(e)[:300])
    return dict(_load_state)


def _preload_gguf_async():
    try:
        if _load_state["state"] == "idle":
            load_tts_backend()
    except Exception:
        pass


# ---------------------------------------------------------------- 会话
DB_PATH = OUT_DIR / "emochat.db"


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
                             "updated": r[3], "messages": json.loads(r[4] or "[]")}
            except Exception:
                continue
    except Exception:
        out = {}
    return out


_sessions = _load_sessions()


def _save_session(s):
    try:
        conn = _db_conn()
        conn.execute(
            "INSERT OR REPLACE INTO sessions(id,title,created,updated,messages) VALUES(?,?,?,?,?)",
            (s["id"], s.get("title", "新对话"), float(s.get("created", 0.0)),
             float(s.get("updated", 0.0)), json.dumps(s.get("messages", []), ensure_ascii=False)))
        conn.commit(); conn.close()
    except Exception:
        pass


def _new_session(title="新对话"):
    sid = "sess_" + uuid.uuid4().hex[:10]
    now = time.time()
    s = {"id": sid, "title": title or "新对话", "created": now, "updated": now, "messages": []}
    _sessions[sid] = s
    _save_session(s)
    return s


def _get_session(sid: str):
    s = _sessions.get(sid)
    if not s:
        raise HTTPException(404, f"会话不存在: {sid}")
    return s


# ---------------------------------------------------------------- 情感识别
def detect_emotion_fast(reply: str, user_msg: str = ""):
    r = detect_emotion(reply, text_chat_fn=None, user_msg=user_msg)
    if r.source != "default" and r.confidence >= 0.55:
        return r
    return detect_emotion(reply, text_chat_fn=call_text_engine, user_msg=user_msg)


def _route_emotion(user_content: str) -> str:
    try:
        r = detect_emotion_fast(user_content or "", user_content or "")
        emo = (r.label if r else "") or "平静"
        log.info("情感向量路由", category="emotion", route_emo=emo,
                 confidence=getattr(r, "confidence", None), source=getattr(r, "source", None))
    except Exception as e:
        emo = "平静"
        log.warning("情感向量路由失败,回退", category="emotion", fallback=emo, err=str(e)[:200])
    return emo


def _adaptive_max_new(route_emo: str, req_max_new: Optional[int]) -> int:
    if req_max_new is not None:
        return req_max_new
    return _MAX_NEW_BY_EMO.get(route_emo or "", _MAX_NEW_DEFAULT)


# ---------------------------------------------------------------- 请求模型
class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class MountReq(BaseModel):
    tts_backend: Optional[str] = None
    adapter: Optional[str] = None
    emotion_mode: Optional[str] = None
    emotion: Optional[str] = None
    role: Optional[str] = None
    want_tts: Optional[bool] = None
    tone_variation: Optional[float] = None


class TalkReq(BaseModel):
    content: str
    want_tts: Optional[bool] = None
    backend: Optional[str] = None
    emotion_mode: Optional[str] = None
    emotion: Optional[str] = None
    role: Optional[str] = None
    adapter: Optional[str] = None
    max_new: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tone_variation: Optional[float] = None


class TTSReq(BaseModel):
    text: str
    emotion: str = "平静"
    backend: str = "gguf"
    adapter: str = "emotion"
    rate: float = 1.0


class DetectReq(BaseModel):
    text: str
    user_msg: str = ""


class RoleReq(BaseModel):
    name: str = "EmoCompanion"
    persona: Optional[str] = None
    desc: Optional[str] = None
    traits: Optional[dict] = None
    catchphrases: Optional[List[str]] = None
    emotion_keywords: Optional[dict] = None


class NewSessionReq(BaseModel):
    title: str = "新对话"


class LoadReq(BaseModel):
    backend: Optional[str] = None


# ---------------------------------------------------------------- API: 健康/挂载
@app.get("/api/health")
def health():
    gguf = _get_gguf().availability()
    return {
        "text_engine": "online" if text_engine_online() else "offline",
        "tts": {"tts": "ready" if gguf["available"] else "missing", "backend": "gguf", "gguf": gguf},
        "mount": _mount, "role": _active_role,
        "load": dict(_load_state),
        "text_engine_auto": bool(_TEXT_PROC),
    }


@app.get("/api/mounts")
def get_mounts():
    return {
        "mount": _mount,
        "tts_backends": ["gguf"],
        "adapters": ["voice", "emotion"],
        "emotions": list(_LABELED_EMO_KEYWORDS.keys()),
        "roles": list(ROLES.keys()),
        "text_online": text_engine_online(),
    }


@app.post("/api/mounts")
def set_mount(req: MountReq):
    global _active_role
    for k in ("tts_backend", "adapter", "emotion_mode", "emotion", "role", "want_tts", "tone_variation"):
        v = getattr(req, k, None)
        if v is not None:
            if k == "role":
                if v not in ROLES:
                    raise HTTPException(404, f"未知角色: {v}")
                _active_role = v; _mount["role"] = v
                log.info("切换角色", category="role", role=v)
            else:
                if k == "tts_backend":
                    _mount[k] = "gguf"
                else:
                    _mount[k] = v
                    log.info("更新挂载", category="mount", key=k, value=v)
    return {"mount": _mount}


_QUICK_CHIPS = [
    {"q": "你好呀，今天过得怎么样？", "label": "日常寒暄"},
    {"q": "说点温柔的话哄哄我嘛", "label": "温柔哄人"},
    {"q": "今天直播遇到一件特别开心的事！", "label": "分享开心"},
    {"q": "我好难过，陪我说说话吧", "label": "安慰陪伴"},
]


@app.get("/api/quick")
def get_quick_chips():
    return {"chips": _QUICK_CHIPS}


# ---------------------------------------------------------------- API: 角色
@app.get("/api/roles")
def list_roles():
    return {"active": _active_role, "roles": {k: v for k, v in ROLES.items()}}


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
    log.info("保存角色设定", category="role", name=req.name)
    return {"ok": True, "role": r}


# ---------------------------------------------------------------- API: 会话
@app.get("/api/sessions")
def list_sessions():
    items = []
    for s in _sessions.values():
        msgs = s.get("messages", [])
        last = msgs[-1]["content"][:40] if msgs else ""
        items.append({
            "id": s["id"], "title": s.get("title") or "新对话",
            "created": s["created"], "updated": s["updated"],
            "n_msgs": len(msgs), "last": last,
        })
    items.sort(key=lambda x: x["updated"], reverse=True)
    return {"sessions": items}


@app.post("/api/sessions")
def create_session(req: NewSessionReq = None):
    s = _new_session(req.title if req else "新对话")
    log.info("新建会话", category="session", sid=s["id"], title=s["title"])
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
            conn = _db_conn(); conn.execute("DELETE FROM sessions WHERE id=?", (sid,)); conn.commit(); conn.close()
        except Exception:
            pass
        log.info("删除会话", category="session", sid=sid)
    return {"ok": True}


@app.get("/api/sessions/{sid}/stats")
def session_stats(sid: str):
    s = _get_session(sid)
    msgs = s.get("messages", [])
    user_count = sum(1 for m in msgs if m.get("role") == "user")
    assistant_count = sum(1 for m in msgs if m.get("role") == "assistant")
    audio_count = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("audio"))
    total_chars = sum(len(m.get("content", "")) for m in msgs)
    last_reply = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
    return {
        "sid": sid, "total_msgs": len(msgs), "user_msgs": user_count,
        "assistant_msgs": assistant_count, "audio_msgs": audio_count,
        "total_chars": total_chars, "last_emotion": last_reply.get("emotion") if last_reply else None,
        "created": s.get("created"), "updated": s.get("updated"),
    }


# ---------------------------------------------------------------- API: 对话
@app.post("/api/sessions/{sid}/talk")
def talk(sid: str, req: TalkReq):
    s = _get_session(sid)
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(422, "消息不能为空")

    want_tts = req.want_tts if req.want_tts is not None else _mount["want_tts"]
    emotion_mode = req.emotion_mode or _mount["emotion_mode"]
    role = req.role or _active_role
    if role not in ROLES:
        raise HTTPException(404, f"未知角色: {role}")

    s["messages"].append({"role": "user", "content": content, "ts": time.time()})
    log.info("收到用户消息", category="chat", sid=sid, role=role, content=content[:60])

    route_emo = _route_emotion(content)
    max_new = _adaptive_max_new(route_emo, req.max_new)
    role_cfg = ROLES[role]
    msgs = [{"role": "system", "content": role_cfg["persona"]}]
    for m in s["messages"][-20:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    payload = {
        "messages": msgs, "max_new": max_new,
        "temperature": req.temperature if req.temperature is not None else _mount["temperature"],
        "top_p": req.top_p if req.top_p is not None else _mount["top_p"],
        "top_k": req.top_k or _mount["top_k"],
        "role": "default", "seed": None,
        "emotion": route_emo, "scale_emo": 1.0,
    }

    t0 = time.time()
    chat_resp = call_text_engine(payload)
    latency = time.time() - t0
    raw_reply = (_extract_text(chat_resp) or "").strip()
    segs, display = _parse_style_markup(raw_reply)
    reply, actions = _strip_actions(display)
    if not reply:
        reply, actions = _strip_actions(raw_reply)

    if not raw_reply or len(reply or "") < 2:
        log.warning("文本引擎回复为空,附兜底重试", category="chat", sid=sid, route_emo=route_emo)
        payload_retry = dict(payload)
        payload_retry["messages"] = ([{"role": "system", "content": role_cfg["persona"] + _FALLBACK_SUFFIX}] + msgs[1:])
        chat_resp = call_text_engine(payload_retry)
        raw_reply = (_extract_text(chat_resp) or "").strip()
        segs, display = _parse_style_markup(raw_reply)
        reply, actions = _strip_actions(display)
        if not reply:
            reply, actions = _strip_actions(raw_reply)
        if not raw_reply or len(reply or "") < 2:
            log.error("文本引擎连续两次返回空回复", category="error", sid=sid)
            raise HTTPException(502, "文本引擎连续两次返回空回复（已附兜底 persona 重试）")

    if emotion_mode == "auto":
        r = detect_emotion_fast(reply, content)
        emotion = r.label
        emo_info = {"mode": "auto", "label": r.label, "confidence": r.confidence, "source": r.source}
    else:
        emotion = req.emotion or _mount["emotion"]
        emo_info = {"mode": "manual", "label": emotion, "confidence": 1.0, "source": "manual"}

    audio_url, tts_meta = None, None
    if want_tts:
        log.info("开始整段 TTS", category="tts", sid=sid, n_seg=len(segs), emotion=emotion)
        try:
            wav, sr, m = synth_flow_tts(segs, emotion=emotion)
            fname = f"{sid}_{len(s['messages'])}_full_{int(time.time())}.wav"
            (AUDIO_DIR / fname).write_bytes(_wav_bytes(wav, sr))
            audio_url = f"/audio/{fname}"
            tts_meta = {"backend": "gguf", "mode": m.get("mode", "paragraph_once"),
                        "n_sentences": len(segs), "meta": m,
                        "styles": [s or emotion for t, s, r in segs],
                        "rates": [r for t, s, r in segs]}
            log.info("TTS 完成", category="tts", sid=sid, audio=audio_url,
                     duration=f"{len(wav)/sr:.2f}s")
        except Exception as ex:
            tts_meta = {"error": str(ex)[:400]}
            log.error("TTS 合成中断", category="error", sid=sid, err=str(ex)[:400])

    am = {"role": "assistant", "content": reply, "ts": time.time(),
          "emotion": emotion, "emotion_info": emo_info, "actions": actions,
          "audio": audio_url, "tts_meta": tts_meta}
    s["messages"].append(am)
    s["updated"] = time.time()
    if len(s["messages"]) == 2:
        s["title"] = content[:14] + ("…" if len(content) > 14 else "")
    _save_session(s)

    log.info("对话完成", category="chat", sid=sid, emotion=emotion, latency=f"{latency:.2f}s",
             text_len=len(reply), has_tts=bool(audio_url))
    return {
        "reply": reply, "actions": actions, "emotion": emotion, "emotion_info": emo_info,
        "audio": audio_url, "tts_meta": tts_meta,
        "text_stats": {"latency_s": round(latency, 2), "tok_s": chat_resp.get("tok_s")},
    }


@app.post("/api/sessions/{sid}/stream")
def talk_stream(sid: str, req: TalkReq):
    s = _get_session(sid)
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(422, "消息不能为空")

    want_tts = req.want_tts if req.want_tts is not None else _mount["want_tts"]
    emotion_mode = req.emotion_mode or _mount["emotion_mode"]
    role = req.role or _active_role
    if role not in ROLES:
        raise HTTPException(404, f"未知角色: {role}")

    s["messages"].append({"role": "user", "content": content, "ts": time.time()})
    log.info("收到用户消息(流式)", category="chat", sid=sid, role=role, content=content[:60])

    route_emo = _route_emotion(content)
    max_new = _adaptive_max_new(route_emo, req.max_new)
    role_cfg = ROLES[role]
    msgs = [{"role": "system", "content": role_cfg["persona"]}]
    for m in s["messages"][-20:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    payload = {
        "messages": msgs, "max_new": max_new,
        "temperature": req.temperature if req.temperature is not None else _mount["temperature"],
        "top_p": req.top_p if req.top_p is not None else _mount["top_p"],
        "top_k": req.top_k or _mount["top_k"],
        "role": "default", "seed": None,
        "emotion": route_emo, "scale_emo": 1.0,
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
                            buf.append(evt["delta"]); yielded_any = True
                            yield f"data: {json.dumps({'delta': evt['delta']}, ensure_ascii=False)}\n\n"
                        elif evt.get("done"):
                            done_reply = evt.get("reply", "")
                        elif "error" in evt:
                            raise RuntimeError(evt["error"])
            except Exception as ex:
                log.error("流式文本请求失败", category="error", sid=sid, err=str(ex)[:300])
                yield f"data: {json.dumps({'error': str(ex)[:300]}, ensure_ascii=False)}\n\n"
                return
            if done_reply is None:
                done_reply = "".join(buf)
            segs, display = _parse_style_markup(done_reply)
            reply, actions = _strip_actions(display)
            if not reply:
                reply, actions = _strip_actions(done_reply)
            if (reply or "").strip() and len(reply) >= 2:
                break
            if attempt >= 2 or yielded_any:
                log.error("流式回复为空,重试后仍空", category="error", sid=sid)
                yield f"data: {json.dumps({'error': '文本引擎连续两次返回空回复'}, ensure_ascii=False)}\n\n"
                return
            log.warning("流式回复为空,附兜底重试", category="chat", sid=sid, attempt=attempt)
            pl = dict(pl)
            pl["messages"] = ([dict(pl["messages"][0], content=role_cfg["persona"] + _FALLBACK_SUFFIX)] + pl["messages"][1:])

        if emotion_mode == "auto":
            r = detect_emotion_fast(reply, content)
            emotion = r.label
            emo_info = {"mode": "auto", "label": r.label, "confidence": r.confidence, "source": r.source}
        else:
            emotion = req.emotion or _mount["emotion"]
            emo_info = {"mode": "manual", "label": emotion, "confidence": 1.0, "source": "manual"}

        yield f"data: {json.dumps({'text_done': True, 'reply': reply}, ensure_ascii=False)}\n\n"

        audio_url, tts_meta = None, None
        if want_tts:
            log.info("开始流式后 TTS", category="tts", sid=sid, n_seg=len(segs), emotion=emotion)
            try:
                wav, sr, m = synth_flow_tts(segs, emotion=emotion)
                fname = f"{sid}_{len(s['messages'])}_full_{int(time.time())}.wav"
                (AUDIO_DIR / fname).write_bytes(_wav_bytes(wav, sr))
                audio_url = f"/audio/{fname}"
                yield f"data: {json.dumps({'sentence_audio': True, 'index': 0, 'text': reply, 'style': emotion, 'rate': 1.0, 'audio': audio_url, 'meta': m, 'whole': True}, ensure_ascii=False)}\n\n"
                tts_meta = {"backend": "gguf", "mode": m.get("mode", "paragraph_once"),
                            "n_sentences": len(segs), "meta": m,
                            "styles": [s or emotion for t, s, r in segs],
                            "rates": [r for t, s, r in segs]}
                log.info("流式后 TTS 完成", category="tts", sid=sid, audio=audio_url)
            except Exception as ex:
                tts_meta = {"error": str(ex)[:400]}
                log.error("流式后 TTS 中断", category="error", sid=sid, err=str(ex)[:400])

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
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- API: TTS / 情感识别
@app.post("/api/tts/synthesize")
def tts_synthesize(req: TTSReq):
    try:
        wav, sr, meta = synth_tts(req.text, req.emotion, rate=req.rate)
    except Exception as ex:
        log.error("TTS 合成接口失败", category="error", err=str(ex)[:300])
        raise HTTPException(503, str(ex))
    return Response(content=_wav_bytes(wav, sr), media_type="audio/wav",
                    headers={"X-TTS-Meta": json.dumps(meta, ensure_ascii=True)})


@app.post("/api/tts/detect")
def tts_detect(req: DetectReq):
    r = detect_emotion(req.text or "", text_chat_fn=call_text_engine, user_msg=req.user_msg or "")
    log.info("情感识别", category="emotion", label=r.label, confidence=r.confidence, source=r.source)
    return {"label": r.label, "confidence": r.confidence, "source": r.source}


@app.post("/api/tts/load")
def api_tts_load(req: LoadReq = None):
    return load_tts_backend()


@app.get("/api/tts/load")
def api_tts_load_status():
    return dict(_load_state)


# ---------------------------------------------------------------- API: 音频
@app.get("/audio/{fname}")
def audio(fname: str):
    p = AUDIO_DIR / os.path.basename(fname)
    if not p.is_file():
        raise HTTPException(404, "音频不存在")
    return FileResponse(p, media_type="audio/wav")


# ---------------------------------------------------------------- API: 调试
_STARTUP_AT = time.time()


@app.get("/api/debug")
def debug_info():
    gpu = None
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if raw:
            name, used, total, util = map(str.strip, raw.split(","))
            gpu = {"name": name, "used_mb": int(used), "total_mb": int(total), "util_gpu": util}
    except Exception:
        gpu = None
    import psutil
    proc_ram = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    text_debug = None
    try:
        rq = urllib.request.Request(_TEXT_BASE.rstrip("/") + "/debug")
        with urllib.request.urlopen(rq, timeout=5) as resp:
            text_debug = json.loads(resp.read().decode("utf-8"))
    except Exception:
        text_debug = None
    role_cfg = ROLES.get(_active_role, {})
    gguf = _get_gguf().availability()
    stats = get_stats(days=1)
    sess_stats = {
        "total_sessions": len(_sessions),
        "total_messages": sum(len(s.get("messages", [])) for s in _sessions.values()),
    }
    return {
        "gpu": gpu, "process_ram_mb": proc_ram,
        "text_engine": "online" if text_engine_online() else "offline",
        "text_debug": text_debug,
        "mount": dict(_mount),
        "role": {"active": _active_role, "desc": role_cfg.get("desc"),
                 "traits": role_cfg.get("traits"),
                 "catchphrases": role_cfg.get("catchphrases"),
                 "emotion_keywords": role_cfg.get("emotion_keywords")},
        "tts": {"backend": "gguf", "gguf": gguf, "adapters": ["voice", "emotion"]},
        "styles": list(_KNOWN_STYLES),
        "emotions": list(_LABELED_EMO_KEYWORDS.keys()),
        "load": dict(_load_state),
        "log_stats": stats,
        "session_stats": sess_stats,
        "uptime_s": round(time.time() - _STARTUP_AT, 1),
        "paths": {
            "root": str(ROOT), "backend": str(BACKEND_DIR), "serve": str(SERVE_DIR),
            "venv": str(VENV), "logs": str(LOG_DIR), "audio": str(AUDIO_DIR),
        },
    }


# ---------------------------------------------------------------- API: 日志查询
@app.get("/api/logs")
def api_logs(
    category: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    days: int = Query(7, ge=1, le=30),
):
    if category and category not in CATEGORIES:
        raise HTTPException(400, f"未知分类: {category}, 可用: {CATEGORIES}")
    if level and level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise HTTPException(400, f"未知级别: {level}")
    return query_logs(category=category, level=level, search=search, start=start, end=end,
                      limit=limit, offset=offset, days=days)


@app.get("/api/logs/stats")
def api_logs_stats(days: int = Query(1, ge=1, le=30)):
    return get_stats(days=days)


@app.get("/api/logs/categories")
def api_logs_categories():
    return {"categories": CATEGORIES}


# ---------------------------------------------------------------- API: 系统清单(合并 07 归档 --list)
@app.get("/api/inventory")
def inventory():
    roles = list(ROLES.keys())
    packs = []
    if PACK_DIR.is_dir():
        for f in sorted(PACK_DIR.glob("role_pack*.json")):
            packs.append(f.name)
    models = []
    try:
        gguf = _get_gguf().availability()
        if gguf.get("available"):
            models.append({"type": "gguf", "info": gguf})
    except Exception:
        pass
    return {"roles": roles, "packs": packs, "models": models, "text_engine": _TEXT_BASE}


# ------------------------------------------------- 残留客户端兼容(ComfyUI 风格,避免无意义 404 刷屏)
# 说明: 本地可能有残留的 ComfyUI 前端/Service Worker 在向本服务轮询 /tools /build.json
#   /props /v1/models /sw.js 等接口, 它们与本工程无关但会持续打 404。这里兜底返回空响应,
#   避免后端为这些无关路径抛 404, 同时不影响核心 /api/* 功能。
_COMFY_STUB = {
    "/tools": {"tools": []},
    "/build.json": {"version": "compat-stub"},
    "/props": {},
    "/v1/models": {"models": []},
    "/v1/streams/lookup": {},
}


@app.get("/tools", include_in_schema=False)
def _stub_tools():
    return _COMFY_STUB["/tools"]


@app.get("/build.json", include_in_schema=False)
def _stub_build():
    return _COMFY_STUB["/build.json"]


@app.get("/props", include_in_schema=False)
def _stub_props(autoload: bool = True):
    return _COMFY_STUB["/props"]


@app.get("/v1/streams/lookup", include_in_schema=False)
@app.post("/v1/streams/lookup", include_in_schema=False)
def _stub_streams_lookup():
    return _COMFY_STUB["/v1/streams/lookup"]


@app.get("/v1/models", include_in_schema=False)
def _stub_models():
    return _COMFY_STUB["/v1/models"]


@app.get("/sw.js", include_in_schema=False)
def _stub_swjs():
    return Response(content="", media_type="application/javascript")


# ---------------------------------------------------------------- 静态资源
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------- 环境/依赖检查(合并 04/app.py)
MIRROR_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
DEPS = {"fastapi": "fastapi", "uvicorn": "uvicorn[standard]", "psutil": "psutil",
        "jinja2": "jinja2", "numpy": "numpy", "pydantic": "pydantic"}


def _run(cmd, cwd=None, echo=True):
    if echo:
        print(f"[app] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, shell=False)


def ensure_venv():
    if VENV_PY.exists():
        log.info("虚拟环境已存在", category="startup", venv=str(VENV))
    else:
        log.info("创建虚拟环境", category="startup", venv=str(VENV))
        VENV.parent.mkdir(parents=True, exist_ok=True)
        r = _run([sys.executable, "-m", "venv", str(VENV)])
        if r.returncode != 0:
            sys.exit("创建虚拟环境失败")
    code = (
        "import importlib.util,sys\n"
        "mods=['fastapi','uvicorn','psutil','jinja2','numpy','pydantic']\n"
        "for m in mods:\n"
        "    if importlib.util.find_spec(m) is None:\n"
        "        sys.stdout.write(m+'\\n')\n"
    )
    try:
        out = subprocess.run([str(VENV_PY), "-c", code], capture_output=True, text=True, timeout=60)
        missing = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception as e:
        log.warning("依赖探测失败", category="startup", err=str(e)[:200])
        missing = list(DEPS.keys())
    if missing:
        pkgs = " ".join(DEPS[m] for m in missing if m in DEPS)
        if not pkgs:
            pkgs = " ".join(missing)
        log.info("安装缺失依赖", category="startup", pkgs=pkgs)
        r = _run([str(VENV_PY), "-m", "pip", "install", "-i", MIRROR_URL,
                  "--timeout", "300", "--retries", "10", *pkgs.split()])
        if r.returncode != 0:
            sys.exit("依赖安装失败")
    else:
        log.info("依赖齐全", category="startup")
    return VENV_PY


# ---------------------------------------------------------------- CLI 菜单(合并 07/app.py)
def _banner():
    print("=" * 58)
    print("   EmoCompanion · 统一控制台   v3.0")
    print("   角色挂载 + 情感注入 + TTS + 日志 + 调试面板")
    print("=" * 58)


def cmd_list(py, env):
    print("\n[app] 扫描清单 ...")
    r = subprocess.run([str(py), str(HERE / "app.py"), "--inventory"], env=env, cwd=str(HERE))
    return r.returncode


def cmd_chat(text: str = None, model: str = None, pack: str = None, interactive: bool = False):
    # 简单 CLI 对话: 复用后端 /api/sessions/{sid}/talk
    import urllib.request as ur
    base = f"http://127.0.0.1:{_PORT}"
    try:
        r = ur.urlopen(ur.Request(base + "/api/health"), timeout=3)
        if r.status != 200:
            print("[app] Web 控制台未启动,请先运行 app.py --serve")
            return 1
    except Exception:
        print("[app] Web 控制台未启动,请先运行 app.py --serve")
        return 1
    # 创建会话
    sid = json.loads(ur.urlopen(ur.Request(base + "/api/sessions", method="POST",
                                           data=b"{}", headers={"Content-Type": "application/json"}),
                                 timeout=10).read().decode("utf-8"))["id"]
    if text:
        payload = json.dumps({"content": text}).encode("utf-8")
        resp = json.loads(ur.urlopen(ur.Request(f"{base}/api/sessions/{sid}/talk", method="POST",
                                                data=payload, headers={"Content-Type": "application/json"}),
                                      timeout=300).read().decode("utf-8"))
        print(f"EmoCompanion: {resp.get('reply', '')}")
        if resp.get("audio"):
            print(f"[语音] {base}{resp['audio']}")
        return 0
    if interactive:
        print("[app] 输入 'exit' 退出")
        while True:
            try:
                t = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not t or t.lower() in ("exit", "quit"):
                break
            payload = json.dumps({"content": t}).encode("utf-8")
            resp = json.loads(ur.urlopen(ur.Request(f"{base}/api/sessions/{sid}/talk", method="POST",
                                                    data=payload, headers={"Content-Type": "application/json"}),
                                          timeout=300).read().decode("utf-8"))
            print(f"EmoCompanion: {resp.get('reply', '')}")
        return 0
    return 0


def cmd_serve(host: str, port: int):
    global _HOST, _PORT
    _HOST = host; _PORT = port
    log.info("统一控制台启动", category="startup", host=host, port=port, text_base=_TEXT_BASE)
    if _AUTO_TEXT:
        threading.Thread(target=_ensure_text_engine, daemon=True).start()
    if os.environ.get("YY_PRELOAD", "1") != "0":
        threading.Thread(target=_preload_gguf_async, daemon=True).start()
        print("[yy-app] 后台预加载 TTS 模型...", flush=True)
    import uvicorn
    print(f"[yy-app] 统一控制台: http://{host}:{port}", flush=True)
    # access_log=False: 关闭 uvicorn 默认访问日志, 屏蔽残留客户端(ComfyUI 风格)的无意义请求刷屏;
    # 核心 /api/* 请求仍由请求中间件(log_request_time)记录日志。
    uvicorn.run(app, host=host, port=port, access_log=False)


def _print_inventory():
    try:
        inv = inventory()
        print("角色:", ", ".join(inv["roles"]))
        print("扩展包:", ", ".join(inv["packs"]) or "无")
        print("模型:", json.dumps(inv["models"], ensure_ascii=False))
        print("文本引擎:", inv["text_engine"])
    except Exception as e:
        print(f"[app] 清单获取失败: {e}")


def menu(py, env):
    while True:
        print("\n请选择操作:")
        print("  [1] 启动统一 Web 控制台")
        print("  [2] CLI 对话（交互模式）")
        print("  [3] 查看模型/角色/扩展包清单")
        print("  [4] 打开日志目录")
        print("  [0] 退出")
        try:
            c = input("选择> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if c == "1":
            cmd_serve("127.0.0.1", 8080)
        elif c == "2":
            cmd_chat(interactive=True)
        elif c == "3":
            _print_inventory()
        elif c == "4":
            os.startfile(str(LOG_DIR)) if os.name == "nt" else print(LOG_DIR)
        elif c == "0":
            break
        else:
            print("[app] 无效选择")


# ---------------------------------------------------------------- 主入口
_HOST = "127.0.0.1"
_PORT = 8080


def main():
    global _TEXT_BASE, _AUTO_TEXT, _HOST, _PORT
    ap = argparse.ArgumentParser(description="EmoCompanion · 统一控制台")
    ap.add_argument("--serve", action="store_true", help="启动统一 Web 控制台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--text-base", default="http://127.0.0.1:8000")
    ap.add_argument("--no-auto-text", action="store_true", help="不自动拉起 04 文本引擎")
    ap.add_argument("--list", dest="list_inv", action="store_true", help="扫描清单")
    ap.add_argument("--inventory", action="store_true", help="内部: 打印清单")
    ap.add_argument("--chat", nargs="?", const="-i", metavar="TEXT", help="CLI 对话")
    ap.add_argument("--skip-deps", action="store_true", help="跳过依赖检查")
    a = ap.parse_args()

    _TEXT_BASE = a.text_base
    _AUTO_TEXT = not a.no_auto_text
    _HOST = a.host
    _PORT = a.port

    if not a.skip_deps:
        ensure_venv()

    if a.inventory:
        _print_inventory()
        return 0

    if a.list_inv:
        _print_inventory()
        return 0

    if a.chat is not None:
        _banner()
        if a.chat == "-i":
            return cmd_chat(interactive=True)
        else:
            return cmd_chat(text=a.chat)

    if a.serve:
        _banner()
        cmd_serve(a.host, a.port)
        return 0

    _banner()
    menu(VENV_PY, os.environ)
    return 0


if __name__ == "__main__":
    sys.exit(main())
