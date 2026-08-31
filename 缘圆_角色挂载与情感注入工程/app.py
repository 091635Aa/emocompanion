# -*- coding: utf-8 -*-
"""缘圆 · 桌面对话台（LLM + 外挂情感 TTS）

一个零额外依赖的桌面小应用：
  - LLM：复用 04 文本引擎（llama.cpp + Qwen3-4B）。若未启动则自动拉起。
  - TTS：复用 06 外挂情感方案（llama-tts.exe GGUF，voice+emotion LoRA）。
  - 界面：内置 tkinter（无第三方GUI依赖）；语音用 winsound 播放 WAV。
  - 对话：文本按句打上 [风格] 标记，逐句合成 → 拼接整段播放。

用法:
  python app.py                # 默认文本引擎 http://127.0.0.1:8000
  python app.py --text-base http://127.0.0.1:9000
  python app.py --no-auto-llm  # 不自动拉起 04 文本引擎（仅连接已运行实例）
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------- 路径
APP_DIR = Path(__file__).resolve().parent            # 缘圆_角色挂载与情感注入工程
SERVE_DIR = APP_DIR / "06_Qwen3TTS外挂" / "serve"    # 复用 tts_gguf.py
BACKEND_DIR = APP_DIR / "04_源码与原型" / "backend"   # 复用 04 文本引擎
VENV_PY = APP_DIR / "04_源码与原型" / ".venv" / "Scripts" / "python.exe"
PYKITS = APP_DIR.parent / "pykits"

# tts_gguf 依赖的模块目录挂进 sys.path
if str(SERVE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVE_DIR))

from tts_gguf import GGUFTTS  # noqa: E402  （复用外挂情感方案）

TEXT_BASE = "http://127.0.0.1:8000"
AUTO_LLM = True

# ---------------------------------------------------------------- 人设/风格
PERSONA = (
    "你是'缘圆'，一个温柔、爱撒娇、爱黏人、口语化的直播情感主播。"
    "说话习惯：常带'呀''嘛''啦''呗'；口头禅'我去''哎呀''家人们''有感觉吗'。"
    "回答要简短口语化：最多 1~3 句，多用短句，说完就停，绝不长篇大论，绝不书面化/翻译腔/AI腔/透露是AI。\n"
    "【风格标记·必须】你回复里的每一句话前面都必须写 [风格] 语气词（从 开心/俏皮/悲伤/平静/兴奋/撒娇/温柔/激动 里选一个），"
    "句末按情绪按需写 (语速)（快速/慢速/舒缓/轻声）。相邻句尽量换不同的风格，让整段语音有起伏。示例："
    "[俏皮]嘿嘿~你终于来啦(快速) [温柔]今天辛苦啦，我陪你(舒缓)。"
    "这些标记只用于控制语音，绝不解释它、绝不把它念出来。"
)

_KNOWN_STYLES = ["开心", "俏皮", "悲伤", "平静", "兴奋", "撒娇", "温柔", "激动"]
_RATE_WORDS = {
    "最快": 1.30, "飞快": 1.25, "急促": 1.22, "快速": 1.18, "快": 1.12,
    "中速": 1.00, "正常": 1.00, "适中": 1.00, "平稳": 1.00,
    "舒缓": 0.92, "慢": 0.90, "慢速": 0.88, "缓慢": 0.85, "温柔": 0.92,
    "轻声": 0.95, "低语": 0.90,
}
_SENT_SPLIT_RE = re.compile(r"([^。！？!?…～~]+[。！？!?…～~]*)")
# 动作/说明提示（全宽/半宽括号与〔〕）—— 既不展示、也不喂给 TTS
_ACT_RE = re.compile(r"〔[^〕]*〕|（[^）]*）|\([^)]*\)|【[^】]*】")


def _norm_style(style):
    s = (style or "").strip()
    for k in _KNOWN_STYLES:
        if k in s:
            return k
    return None


def _parse_rate(s):
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


def _split(text, max_n=8):
    segs = [x.strip() for x in _SENT_SPLIT_RE.findall(text or "") if x and x.strip()]
    if not segs and (text or "").strip():
        segs = [text.strip()]
    if len(segs) > max_n:
        segs = segs[:max_n - 1] + ["".join(segs[max_n - 1:])]
    return segs


def parse_markup(text):
    """解析 [风格]正文(语速)，返回 [(正文, 风格orNone, 语速), ...]。"""
    raw = (text or "").strip()
    out, pos, n = [], 0, len(raw)
    while True:
        m = re.search(r"\[([^\[\]]+)\]", raw[pos:])
        if m is None:
            rest = _ACT_RE.sub("", raw[pos:]).strip()
            for s in _split(rest):
                if s.strip():
                    out.append((s.strip(), None, 1.0))
            break
        head = raw[pos:pos + m.start()].strip()
        if head:
            for s in _split(head):
                s = _ACT_RE.sub("", s).strip()
                if s.strip():
                    out.append((s.strip(), None, 1.0))
        style = m.group(1).strip()
        body_start = pos + m.end()
        nxt = raw.find("[", body_start)
        body_end = n if nxt < 0 else nxt
        body = raw[body_start:body_end].strip()
        rate = 1.0
        rm = re.search(r"\(([^()]*)\)\s*$", body)
        if rm:
            rate = _parse_rate(rm.group(1))
            body = body[:rm.start()].strip()
        if body:
            out.append((_ACT_RE.sub("", body).strip(), _norm_style(style), rate))
        pos = body_end if nxt < 0 else nxt
    if not out:
        out = [(raw or " ", None, 1.0)]
    return out


# ---------------------------------------------------------------- LLM（文本引擎）
def llm_online(timeout=3.0):
    try:
        req = urllib.request.Request(TEXT_BASE.rstrip("/") + "/health")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read().decode("utf-8"))
            return j.get("status") == "ok"
    except Exception:
        return False


def _llm_env():
    env = dict(os.environ)
    llamacpp_pkg = PYKITS / "llamacpp"
    llamacpp_lib = llamacpp_pkg / "llama_cpp" / "lib"
    torch_lib = Path(
        r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib")
    env["PATH"] = str(llamacpp_lib) + os.pathsep + str(torch_lib) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + str(llamacpp_pkg) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def start_llm(log):
    """拉起 04 文本引擎（后台子进程）。返回 success / already_running。"""
    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    log("正在启动 04 文本引擎…")
    cmd = [str(py), str(BACKEND_DIR / "server.py"),
           "--host", "127.0.0.1", "--port", str(int(TEXT_BASE.rsplit(":", 1)[-1]))]
    return subprocess.Popen(cmd, cwd=str(BACKEND_DIR), env=_llm_env(),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def llm_chat(messages, log=None, host="127.0.0.1"):
    """POST /chat 到 04 文本引擎，返回清洗后的回复文本。"""
    pay = {"messages": messages, "max_new": 160, "temperature": 0.9,
           "top_p": 0.9, "top_k": 50, "role": "default", "seed": None}
    req = urllib.request.Request(
        TEXT_BASE.rstrip("/") + "/chat",
        data=json.dumps(pay).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        j = json.loads(r.read().decode("utf-8"))
    return (j.get("reply") or "").strip()


# ---------------------------------------------------------------- TTS（外挂情感 GGUF）
def synth_segment(text, style):
    """复用外挂情感 GGUF 方案合成单句，返回 (float32 wav, sr, meta)。"""
    gguf = GGUFTTS.get()
    emotion = style or "平静"
    return gguf.synthesize(text, emotion=emotion)


def concat_wavs(wavs, sr):
    import numpy as np
    gap = np.zeros(int(sr * 0.08), dtype="float32")
    pieces = []
    for i, w in enumerate(wavs):
        pieces.append(w)
        if i < len(wavs) - 1:
            pieces.append(gap)
    return np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")


def write_wav_bytes(arr, sr):
    import numpy as np
    import wave, io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        pcm = (np.clip(arr.astype("float32"), -1, 1) * 32767).astype(np.int16)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------- 桌面 App
import tkinter as tk
from tkinter import scrolledtext, ttk
import winsound

DBL = "\u2022"


class YuanyuanApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("缘圆 · 桌面对话台（LLM + 外挂情感 TTS）")
        root.geometry("680x640")
        root.minsize(560, 500)
        style = ttk.Style()
        style.theme_use("clam")

        self.msgs = []   # 会话历史 [{"role","content"}]
        self.llm_proc = None
        self.synth_lock = threading.Lock()
        self.stop_play = threading.Event()

        top = ttk.Frame(root); top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="LLM:").pack(side="left")
        self.lbl_llm = ttk.Label(top, text="检测中…", foreground="#e67e22")
        self.lbl_llm.pack(side="left", padx=(2, 12))
        ttk.Label(top, text="TTS(外挂GGUF):").pack(side="left")
        self.lbl_tts = ttk.Label(top, text="检查…", foreground="#e67e22")
        self.lbl_tts.pack(side="left", padx=(2, 12))
        self.btn_wanttts = tk.IntVar(value=1)
        tk.Checkbutton(top, text="语音回复", variable=self.btn_wanttts, bg="#eceff1",
                       font=("Microsoft YaHei", 9)).pack(side="left", padx=(4, 10))
        ttk.Label(top, text="情感:").pack(side="left")
        self.emo_var = tk.StringVar(value="auto")
        self.emo_cb = ttk.Combobox(top, textvariable=self.emo_var, state="readonly",
                                   width=8, values=["auto"] + _KNOWN_STYLES)
        self.emo_cb.pack(side="left")

        self.chat = scrolledtext.ScrolledText(root, wrap="word",
                                              font=("Microsoft YaHei", 11),
                                              bg="#fbfcfe", fg="#263238",
                                              state="disabled", relief="flat", borderwidth=1)
        self.chat.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        mid = ttk.Frame(root); mid.pack(fill="x", padx=8)
        self.entry = ttk.Entry(mid, font=("Microsoft YaHei", 11))
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.btn_send = ttk.Button(mid, text="发送", command=self.on_send)
        self.btn_send.pack(side="left", padx=(6, 0), ipadx=10)

        bar = ttk.Frame(root); bar.pack(fill="x", padx=8, pady=(4, 8))
        self.lbl_status = ttk.Label(bar, text="就绪", foreground="#333")
        self.lbl_status.pack(side="left")
        ttk.Button(bar, text="停止播放", command=self.stop_play.set).pack(side="right")
        ttk.Button(bar, text="清空对话", command=self.clear_chat).pack(side="right", padx=6)

        root.bind("<Return>", lambda e: self.on_send())
        self.append_hint("输入消息后回车发送。首次 TTS 需加载模型，略慢。")

        threading.Thread(target=self.init_backends, daemon=True).start()

    # ---------------- 初始化后端 ----------------
    def init_backends(self):
        # LLM：在线则复用，否则拉起
        if llm_online():
            self.safe_set(self.lbl_llm, "在线", "#27ae60")
        elif AUTO_LLM:
            try:
                self.llm_proc = start_llm(self.log)
                deadline = time.time() + 120
                while time.time() < deadline:
                    if llm_online():
                        self.safe_set(self.lbl_llm, "在线(已拉起)", "#27ae60")
                        break
                    time.sleep(2)
                else:
                    self.safe_set(self.lbl_llm, "连接失败", "#e74c3c")
                    self.log("LLM 启动超时，请手动启动 04 文本引擎")
            except Exception as e:
                self.safe_set(self.lbl_llm, "启动异常", "#e74c3c")
                self.log(f"LLM 启动异常: {e}")
        else:
            self.safe_set(self.lbl_llm, "离线", "#e74c3c")

        # TTS（外挂情感 GGUF）
        try:
            gguf = GGUFTTS.get()
            a = gguf.availability()
            if a.get("available"):
                self.safe_set(self.lbl_tts, "就绪(voice+emotion LoRA)", "#27ae60")
                self.log("TTS 外挂情感方案就绪")
            else:
                self.safe_set(self.lbl_tts, "不可用", "#e74c3c")
                self.log(f"TTS 缺失: {a.get('reason')}")
        except Exception as e:
            self.safe_set(self.lbl_tts, "异常", "#e74c3c")
            self.log(f"TTS 初始化异常: {e}")

    # ---------------- 发送 ----------------
    def on_send(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.msgs.append({"role": "user", "content": text})
        self.append_chat("你", text, "#2980b9")
        self.lbl_status.config(text="对方正在输入…")
        threading.Thread(target=self.work, args=(text,), daemon=True).start()

    def work(self, user_text):
        history = self.msgs[-8:-1]
        try:
            messages = [{"role": "system", "content": PERSONA}]
            messages += history
            messages.append({"role": "user", "content": user_text})
            reply = llm_chat(messages, self.log)
        except Exception as e:
            self.lbl_status.config(text=f"LLM 出错: {e}")
            self.append_chat("系统", f"文本引擎错误：{e}", "#e74c3c")
            return
        self.msgs.append({"role": "assistant", "content": reply})
        display = _ACT_RE.sub("", reply).strip()
        display = re.sub(r"\[[^\[\]]*\]|\([^()]*\)", "", display)
        self.safe_set(self.lbl_status, "就绪")
        self.append_chat("缘圆", display or reply, "#27ae60")

        if self.btn_wanttts.get():
            self.synthesize_and_play(reply)

    # ---------------- 外挂情感 TTS ----------------
    def synthesize_and_play(self, reply):
        with self.synth_lock:
            self.stop_play.clear()
            segs = parse_markup(reply)
            tmpdir = tempfile.mkdtemp(prefix="yy_tts_")
            wavs, srs = [], []
            try:
                for i, (seg_text, style, rate) in enumerate(segs):
                    if self.stop_play.is_set():
                        break
                    emo = style or ("平静" if self.emo_var.get() == "auto" else self.emo_var.get())
                    self.safe_set(self.lbl_status, f"合成第{i+1}/{len(segs)}句 [{emo}]…")
                    wav, sr, meta = synth_segment(seg_text, emo)
                    wavs.append(wav); srs.append(sr)
                if not wavs:
                    return
                full = concat_wavs(wavs, srs[0])
                p = Path(tmpdir) / "full.wav"
                p.write_bytes(write_wav_bytes(full, srs[0]))
                self.safe_set(self.lbl_status, "播放中…（可点停止）")
                winsound.PlaySound(str(p), winsound.SND_FILENAME)
            except Exception as e:
                self.safe_set(self.lbl_status, "TTS 出错")
                self.append_chat("系统", f"TTS 合成失败：{e}", "#e74c3c")
            finally:
                self.safe_set(self.lbl_status, "就绪")

    # ---------------- UI helpers ----------------
    def append_chat(self, name, text, color):
        def _do():
            self.chat.config(state="normal")
            self.chat.insert("end", f"{name}：", ("name",))
            self.chat.insert("end", text + "\n\n")
            self.chat.tag_config("name", foreground=color, font=("Microsoft YaHei", 11, "bold"))
            self.chat.see("end")
            self.chat.config(state="disabled")
        self.root.after(0, _do)

    def append_hint(self, text):
        def _do():
            self.chat.config(state="normal")
            self.chat.insert("end", "· " + text + "\n\n", ("hint",))
            self.chat.tag_config("hint", foreground="#90a4ae", font=("Microsoft YaHei", 9, "italic"))
            self.chat.config(state="disabled")
        self.root.after(0, _do)

    def log(self, text):
        self.append_hint(text)

    def safe_set(self, widget, text, color=None):
        def _do():
            widget.config(text=text)
            if color:
                widget.config(foreground=color)
        self.root.after(0, _do)

    def clear_chat(self):
        self.msgs.clear()
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self.append_hint("对话已清空")

    def on_close(self):
        if self.llm_proc is not None:
            try:
                self.llm_proc.terminate()
            except Exception:
                pass
        self.root.destroy()


def main():
    global TEXT_BASE, AUTO_LLM
    ap = argparse.ArgumentParser(description="缘圆桌面对话台")
    ap.add_argument("--text-base", default="http://127.0.0.1:8000")
    ap.add_argument("--no-auto-llm", action="store_true", help="不自动拉起 04 文本引擎")
    a = ap.parse_args()
    TEXT_BASE = a.text_base
    AUTO_LLM = not a.no_auto_llm

    root = tk.Tk()
    app = YuanyuanApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()