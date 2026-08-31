# -*- coding: utf-8 -*-
"""tts_tf_worker —— tf(transformers 情感外挂) 后端子进程 · 常驻版

常驻进程避免每次调用重新加载模型(~47s)，模型只加载一次，stdin JSON 逐条合成。

协议（stdin/stdout 逐行 JSON）:
  - 启动即加载模型，就绪后 stdout 打一行 `{"event":"ready","pid":...}`
  - 每行请求:  {"id": "...", "text": "...", "emotion": "开心",
                "adapter": "emotion", "tone": 0.35,
                "refs": [{"audio": "...", "text": "..."}, ...],   # 可选，多段 ICL
                "out_dir": "...", "prefix": "out"}
  - 每行响应:  {"id": "...", "ok": true, "wav": "...", "sr": 24000, "meta": {...}}
            或 {"id": "...", "ok": false, "error": "..."}
  合成成功后写出 <prefix>.wav 与 <prefix>.json{meta}。

用法(集成侧):
  p = subprocess.Popen([sys.executable, "tts_tf_worker.py"],
                       stdin=PIPE, stdout=PIPE, stderr=..., text=True, bufsize=1)
  等 p.stdout 读到 ready 行后，向 p.stdin 写一行 JSON，再读一行响应。
"""
import contextlib
import json
import os
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402


@contextlib.contextmanager
def _silence_stdout():
    """把模型/transformers 的打印(如 pad_token 警告、LoRA 权重名)重定向到 stderr，
    保证 stdout 只承载协议 JSON 行。"""
    old = sys.stdout
    try:
        sys.stdout = sys.stderr
        yield
    finally:
        sys.stdout = old


def _write_wav(arr, sr, out_wav):
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype("int16")
        w.writeframes(pcm.tobytes())


def _respond(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    from tts_engine import get_engine

    eng = get_engine()
    with _silence_stdout():
        eng.load()   # bf16 全精度；常驻 ~4.4GB 显存，需确保有空闲
    _respond({"event": "ready", "pid": os.getpid()})

    for line in sys.stdin:
        line = (line or "").strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            rid = req.get("id", "")
            text = req.get("text", "")
            emotion = req.get("emotion", "平静")
            adapter = req.get("adapter", "emotion")
            tone = float(req.get("tone", 0.35))
            refs = req.get("refs")   # 可选：list[dict{audio,text}]
            out_dir = req.get("out_dir", ".")
            prefix = req.get("prefix", "out")
            os.makedirs(out_dir, exist_ok=True)

            with _silence_stdout():
                wav, sr, meta = eng.synthesize(
                    text, emotion, adapter=adapter, tone_variation=tone, refs=refs)
            arr = np.asarray(wav, dtype="float32")
            out_wav = os.path.join(out_dir, prefix + ".wav")
            _write_wav(arr, sr, out_wav)
            meta["sr"] = int(sr)
            with open(os.path.join(out_dir, prefix + ".json"), "w",
                      encoding="utf-8") as f:
                json.dump({"sr": int(sr), "meta": meta}, f, ensure_ascii=False)
            _respond({"id": rid, "ok": True, "wav": out_wav, "sr": int(sr),
                      "meta": meta})
        except Exception as e:
            _respond({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
