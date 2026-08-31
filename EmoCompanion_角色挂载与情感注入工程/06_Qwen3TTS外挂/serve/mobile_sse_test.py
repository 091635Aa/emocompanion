# -*- coding: utf-8 -*-
"""SSE 流式接口实测（手机前端路径）"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8071"


def req(method, path, body=None, timeout=600):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


sess = req("POST", "/api/sessions", {"title": "SSE流式实测"})
sid = sess["id"]
print("会话:", sid)

payload = json.dumps({
    "content": "明天我要去面试了，有点紧张，给我点鼓励吧",
    "want_tts": True,
}).encode("utf-8")
r = urllib.request.Request(f"{BASE}/api/sessions/{sid}/stream", data=payload,
                           method="POST",
                           headers={"Content-Type": "application/json"})

t0 = time.time()
with urllib.request.urlopen(r, timeout=900) as resp:
    buf = b""
    events = 0
    text_ready = False
    audio = None
    while True:
        chunk = resp.read(512)
        if not chunk:
            break
        buf += chunk
        text = buf.decode("utf-8", errors="replace")
        # 按换行切事件
        while "\n" in text:
            line, text = text.split("\n", 1)
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            events += 1
            data = line[5:].strip()
            try:
                ev = json.loads(data)
            except Exception:
                continue
            if "delta" in ev:
                sys.stdout.write(ev["delta"])
                sys.stdout.flush()
            if ev.get("text_done"):
                text_ready = True
            if ev.get("audio"):
                audio = ev["audio"]
            if ev.get("done"):
                print("\n[done event] 情感:", ev.get("emotion"),
                      "| TTS:", ev.get("tts_meta", {}).get("backend"),
                      "| 音频:", audio)
        buf = text.encode("utf-8")

print(f"\n事件数={events} text_ready={text_ready} 耗时 {time.time()-t0:.1f}s")
print("SSE 流式 ✅")
