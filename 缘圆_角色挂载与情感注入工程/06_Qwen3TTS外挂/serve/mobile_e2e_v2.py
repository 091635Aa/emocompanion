# -*- coding: utf-8 -*-
"""全流程实测 v2：自然语言风格标记 + 逐句风格/语速 + 整段合并音频修复"""
import io
import json
import sys
import time
import urllib.request
import wave

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8071"


def req(method, path, body=None, timeout=900):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wav_dur(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        wav = resp.read()
    with wave.open(io.BytesIO(wav)) as w:
        return round(w.getnframes() / w.getframerate(), 2)


print("== 1. talk（非流式）：风格标记 + 整段合并 ==")
sid = req("POST", "/api/sessions", {"title": "风格标记实测"})["id"]
t0 = time.time()
r = req("POST", f"/api/sessions/{sid}/talk",
        {"content": "我现在心情很复杂，既开心又有点舍不得", "want_tts": True})
dt = time.time() - t0
print(f"耗时 {dt:.1f}s")
print("回复(应为可见文本,无标记):", r.get("reply"))
print("情感:", r.get("emotion"), r.get("emotion_info"))
tm = r.get("tts_meta") or {}
print("TTS styles:", tm.get("styles"), "| rates:", tm.get("rates"),
      "| n_sentences:", tm.get("n_sentences"))
print("整段音频:", r.get("audio"))
if r.get("audio"):
    print("整段时长:", wav_dur(r["audio"]), "s")
for seg in (tm.get("segments") or []):
    print(f"  句{seg['index']}: style={seg['style']} rate={seg['rate']} "
          f"dur={wav_dur(seg['audio'])}s")
assert "[" not in (r.get("reply") or ""), "回复不应含风格标记!"

print("\n== 2. SSE 流式：逐句事件 + 整段音频 ==")
sid2 = req("POST", "/api/sessions", {"title": "SSE风格实测"})["id"]
payload = json.dumps({
    "content": "给我讲个开心的笑话吧，然后温柔地安慰我一下",
    "want_tts": True,
}).encode("utf-8")
rq = urllib.request.Request(f"{BASE}/api/sessions/{sid2}/stream", data=payload,
                            method="POST",
                            headers={"Content-Type": "application/json"})
t0 = time.time()
sentence_events, final = [], None
with urllib.request.urlopen(rq, timeout=900) as resp:
    buf = b""
    while True:
        chunk = resp.read(512)
        if not chunk:
            break
        buf += chunk
        text = buf.decode("utf-8", "replace")
        while "\n" in text:
            line, text = text.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except Exception:
                continue
            if ev.get("sentence_audio"):
                sentence_events.append(ev)
            elif ev.get("done"):
                final = ev
        buf = text.encode("utf-8")
print(f"耗时 {time.time()-t0:.1f}s | 逐句事件 {len(sentence_events)} 个")
print("最终 reply(无标记):", final.get("reply"))
print("最终 audio(整段):", final.get("audio"))
for e in sentence_events:
    print(f"  句{e['index']}: style={e.get('style')} rate={e.get('rate')} audio={e['audio']}")
if final and final.get("audio") and sentence_events:
    full = final["audio"]
    first = sentence_events[0]["audio"]
    print(f"\n整段文件 {full} == 第一句 {first} ? {full == first}")
    assert full != first, "BUG: 整段音频仍是第一句!"
    if len(sentence_events) > 1:
        full_d = wav_dur(full)
        sum_d = sum(wav_dur(e["audio"]) for e in sentence_events)
        print(f"整段时长 {full_d}s ≈ 各句之和 {sum_d}s")
print("\n完成 ✅")
