# -*- coding: utf-8 -*-
"""手机端全流程 Web API 实测：建会话 → 对话(情感+TTS) → 取音频(局域网 IP 验证)"""
import io
import json
import sys
import time
import urllib.request
import wave

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8071"


def req(method, path, body=None, timeout=600):
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("== 1. health ==")
    print(req("GET", "/api/health"))

    print("\n== 2. 创建会话 ==")
    sess = req("POST", "/api/sessions", {"title": "手机全流程实测"})
    sid = sess["id"]
    print(sess)

    print("\n== 3. 对话 talk（文本→自动情感→TTS 合成）==")
    t0 = time.time()
    r = req("POST", f"/api/sessions/{sid}/talk",
            {"content": "你好呀EmoCompanion，今天有点累，想听你说说话", "want_tts": True})
    dt = time.time() - t0
    print(f"耗时 {dt:.1f}s")
    print("回复:", r.get("reply"))
    print("情感:", r.get("emotion"), r.get("emotion_info"))
    print("音频:", r.get("audio"))
    print("TTS 元信息:", json.dumps(r.get("tts_meta"), ensure_ascii=False)[:400])

    # 4) 下载音频验证（局域网 IP）
    audio_path = r.get("audio")
    if audio_path:
        print("\n== 4. 下载音频 ==")
        with urllib.request.urlopen(BASE + audio_path, timeout=30) as resp:
            wav = resp.read()
        sr = 0
        try:
            with wave.open(io.BytesIO(wav)) as w:
                sr = w.getframerate()
                dur = w.getnframes() / sr
        except Exception:
            dur = None
        print(f"音频 {len(wav)} bytes, {dur:.1f}s @{sr}Hz")

    print("\n== 5. 会话历史 ==")
    g = req("GET", f"/api/sessions/{sid}")
    print("消息数:", len(g["messages"]), "| 最后一条角色:", g["messages"][-1]["role"])
    print("完成 ✅")


if __name__ == "__main__":
    main()
