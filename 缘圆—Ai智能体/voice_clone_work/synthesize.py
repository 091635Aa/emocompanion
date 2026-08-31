# -*- coding: utf-8 -*-
"""用已注册的复刻音色调用 qwen-audio-3.0-tts-plus 合成一段语音（带情感指令）。"""
import os, requests, json, pathlib, time, base64, subprocess

API_KEY = os.environ["DASHSCOPE_API_KEY"]
WORK = pathlib.Path(__file__).parent
VOICE_ID = (WORK / "voice_id.txt").read_text(encoding="utf-8").strip()
URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

TEXT = ("你好，我是缘圆。很高兴在这里遇见你。"
        "无论命运如何兜兜转转，我都愿意陪你把每一段路走完。")
INSTRUCTION = "用温柔、清澈、带一点俏皮的少女语气，自然地说话。"

def synth():
    payload = {
        "model": "qwen-audio-3.0-tts-plus",
        "input": {
            "text": TEXT,
            "voice": VOICE_ID,
            "format": "wav",
            "sample_rate": 48000,
            "instruction": INSTRUCTION,
        },
    }
    r = requests.post(URL, json=payload, headers=HEADERS, timeout=180)
    print("HTTP", r.status_code)
    body = r.text[:2000]
    print(body)
    if r.status_code != 200:
        return None
    j = r.json()
    audio = j.get("output", {}).get("audio", {}) or {}
    url = audio.get("url") or audio.get("data") or ""
    if url.startswith("http"):
        return requests.get(url, timeout=120).content
    if url:
        return base64.b64decode(url)
    return None

data = None
for attempt in range(1, 8):
    print(f"== attempt {attempt} ==")
    data = synth()
    if data:
        break
    time.sleep(15)

if not data:
    raise SystemExit("synthesis failed")

wav_out = WORK / "tts_output.wav"
mp3_out = WORK / "tts_output.mp3"
wav_out.write_bytes(data)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav_out),
                "-b:a", "192k", str(mp3_out)], check=True)
print("SAVED", wav_out, len(data), "bytes ->", mp3_out)
