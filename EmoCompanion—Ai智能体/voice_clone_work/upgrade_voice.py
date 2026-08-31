# -*- coding: utf-8 -*-
"""用全域重排后的最优片段(883.5-895.5s)重新注册音色并合成，供新旧对比。"""
import numpy as np, soundfile as sf, requests, base64, json, os, pathlib, subprocess

API_KEY = os.environ["DASHSCOPE_API_KEY"]
WORK = pathlib.Path(__file__).parent
SRC = WORK / "full_24k_mono.wav"
ENROLL_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
SYNTH_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
A, B = 883.5, 895.5
SR = 24000

# 1) 提取最优片段
x, sr = sf.read(SRC, dtype="float32")
seg = x[int(A * sr): int(B * sr)]
wav_v2 = WORK / "best_segment_v2.wav"
sf.write(wav_v2, seg, SR, subtype="PCM_16")
print(f"[1] 提取 {A}-{B}s -> {wav_v2} ({len(seg)/SR:.1f}s)")

# 2) base64 注册
data_uri = "data:audio/wav;base64," + base64.b64encode(wav_v2.read_bytes()).decode()
payload = {
    "model": "voice-enrollment",
    "input": {
        "action": "create_voice",
        "target_model": "qwen-audio-3.0-tts-plus",
        "prefix": "emocompanion",
        "language_hints": ["zh"],
        "url": data_uri,
    },
}
r = requests.post(ENROLL_URL, json=payload, headers=HEADERS, timeout=120)
print("[2] enroll HTTP", r.status_code)
j = r.json()
print(json.dumps(j, ensure_ascii=False)[:800])
voice_id = j.get("output", {}).get("voice_id")
if not voice_id:
    raise SystemExit("enroll failed")
(WORK / "voice_id_v2.txt").write_text(voice_id, encoding="utf-8")
print("   NEW VOICE_ID:", voice_id)

# 3) 合成（同一文本/指令，便于对比）
TEXT = ("你好，我是EmoCompanion。很高兴在这里遇见你。"
        "无论命运如何兜兜转转，我都愿意陪你把每一段路走完。")
INSTRUCTION = "用温柔、清澈、带一点俏皮的少女语气，自然地说话。"
r = requests.post(SYNTH_URL, json={
    "model": "qwen-audio-3.0-tts-plus",
    "input": {
        "text": TEXT, "voice": voice_id,
        "format": "wav", "sample_rate": 48000, "instruction": INSTRUCTION,
    },
}, headers=HEADERS, timeout=180)
print("[3] synth HTTP", r.status_code)
js = r.json()
url = js.get("output", {}).get("audio", {}).get("url", "")
if not url:
    print(json.dumps(js, ensure_ascii=False)[:800])
    raise SystemExit("synth failed")
data = requests.get(url, timeout=120).content
wav_out = WORK / "tts_output_v2.wav"
mp3_out = WORK / "tts_output_v2.mp3"
wav_out.write_bytes(data)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav_out),
                "-b:a", "192k", str(mp3_out)], check=True)
print("SAVED", mp3_out, f"({len(data)} bytes)")
