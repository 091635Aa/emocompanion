# -*- coding: utf-8 -*-
"""通过运行中的 unified_server 用 tf 后端生成 7 情感正规样片（复用已加载引擎，安全）。
用新补的 serve/refs/<emotion>.wav —— 验证策略是否从 voice_clone_emb 切到 ref_icl。
"""
import json, os, urllib.request, wave, numpy as np, time

BASE = "http://127.0.0.1:8070"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "certify_bestcfg_refs")
os.makedirs(OUT, exist_ok=True)

CASES = [
    ("开心", "欢迎来到圆圆直播间，谢谢你来看我，今天心情超级好。"),
    ("平静", "你先忙你的，我在这儿等你忙完再说。"),
    ("悲伤", "你们怎么对我下手这么狠，我都要哭了。"),
    ("温柔", "晚安好梦，会梦见我吗？小猫咪。"),
    ("激动", "欢迎大家来到直播间，走过路过千万不要错过了，现在是小萌新主播。"),
    ("俏皮", "有感觉吗？有感觉吗？你看我笑死，怎么可能在偷看我。"),
    ("撒娇", "你要是再不给我加个灯牌，我可就要生气了哦。"),
]
summary = {"samples": []}
for emo, text in CASES:
    body = json.dumps({"text": text, "emotion": emo, "backend": "tf"}).encode("utf-8")
    req = urllib.request.Request(BASE + "/api/tts/synthesize", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        wavdata = resp.read()
        meta = json.loads(resp.headers["X-TTS-Meta"])
    sr = meta.get("sr") or 24000
    a = np.frombuffer(wavdata, dtype=np.int16).astype("float32") / 32767.0
    fn = os.path.join(OUT, f"{emo}.wav")
    with wave.open(fn, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((a * 32767).astype(np.int16).tobytes())
    row = {"emotion": emo, "strategy": meta.get("strategy"),
           "audio_seconds": round(meta.get("audio_seconds", a.shape[0] / sr), 2),
           "seconds": meta.get("seconds"), "rtf": meta.get("rtf"),
           "ref_audio": meta.get("ref_audio"), "file": fn}
    summary["samples"].append(row)
    print(f"[{emo}] strategy={meta.get('strategy')} aud={row['audio_seconds']}s "
          f"syn={row['seconds']}s rtf={meta.get('rtf')} ref={os.path.basename(str(meta.get('ref_audio')))}")
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("DONE ->", OUT)