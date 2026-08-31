# -*- coding: utf-8 -*-
"""完整音频链路测试：对话 → 情感自动识别 → TTS，多情感场景产出可试听语音。

- 走 /api/pipeline/talk 完整闭环（04 文本引擎生成回复 + LLM 情感识别 + TTS 合成）
- 默认 backend=tf（情感外挂，情感精度最高）；可 --backend gguf 对比快速后端
- 落盘 serve/out/test_audio/<idx>_<情感>.wav + summary.json（含回复/情感/时长/文件）
"""
import argparse
import json
import os
import time
import urllib.request

BASE = "http://127.0.0.1:8070/api/pipeline/talk"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "test_audio")
SYSTEM = "你是萌系女主播缘圆，说话俏皮温柔爱撒娇，粉丝叫你圆圆，像个真实萌系女主播一样自然回应。"

CASES = [
    ("开心", "圆圆你今天也太好看了吧，气质绝了，我爱死你了！"),
    ("悲伤", "圆圆，我今天被领导批评了，好委屈，想哭。"),
    ("温柔", "圆圆，夜深了，你早点休息，别太累。"),
    ("激动", "圆圆，我们破十万粉了！大家都在刷屏庆祝！"),
    ("俏皮", "圆圆，我不喜欢你这款啦，我喜欢隔壁小雨～"),
    ("撒娇", "圆圆，你就再唱一首嘛，人家特意来听你的。"),
    ("平静", "圆圆，今天直播几点开始呀？"),
]


def post(user, backend):
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "max_new": 40, "auto_emotion": True, "backend": backend}
    req = urllib.request.Request(
        BASE, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        wav = r.read()
        return json.loads(r.headers.get("X-Pipe") or "{}"), wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="tf", choices=["tf", "gguf"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for i, (tag, user) in enumerate(CASES):
        t0 = time.perf_counter()
        try:
            pipe, wav = post(user, a.backend)
            emo = pipe.get("emotion") or "?"
            fn = os.path.join(OUT, f"{i:02d}_{emo}.wav")
            open(fn, "wb").write(wav)
            tt = pipe.get("tts", {})
            wall = round(time.perf_counter() - t0, 1)
            summary.append({"idx": i, "scenario": tag, "emotion": emo,
                            "source": pipe.get("source"), "reply": pipe.get("reply"),
                            "wall_s": wall, "audio_s": tt.get("audio_seconds"),
                            "file": fn, "bytes": len(wav)})
            print(f"[{i}] {tag} -> {emo}({pipe.get('source')}) "
                  f"aud={tt.get('audio_seconds')}s wall={wall}s file={fn}", flush=True)
        except Exception as e:
            print(f"[{i}] {tag} FAILED: {e}", flush=True)
            summary.append({"idx": i, "scenario": tag, "error": str(e)})
    with open(os.path.join(OUT, f"summary_{a.backend}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"== done backend={a.backend} ok={len([s for s in summary if 'file' in s])}/{len(CASES)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()