# -*- coding: utf-8 -*-
"""一体化全流程·即时生成：经统一服务 GGUF 快速后端合成一组多情感音频。

输出 serve/out/pipeline_live/<idx>_<情感>.wav（供前端/回档试听）。
"""
import json
import os
import time
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "pipeline_live")
API = "http://127.0.0.1:8070/api/tts/synthesize"
CASES = [("开心", "见到你心情超级好，今天真是太开心啦！"),
         ("悲伤", "我有点难过，你能不能陪我一会儿嘛。"),
         ("温柔", "晚安好梦，会梦到我吗？"),
         ("激动", "哇塞，我们破十万粉了，简直太棒啦！"),
         ("俏皮", "你猜猜我给你藏了什么小惊喜呀？")]


def main():
    os.makedirs(OUT, exist_ok=True)
    for i, (emo, text) in enumerate(CASES):
        body = json.dumps({"text": text, "emotion": emo, "backend": "gguf"},
                          ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            wav = r.read()
            fn = os.path.join(OUT, "%02d_%s.wav" % (i, emo))
            open(fn, "wb").write(wav)
            print("[%d] %s: %dB %.0fs %s" % (i, emo, len(wav), time.perf_counter() - t0,
                                             r.headers.get("X-TTS-Meta", "")), flush=True)


if __name__ == "__main__":
    main()