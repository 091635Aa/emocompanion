# -*- coding: utf-8 -*-
"""
缘圆 audio_codes 编码（稳健版，支持 detached 运行）
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSTokenizer
    tok = Qwen3TTSTokenizer.from_pretrained(args.tok, device_map="cuda:0")

    lines = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    f = open(args.output, "w", encoding="utf-8")
    t0 = time.time()
    n = skipped = 0
    for x in lines:
        try:
            enc = tok.encode([x["audio"]])
            x["audio_codes"] = enc.audio_codes[0].cpu().tolist()
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
        except Exception as e:
            skipped += 1
            print(f"[skip] {x['audio']}: {str(e)[:80]}", flush=True)
        if n % 25 == 0:
            print(f"[{n}/{len(lines)}] {round(time.time()-t0,1)}s skip{skipped}", flush=True)
    f.close()
    print(f"[done] {n}/{len(lines)} skip{skipped} in {round(time.time()-t0,1)}s", flush=True)


if __name__ == "__main__":
    main()