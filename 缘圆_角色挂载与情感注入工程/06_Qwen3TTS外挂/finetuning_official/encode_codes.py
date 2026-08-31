# -*- coding: utf-8 -*-
"""
缘圆 audio_codes 稳健批量编码器（绕过 OOM/超长音频卡死）
====================================================================
- batch 小（默认 2），避免超长音频（>50s）在 batch 增大时耗尽显存
- 每批一个 try/except：单条失败跳过并记录，不阻塞整体
- 边 encode 边 flush 输出
- 可由已有输出续转（--resume）
用法：
  python encode_codes.py --tokenizer_model_path <tok> --input in.jsonl --output out.jsonl [--resume]
"""
import argparse
import json
import time
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_model_path", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSTokenizer
    tok = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_model_path, device_map="cuda:0")

    lines = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    done_audio = set()
    if args.resume and os.path.exists(args.output):
        done_audio = {json.loads(l)["audio"] for l in open(args.output, encoding="utf-8")}
        print(f"[resume] 已有 {len(done_audio)} 条", flush=True)

    f = open(args.output, "a" if args.resume else "w", encoding="utf-8")
    skip = processed = failed = 0
    B = args.batch
    t0 = time.time()
    for i in range(0, len(lines), B):
        b = lines[i:i + B]
        b = [x for x in b if x["audio"] not in done_audio]
        if not b:
            continue
        try:
            enc = tok.encode([x["audio"] for x in b])
            for code, x in zip(enc.audio_codes, b):
                x["audio_codes"] = code.cpu().tolist()
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                processed += 1
            f.flush()
        except Exception as e:
            failed += 1
            if "out of memory" in str(e).lower():
                # batch 太大 → 单条逐个试
                f.flush()
                for x in b:
                    try:
                        enc1 = tok.encode([x["audio"]])
                        x["audio_codes"] = enc1.audio_codes[0].cpu().tolist()
                        f.write(json.dumps(x, ensure_ascii=False) + "\n")
                        f.flush()
                        processed += 1
                    except Exception as e2:
                        failed += 1
                        skip += 1
                        print(f"[skip] {x['audio']}: {str(e2)[:80]}", flush=True)
            else:
                skip += len(b)
                print(f"[skip-batch] {b[0]['audio']}...: {str(e)[:80]}", flush=True)
        if processed % 40 < B:
            print(f"[{processed}/{len(lines)} 耗时{round(time.time()-t0,1)}s "
                  f"skip{skip} fail{failed}]", flush=True)
    f.close()
    print(f"[done] processed={processed}/{len(lines)} skip={skip} fail={failed}", flush=True)


if __name__ == "__main__":
    main()