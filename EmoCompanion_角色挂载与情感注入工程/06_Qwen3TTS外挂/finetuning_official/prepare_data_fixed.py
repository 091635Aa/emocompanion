# -*- coding: utf-8 -*-
"""
EmoCompanion prepare_data 修复版
====================================================================
官方 prepare_data.py 的 BATCH_INFER_NUM=32 在 16GB 显存会 OOM（每条 44s 音频
≈593帧×16 codebook，32 条一次性编码需 8GB+）。本版：
  - batch 改为可配置（默认 4），逐批 encode 并**边编码边写临时结果**，
    避免一次性 OOM 和进程卡死
  - 边处理边把结果 append 到输出文件（进度可恢复）
用法：
  python prepare_data_fixed.py --device cuda:0
      --tokenizer_model_path <tok> --input_jsonl in --output_jsonl out
      --batch 4  [--resume]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--tokenizer_model_path", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--batch", type=int, default=4, help="编码批次大小（避免 OOM）")
    ap.add_argument("--resume", action="store_true", help="跳过已写出的行数（断点续转）")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSTokenizer

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path, device_map=args.device)

    lines = [json.loads(l.strip()) for l in open(args.input_jsonl, encoding="utf-8")
             if l.strip()]
    total = len(lines)

    # 断点：若已生成了前 n 行，跳过前 n 条
    done = set()
    resume_from = 0
    if args.resume and os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, encoding="utf-8") as f:
            resume_from = sum(1 for _ in f)
        print(f"[resume] 已存在 {resume_from} 行，续转剩余 {total - resume_from}", flush=True)
        done = {json.loads(l)["audio"] for l in open(args.output_jsonl, encoding="utf-8")}

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    out_handle = open(args.output_jsonl, "a", encoding="utf-8") if args.resume \
        else open(args.output_jsonl, "w", encoding="utf-8")

    batch_lines, batch_audios = [], []
    processed = resume_from

    def flush_batch():
        nonlocal batch_lines, batch_audios, processed
        if not batch_audios:
            return
        try:
            enc_res = tokenizer.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line['audio_codes'] = code.cpu().tolist()
                out_handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            out_handle.flush()
            processed += len(batch_lines)
            print(f"[{processed}/{total}]", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM at {processed}, batch={len(batch_audios)}] too large; "
                      f"skip this batch（提示：减小 --batch）", flush=True)
            else:
                print(f"[error] {e}", flush=True)
        batch_lines.clear()
        batch_audios.clear()

    for line in lines:
        audio = line['audio']
        if audio in done:
            continue  # 已写出，跳过
        batch_lines.append(line)
        batch_audios.append(audio)
        if len(batch_lines) >= args.batch:
            flush_batch()

    flush_batch()  # 尾部不足 batch
    out_handle.close()
    print(f"[done] 共 {processed}/{total} 条 -> {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()