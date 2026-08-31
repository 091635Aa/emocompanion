# -*- coding: utf-8 -*-
"""风格 LoRA 自动断续训练控制器（桌面快捷方式入口）

设计目标：
  - 不触碰任何运行中的进程（RVC / 一体化对话台 / 文本引擎）。
  - 检测 GPU 可用显存：空闲(TOTAL-FREE >= 阈值)才启动风格 LoRA 训练；
    GPU 被占用时静默等待，等有空闲=自动断续推进，无需手动停服。
  - 训练完成自动转 GGUF 到 pykits/models/style_lora_qwen3tts.gguf，供
    tts_gguf 第3 adapter 叠加（--lora voice,emotion,style）。

方式：
  双击桌面快捷方式 -> 运行本脚本 -> 循环【检测GPU -> 训练 or 等待】。

用法:
  python style_lora_autodrive.py [--free-mb 9216] [--interval 60]
                                  [--out-dir ../out/style_lora]
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # 06_Qwen3TTS外挂
DATA = os.path.join(ROOT, "data")
FINETUNE = HERE
SCRIPTS = os.path.join(ROOT, "scripts")
# 固定用系统 Python310（含 torch/qwen_tts/peft/transformers，全精度训练所需）
PY = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"

# 训练用全精度 Base（transformers 路径，modelscope 已缓存）
INIT_MODEL = r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\master"
TRAIN_JSONL = os.path.join(DATA, "style_train_codes.jsonl")
GGUF_OUT = r"D:\AI情感\pykits\models\style_lora_qwen3tts.gguf"

TRAIN_EPOCHS = 4
TRAIN_R = 24
TRAIN_ALPHA = 48


def gpu_free_mb():
    """读取 nvidia-smi memory.free (MiB)。失败返回 -1（视为不可训练）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        line = out.stdout.strip().splitlines()
        if not line:
            return -1
        free_str = line[0].split(",")[0].strip()
        return int(free_str)
    except Exception:
        return -1


def run_train(out_dir):
    """启动风格 LoRA 训练（阻塞直到完成）。"""
    cmd = [PY, os.path.join(FINETUNE, "sft_style_lora.py"),
           "--init_model_path", INIT_MODEL,
           "--output_model_path", out_dir,
           "--train_jsonl", TRAIN_JSONL,
           "--num_epochs", str(TRAIN_EPOCHS),
           "--lora_r", str(TRAIN_R),
           "--lora_alpha", str(TRAIN_ALPHA)]
    print(f"[train] 启动: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=FINETUNE)


def convert_gguf(out_dir):
    """找训练 checkpoint 目录并转 GGUF。"""
    ckpt = None
    for name in sorted(os.listdir(out_dir), reverse=True):
        if name.startswith("style_checkpoint-epoch-"):
            ckpt = os.path.join(out_dir, name)
            break
    if not ckpt:
        print("[gguf] 未找到训练 checkpoint，跳过转换", flush=True)
        return False
    cmd = [PY, os.path.join(SCRIPTS, "hf_lora_to_tts_gguf.py"),
           "--lora-dir", ckpt, "--out", GGUF_OUT, "--f16"]
    print(f"[gguf] 转换: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=SCRIPTS)
    if rc == 0:
        print(f"[gguf] OK -> {GGUF_OUT}", flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-mb", type=int, default=9216,
                    help="GPU 空闲显存阈值(MiB)，低于则等待（RVC/服务占用时不抢）")
    ap.add_argument("--interval", type=int, default=60, help="等待轮询秒数")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out", "style_lora"))
    args = ap.parse_args()

    if not os.path.isfile(TRAIN_JSONL):
        print(f"[err] 无风格语料: {TRAIN_JSONL}，请先运行 build_style_train_jsonl", flush=True)
        return 1
    if not os.path.isdir(INIT_MODEL):
        print(f"[err] Base 模型缺失: {INIT_MODEL}", flush=True)
        return 1

    print("=== 风格 LoRA 自动断续训练（不动 RVC/服务）===", flush=True)
    print(f"  训练语料: {TRAIN_JSONL}", flush=True)
    print(f"  GPU 空闲阈值: {args.free_mb} MiB，轮询: {args.interval}s", flush=True)
    print(f"  输出: {args.out_dir}\n", flush=True)

    already_gguf = os.path.isfile(GGUF_OUT)
    if already_gguf:
        print(f"[info] GGUF 已存在: {GGUF_OUT}（如需重训请先删除）", flush=True)

    trained_ok = False
    while True:
        free = gpu_free_mb()
        if free < 0:
            print(f"[wait] 无法读取 GPU，{args.interval}s 后重试", flush=True)
        elif free >= args.free_mb:
            print(f"[gpu] 空闲显存 {free} MiB >= {args.free_mb}，开始训练", flush=True)
            rc = run_train(args.out_dir)
            if rc == 0:
                trained_ok = True
                print("[train] 训练完成", flush=True)
            else:
                print(f"[train] 训练退出码={rc}，等待后续重训", flush=True)
                time.sleep(args.interval)
                continue
            convert_gguf(args.out_dir)
            print("\n=== 全部完成。关闭本窗口即可。 ===", flush=True)
            return 0
        else:
            print(f"[wait] GPU 忙（空闲 {free}/{args.free_mb} MiB)，{args.interval}s 后重试；"
                  f"RVC/服务不受影响", flush=True)
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())