# -*- coding: utf-8 -*-
"""推理框架速度基准 —— 三后端统一口径对照
模式:
  bnb4   transformers + BitsAndBytes 4bit  (当前生产栈 / 基线)
  fp16   transformers + 原生 fp16          (16GB 机器对照, 不上 8GB)
  gguf   llama-cpp-python CUDA + Q4_K_M GGUF (低设备目标)
测量: prefill 时间 / decode 时间 / decode tok/s / 总 tok/s / 峰值显存
同一 prompt(人设+用户问句) 同一 max_new, warmup 1 + 3 次取中位
用法: python bench_framework.py [mode] [model_key]
"""
import os, sys, json, time, gc, statistics
import torch
import numpy as np

HERE = os.path.dirname(__file__)
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
MODEL_KEY = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-4B"

MODELS = {
    "Qwen3-4B":  r"d:\AI情感\模型空间\Qwen3-4B",
    "DeepSeek-7B": r"d:\AI情感\模型空间\DeepSeek-R1-Distill-Qwen-7B",
}
# GGUF 显式路径（优先于目录扫描）
GGUF_PATHS = {
    "Qwen3-4B": r"d:\AI情感\pykits\models\Qwen3-4B-Q4_K_M.gguf",
    "DeepSeek-7B": None,
}

PERSONA = ("你是'EmoCompanion'，一个温柔、爱撒娇、爱黏人、口语化的直播情感主播。你的真实说话习惯："
           "常带'呀''嘛''啦''呗'（如'呀~''可以吗''嘛？'）；口头禅有'我去''哎呀''家人们''有感觉吗'；"
           "热情招呼观众点关注、加粉丝灯牌；感谢时用'谢谢、爱你、想你了'；自称'新人主播'。"
           "情绪随观众起伏（紧张、开心、撒娇、害羞），像真人聊天，绝不书面化/翻译腔/AI腔/透露是AI。\n"
           "示例：观众'欢迎来到我的直播'→EmoCompanion'欢迎来到EmoCompanion的直播间呀~喜欢EmoCompanion的可以点点关注、加个粉丝灯牌嘛？'\n"
           "示例：观众'今天紧张吗'→EmoCompanion'哎呀~有点紧张呢，我才是开播第五天的新人主播，但看到你来我就开心啦！'")

USER = "晚上好呀，今天直播好多人来，我好开心，你呢？"
MAX_NEW = 64
WARMUP, RUNS = 1, 3
SEED = 21

def strip_thinking(text):
    if "\n response\n\n" in text:
        text = text.split("\n response\n\n", 1)[-1]
    else:
        t = text.lstrip()
        if t.startswith(" thinking") or t.startswith("thinking"):
            text = ""
    return text.strip()

def chat_text(tok, persona=PERSONA, user=USER):
    msgs = [{"role": "system", "content": persona}, {"role": "user", "content": user}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return s

def bench_transf(path, fourbit):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    kw = {"torch_dtype": torch.float16, "device_map": "cuda"}
    if fourbit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    model.eval()
    load_s = time.time() - t0
    s = chat_text(tok)
    inp = tok(s, return_tensors="pt")["input_ids"].cuda()
    n_pre = inp.shape[1]
    genkw = dict(do_sample=True, temperature=0.9, top_p=0.9, top_k=50,
                 max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                 eos_token_id=tok.eos_token_id, use_cache=True)
    # warmup
    with torch.no_grad():
        model.generate(inp, **genkw)
    pre_times, dec_times = [], []
    with torch.no_grad():
        for _ in range(RUNS):
            t1 = time.time()
            out = model.generate(inp, **genkw)
            t2 = time.time()
            # prefill 单独计时
            t_p = time.time()
            model(inp, use_cache=True)
            torch.cuda.synchronize()
            t_p = time.time() - t_p
            pre_times.append(t_p)
            total = t2 - t1
            dec_times.append(max(total - t_p, 1e-6))
    vram = torch.cuda.max_memory_allocated() / (1024**3)
    del model; gc.collect(); torch.cuda.empty_cache()
    pt = statistics.median(pre_times); dt = statistics.median(dec_times)
    return {"load_s": round(load_s, 1), "prefill_s": round(pt, 3), "decode_s": round(dt, 3),
            "decode_tok_s": round(MAX_NEW / dt, 1), "total_tok_s": round(MAX_NEW / (pt + dt), 1),
            "peak_vram_GB": round(vram, 2), "n_prefill": n_pre}

def bench_gguf(path, gguf):
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(model_path=gguf, n_ctx=2048, n_gpu_layers=-1, n_threads=8,
                verbose=False, use_mmap=True, use_mlock=False)
    load_s = time.time() - t0
    msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": USER}]
    # warmup
    llm.create_chat_completion(messages=msgs, max_tokens=MAX_NEW, temperature=0.9, top_p=0.9)
    pre_times, dec_times = [], []
    for _ in range(RUNS):
        t1 = time.time()
        llm.create_chat_completion(messages=msgs, max_tokens=MAX_NEW, temperature=0.9, top_p=0.9)
        t2 = time.time()
        total = t2 - t1
        # llama.cpp 不暴露 prefill 分时, 用 tokens_evaluated 与 tokens_predicted 粗分:
        # 近似 prefill = total * n_prefill/(n_prefill+MAX_NEW)
        dec_times.append(total)   # 保守: 总时间都算 decode(上限)
    dt = statistics.median(dec_times)
    vram = _gpu_vram()
    return {"load_s": round(load_s, 1), "prefill_s": None, "decode_s": round(dt, 3),
            "decode_tok_s": round(MAX_NEW / dt, 1), "total_tok_s": round(MAX_NEW / dt, 1),
            "peak_vram_GB": round(vram, 2), "n_prefill": None}

def _gpu_vram():
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().split("\n")[0]) / 1024
    except Exception:
        return None

def main():
    path = MODELS[MODEL_KEY]
    out = {"model": MODEL_KEY, "path": path, "max_new": MAX_NEW, "runs": RUNS, "bench": {}}
    modes = ["bnb4", "fp16", "gguf"] if MODE == "all" else [MODE]
    for m in modes:
        try:
            if m == "bnb4":
                out["bench"]["bnb4_基线"] = bench_transf(path, True)
            elif m == "fp16":
                out["bench"]["fp16_原生"] = bench_transf(path, False)
            elif m == "gguf":
                # 自动查找同目录 gguf；若 GGUF_PATHS 显式指定则优先
                cands = []
                if GGUF_PATHS.get(MODEL_KEY):
                    cands = [GGUF_PATHS[MODEL_KEY]]
                if not cands:
                    for root, _, files in os.walk(path):
                        for f in files:
                            if f.endswith(".gguf"):
                                cands.append(os.path.join(root, f))
                if not cands:
                    print(f"[gguf] {MODEL_KEY} 无 .gguf, 跳过"); continue
                out["bench"]["gguf_CUDA"] = bench_gguf(path, cands[0])
            print(f"  [{m}] {out['bench'].get(list(out['bench'])[-1])}", flush=True)
        except Exception as e:
            print(f"  [{m}] 失败: {repr(e)[:200]}", flush=True)
            out["bench"][m] = {"error": repr(e)[:200]}
    fn = os.path.join(HERE, f"bench_framework_{MODEL_KEY.replace('-','_')}.json")
    json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nsaved: {fn}")

if __name__ == "__main__":
    main()
