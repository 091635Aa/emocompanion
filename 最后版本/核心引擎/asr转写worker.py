# -*- coding: utf-8 -*-
"""
独立 ASR 转写进程（Qwen2.5-Omni 全模态）

Qwen2.5-Omni-7B 需要 transformers 4.5x（本机主环境为 5.x，不兼容），
因此 ASR 转写由本 worker 在独立环境（.venv_asr）中执行：
    <venv_asr>/Scripts/python 核心引擎/asr转写worker.py <模型路径> <音频路径> <输出JSON> [段边界JSON]

用法:
    参数1: 模型路径（Qwen2.5-Omni 系列，需音频解码器）
    参数2: 音频路径（wav/mp3 等，16kHz 优先，自动重采样）
    参数3: 输出 JSON 路径（写入 {"成功", "文本", "分段", "总时长秒", "耗时秒"}）
    参数4(可选): 段边界 JSON 路径（list of {"起始秒","结束秒"}，缺省整段一次转写）

成功输出: {"成功": true, "文本": "...", "分段": [{"起始秒","结束秒","文本"}], "总时长秒": n, "耗时秒": n}
失败输出: {"成功": false, "错误": "..."}  （写入输出 JSON，退出码 1）
"""
import json
import os
import sys
import time

import torch

sys.stdout.reconfigure(encoding="utf-8")

转写指令 = "请将这段音频的内容逐字转写为文字，不要添加任何解释或润色。"


def 读取音频(音频路径):
    """读取音频为 float32 numpy（16kHz），非 16k 自动重采样。"""
    import numpy as np
    import soundfile as sf
    数组, 采样率 = sf.read(音频路径, dtype="float32")
    if len(数组.shape) > 1:  # 多声道取平均
        数组 = 数组.mean(axis=1)
    if 采样率 != 16000:
        try:
            import librosa
            数组 = librosa.resample(数组, orig_sr=采样率, target_sr=16000)
            采样率 = 16000
        except Exception:
            pass
    return 数组.astype(np.float32), 采样率


def 转写(模型对象, 处理器, 音频数组, 采样率, 起始秒, 结束秒):
    """转写一段音频，返回文本。"""
    print(f"[worker] 转写段 {起始秒}s-{结束秒}s 开始", flush=True)
    段 = 音频数组[int(起始秒 * 采样率): int(结束秒 * 采样率)]
    if len(段) < 采样率 * 0.5:  # 不足 0.5 秒
        return ""
    # 写临时 wav（对话用文件路径，与已验证的调用方式一致）
    import tempfile
    import soundfile as sf
    临时路径 = os.path.join(tempfile.gettempdir(), f"asr段_{int(time.time() * 1000000)}.wav")
    sf.write(临时路径, 段, 采样率)
    print(f"[worker] 临时wav 已写 {临时路径}", flush=True)
    对话 = [{"role": "user", "content": [
        {"type": "audio", "audio": 临时路径},
        {"type": "text", "text": 转写指令},
    ]}]
    文本 = 处理器.apply_chat_template(对话, add_generation_prompt=True, tokenize=False)
    print("[worker] 模板OK", flush=True)
    输入 = 处理器(text=文本, audio=[段], return_tensors="pt", sampling_rate=采样率)
    输入 = {k: v.to("cuda:0") for k, v in 输入.items() if hasattr(v, "to")}
    print("[worker] 输入OK keys:", list(输入.keys()), flush=True)
    输出 = 模型对象.generate(**输入, return_audio=False, thinker_max_new_tokens=128)
    print("[worker] 生成OK", flush=True)
    try:
        os.remove(临时路径)
    except OSError:
        pass
    序列 = getattr(输出, "sequences", 输出)
    全部 = 处理器.batch_decode(序列, skip_special_tokens=True)
    原始 = 全部[0] if 全部 else ""
    if "assistant" in 原始:
        转写结果 = 原始.split("assistant", 1)[-1].strip()
    else:
        转写结果 = 原始
    # 清理可能的角色残留
    for 前缀 in ("system\n", "user\n", "assistant\n"):
        if 转写结果.startswith(前缀):
            转写结果 = 转写结果[len(前缀):]
    return 转写结果.strip()


def 主():
    模型路径 = sys.argv[1]
    音频路径 = sys.argv[2]
    输出路径 = sys.argv[3]
    边界路径 = sys.argv[4] if len(sys.argv) > 4 else ""

    开始 = time.time()
    try:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        if not os.path.isfile(音频路径):
            raise RuntimeError(f"音频文件不存在：{音频路径}")

        音频数组, 采样率 = 读取音频(音频路径)
        总时长秒 = round(len(音频数组) / 采样率, 2)
        if 总时长秒 <= 0:
            raise RuntimeError("音频为空")

        # 段边界
        分段 = []
        if 边界路径 and os.path.isfile(边界路径):
            with open(边界路径, "r", encoding="utf-8") as f:
                try:
                    分段 = json.load(f)
                except Exception:
                    分段 = []
        if not 分段:
            分段 = [{"起始秒": 0, "结束秒": 总时长秒}]

        print(f"[worker] 加载处理器：{模型路径}", flush=True)
        处理器 = Qwen2_5OmniProcessor.from_pretrained(模型路径, trust_remote_code=True)
        print(f"[worker] 加载模型（7B 需 1-2 分钟）...", flush=True)
        模型对象 = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            模型路径, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        print(f"[worker] 模型就绪，显存 {round(torch.cuda.memory_allocated() / 2**30, 1)}GB", flush=True)

        结果分段 = []
        全部文本 = []
        for i, 段 in enumerate(分段, 1):
            起始 = float(段.get("起始秒", 0))
            结束 = float(段.get("结束秒", 总时长秒))
            文本 = 转写(模型对象, 处理器, 音频数组, 采样率, 起始, 结束)
            结果分段.append({"起始秒": 起始, "结束秒": 结束, "文本": 文本})
            全部文本.append(文本)
            print(f"[worker] 段 {i}/{len(分段)} 完成：{文本[:40]}...", flush=True)

        del 模型对象
        torch.cuda.empty_cache()

        结果 = {
            "成功": True,
            "文本": "\n".join(t for t in 全部文本 if t),
            "分段": 结果分段,
            "总时长秒": 总时长秒,
            "耗时秒": round(time.time() - 开始, 1),
        }
        with open(输出路径, "w", encoding="utf-8") as f:
            json.dump(结果, f, ensure_ascii=False, indent=2)
        print(json.dumps(结果, ensure_ascii=False), flush=True)
    except Exception as e:
        结果 = {"成功": False, "错误": f"{type(e).__name__}: {e}"}
        try:
            with open(输出路径, "w", encoding="utf-8") as f:
                json.dump(结果, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print(json.dumps(结果, ensure_ascii=False), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    主()
