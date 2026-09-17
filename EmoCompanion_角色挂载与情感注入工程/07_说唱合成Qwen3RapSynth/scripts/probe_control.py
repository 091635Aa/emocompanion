# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · Phase 0 控制面可行性探查

目的：以"源码审读 + 最小运行时探针"双路验证任务书假设
    "Qwen3-TTS 支持 pitch_curve / duration / 能量等底层声学参数控制"。

本脚本**不加载完整模型**（笔记本电源有限），通过：
  1) 静态检读安装包源码，枚举 generate 实际暴露的控制参数；
  2) 使用 inspect 读取基座包装类签名并打印；
  3) 轻量启动推理环境自检（CUDA 可用性、模型路径、可导入上述类）。

若需真实合成探针（--run 微合成），在电源充足时另行单独启动，
默认关闭以避免长时间占卡。

用法：
  python probe_control.py                # 静态 + 环境自检（快速）
  python probe_control.py --run "测试句"  # 额外跑一次极小合成验证（谨慎）
"""
import argparse
import inspect
import json
import os
import sys


BASE_MODEL = (
    r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\master"
)
TOKENIZER = (
    r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen3-TTS-Tokenizer-12Hz\snapshots\master"
)


def gen_sigs():
    """静态：枚举 Qwen3TTSModel 包装类顶层 API 参数（不 import 重模型）。"""
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
    out = {}
    for name, fn in [
        ("generate_voice_clone", Qwen3TTSModel.generate_voice_clone),
        ("generate_custom_voice", Qwen3TTSModel.generate_custom_voice),
        ("generate_voice_design", Qwen3TTSModel.generate_voice_design),
        ("create_voice_clone_prompt", Qwen3TTSModel.create_voice_clone_prompt),
    ]:
        try:
            s = str(inspect.signature(fn))
        except Exception as e:
            s = f"<err {e}>"
        out[name] = s
    out["VoiceClonePromptItem"] = str(
        [f.name for f in inspect.signature(VoiceClonePromptItem).parameters.values()][:8]
    )
    return out


def lowlevel_generate_sig():
    """更低一层：Qwen3TTSForConditionalGeneration.generate 暴露的控制参数。"""
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        s = str(inspect.signature(Qwen3TTSForConditionalGeneration.generate))
        return s
    except Exception as e:
        return f"<err {e}>"


def env_check():
    """运行时环境自检（不快照完整加载模型）。"""
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except Exception as e:
        info["torch"] = f"err {e}"
    info["base_model_dir"] = os.path.isdir(BASE_MODEL)
    info["tokenizer_dir"] = os.path.isdir(TOKENIZER)

    # 依赖存在性（vadeless 懒加载，仅探测是否可导入）
    for m in ("numpy", "librosa", "soundfile"):
        try:
            __import__(m)
            info[m] = True
        except Exception as e:
            info[m] = f"missing {e}"
    try:
        import peft
        info["peft"] = peft.__version__
    except Exception as e:
        info["peft"] = f"missing {e}"
    return info


def ref_audio_required():
    """代码审读结论：Base 的 generate_voice_clone 是否强制要求 ref_audio。"""
    src = inspect.getsource(
        __import__("qwen_tts.inference.qwen3_tts_model", fromlist=["x"]).Qwen3TTSModel.generate_voice_clone
    )
    return "Either `voice_clone_prompt` or `ref_audio` must be provided." in src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="非空则执行一次极小合成(需电源充足)")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--sigs-only", action="store_true", help="仅打印签名")
    args = ap.parse_args()

    result = {
        "static_wrapper_api": gen_sigs(),
        "static_lowlevel_generate": lowlevel_generate_sig(),
        "ref_audio_mandatory": ref_audio_required(),
        "task_hypothesis_explicit_pitch": False,
        "task_hypothesis_explicit_duration": False,
        "task_hypothesis_explicit_energy": False,
        "env": env_check(),
        "runtime_synthesis": None,
    }

    if args.run:
        # 真实合成探针：加载 Base(bf16)+speaker embedding 进行 x-vector 合成（不需要 ref_audio）
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from tts.synthesizer import RaSynthCore
        try:
            core = RaSynthCore()
            wav, sr, sec = core.synthesize(args.run)
            result["runtime_synthesis"] = {
                "sr": sr, "audio_seconds": round(sec, 3), "samples": int(len(wav)),
            }
        except Exception as e:
            result["runtime_synthesis"] = f"err {type(e).__name__}: {e}"

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()