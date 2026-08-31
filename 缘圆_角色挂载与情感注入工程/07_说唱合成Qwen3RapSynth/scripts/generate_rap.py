# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · 一键生成 CLI：歌词 + 风格 + BPM → 说唱干声 wav

端到端管道：
  歌词/风格/BPM → ProsodyPlan(规则韵律预测器) → 逐行 TTS 中性合成
  → 间接注入(对拍 time-stretch + pitch + 能量) → 干声 wav + 逐行 onset + 后置评估

内置三档风格（快嘴 / 旋律说唱 / 硬核）示例歌词，对应任务书"3 段不同风格样例"。

模式：
  --no-tts      仅跑规则预测 + 对拍编排到"计划 JSON"，不触发 GPU 合成（框架校验用）
  --plan 文件   （预览）：打印 ProsodyPlan 不合成
  默认         全链路合成（需电源充足；首次会加载 Base+LoRA）

用法：
  python generate_rap.py --lyrics-file demo_lyrics/快嘴.txt --style 快嘴 --bpm 96
  python generate_rap.py --no-tts --style 旋律说唱 --bpm 84 --out output/demo_plans
"""
import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from prosody_model.rules import get_predictor, ProsodyPlan, LinePlan  # noqa: E402
from integration.injector import inject, SR  # noqa: E402

# 内置示例（三种风格各 4 行，末字成韵便于 rhyme_hit 演示）
EXAMPLES = {
    "快嘴": ["夜色之下把话说完不留半句", "节奏像是电流穿进每个脉络",
             "一口气贯到底分秒都不犹豫", "饶舌就是我的枪口对准利弊"],
    "旋律说唱": ["晚风穿过街巷把路灯点亮", "思绪像潮水轻轻推到胸腔",
                "唱一段心事送给月光", "让旋律在心底慢慢回响"],
    "硬核": ["麦克风吹过就抛出厚重节拍", "低音砸在鼓点把地板撑开",
            "每一句都像拳头落进尘埃", "我在这条路上不肯退开"],
}


def plan_to_json(plan: ProsodyPlan) -> dict:
    return {
        "bpm": plan.bpm, "style": plan.style, "beat_sec": plan.beat_sec,
        "total_seconds": round(plan.total_seconds(), 3),
        "lines": [{
            "index": lp.index, "text": lp.text, "syllables": lp.syllables,
            "start_sec": round(lp.start_sec, 3), "duration_sec": round(lp.duration_sec, 3),
            "mean_f0": round(lp.mean_f0, 1), "f0_style": lp.f0_style,
            "energy": lp.energy, "jump": lp.jump,
        } for lp in plan.lines],
    }


def build_neutral(lyrics, plan, tts=None):
    """调用 TTS 逐行中性合成；tts=None 且模式非纯规则时抛错提示。"""
    out = []
    for lp in plan.lines:
        if tts is None:
            out.append(None)
            continue
        wav, sr, _ = tts.synthesize(lp.text, emotion="硬核")
        out.append((wav, sr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lyrics", default=None, help="歌词（多行）；缺省取 --style 内置示例")
    ap.add_argument("--lyrics-file", default=None)
    ap.add_argument("--style", default="快嘴", choices=list(EXAMPLES.keys()))
    ap.add_argument("--bpm", type=float, default=96.0)
    ap.add_argument("--base-f0", type=float, default=180.0)
    ap.add_argument("--no-tts", action="store_true", help="跳过 GPU 合成，仅校验管道并出计划")
    ap.add_argument("--plan-only", action="store_true", help="只打印 ProsodyPlan")
    ap.add_argument("--out", default=os.path.join(_ROOT, "output"))
    args = ap.parse_args()

    if args.lyrics_file:
        with open(args.lyrics_file, "r", encoding="utf-8") as f:
            lyrics = f.read()
    else:
        lyrics = args.lyrics or "\n".join(EXAMPLES[args.style])

    plan = get_predictor().predict(lyrics, bpm=args.bpm, style=args.style,
                                   base_f0=args.base_f0)
    os.makedirs(args.out, exist_ok=True)

    if args.plan_only:
        print(json.dumps(plan_to_json(plan), ensure_ascii=False, indent=2))
        return

    # 1) 中性合成（可选开关）
    tts = None
    if not args.no_tts:
        from tts.synthesizer import RaSynthCore
        tts = RaSynthCore()
        tts.load()
    neutral = build_neutral(lyrics.splitlines(), plan, tts)

    # 2) 间接注入（对拍编排）
    wav, onsets = inject(plan, neutral, base_f0=args.base_f0)

    # 3) 写出
    base = f"{args.style}_{int(args.bpm)}bpm"
    if wav is not None:
        import soundfile as sf
        wav_path = os.path.join(args.out, f"{base}.wav")
        sf.write(wav_path, wav, SR)
        print(f"[ok] 干声已写: {wav_path}")
    json_path = os.path.join(args.out, f"{base}_plan.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"plan": plan_to_json(plan), "onsets": [round(o, 3) for o in onsets]},
                  f, ensure_ascii=False, indent=2)

    # 4) 客观指标（含纯规则模式下的对拍自检）
    from eval.metrics import summarize
    texts = [lp.text for lp in plan.lines]
    if wav is not None:
        m = summarize(plan, wav, onsets, texts)
        print("[eval]", json.dumps(m, ensure_ascii=False))
    else:
        print("[skip-eval] --no-tts 未合成，跳过客观指标（可用备用/中性 wav 注入）")


if __name__ == "__main__":
    main()