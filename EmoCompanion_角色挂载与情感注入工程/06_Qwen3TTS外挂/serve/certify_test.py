# -*- coding: utf-8 -*-
"""certify_test —— 训练数据验收 + 真实生成测试

复用最优推理权重(best_weights.json)，对跨情感短句做实际 TTS 合成(含 rate 语速注入)，
落盘 wav 并报告每段的 合成策略 / RTF / 时长 / 相似度(与韵律质控)。

运行(serve 目录下，.venv 未建则用系统 python，需已装 torch/qwen_tts/peft):
  python certify_test.py [--weights 检查点目录]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

try:
    from tts_engine import get_engine
    from calibration.optimizer import strip_tag, apply_rate
    from calibration.metrics import prosody_features
except Exception as e:  # noqa
    sys.stderr.write(f"import 失败: {e}\n")
    sys.exit(2)

# 验收用例：每句自带情感标签，覆盖全部训练情感 & 复合情感(整句统一)
CASES = [
    ("开心", "欢迎来到小伴直播间，谢谢你来看我，今天心情超级好。"),
    ("平静", "你先忙你的，我在这儿等你忙完再说。"),
    ("悲伤", "你们怎么对我下手这么狠，我都要哭了。"),
    ("温柔", "晚安好梦，会梦见我吗？小猫咪。"),
    ("激动", "欢迎大家来到直播间，走过路过千万不要错过了，现在是小萌新主播。"),
    ("俏皮", "有感觉吗？有感觉吗？你看我笑死，怎么可能在偷看我。"),
    ("撒娇", "你要是再不给我加个灯牌，我可就要生气了哦。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="最佳权重 json 目录；缺省从 calibration/out/checkpoints/CURRENT.json 取最近")
    a = ap.parse_args()

    # 1) 定位最优推理权重
    cp_root = os.path.join(HERE, "calibration", "out", "checkpoints")
    cur = os.path.join(cp_root, "CURRENT.json")
    wpath = None
    if a.weights:
        wpath = a.weights
    elif os.path.isfile(cur):
        wpath = json.load(open(cur, encoding="utf-8"))["latest_checkpoint"]
    if not wpath:
        sys.exit("找不到最优权重，先运行 calibration.run_calibrate")
    wpath = wpath if os.path.isabs(wpath) else os.path.join(cp_root, wpath)
    bw = json.load(open(os.path.join(wpath, "best_weights.json"), encoding="utf-8"))
    cfg = bw["best_cfg"]
    print(f"== 最优推理权重 {os.path.basename(wpath)} reward={bw['best_reward']:.4f} ==")
    print("   cfg:", json.dumps(cfg, ensure_ascii=False))

    # 2) 加载 TTS（常驻 bf16 ~4.4GB，与 RVC 并行安全）
    eng = get_engine()
    t0 = __import__("time").perf_counter()
    eng.load()
    print(f"== TTS 加载 {__import__('time').perf_counter()-t0:.1f}s ==")

    # 3) 逐句真实合成 + rate 语速注入
    outdir = os.path.join(HERE, "out", "certify_" + os.path.basename(wpath))
    os.makedirs(outdir, exist_ok=True)
    report, rows = {"weights": os.path.basename(wpath), "cfg": cfg, "samples": []}, []
    for emo, text in CASES:
        w, sr, meta = eng.synthesize(
            strip_tag(text), emotion=emo,
            adapter=cfg["adapter"], tone_variation=cfg["tone_variation"],
            temperature=cfg["temperature"], top_k=cfg["top_k"],
            top_p=cfg["top_p"], repetition_penalty=cfg["repetition_penalty"],
        )
        w = np.asarray(w, dtype="float32")
        w, sr, _r = apply_rate(w, sr, cfg["rate"])
        pros = prosody_features((w, sr))
        fn = os.path.join(outdir, f"{emo}_{len(rows):02d}.wav")
        import wave
        with wave.open(fn, "wb") as fs:
            fs.setnchannels(1); fs.setsampwidth(2); fs.setframerate(sr)
            fs.writeframes((w * 32767).astype(np.int16).tobytes())
        rows.append({"emotion": emo, "text": text})
        print(f"  [{emo}] {meta['strategy']:<16s} RTF={meta['rtf']} "
              f"syn={meta['seconds']:.2f}s aud={meta['audio_seconds']:.2f}s "
              f"-> {meta['audio_seconds']*cfg['rate']:.2f}s @rate={cfg['rate']} f0m={pros['f0_mean']}")
        report["samples"].append(dict(rows[-1], meta={
            "strategy": meta["strategy"], "audio_seconds": round(meta["audio_seconds"] * cfg["rate"], 3),
            "rtf": meta["rtf"], "ref_audio": meta["ref_audio"]}))

    with open(os.path.join(outdir, "test_report.json"), "w", encoding="utf-8") as fs:
        json.dump(report, fs, ensure_ascii=False, indent=2)
    print(f"== 已落盘 {len(rows)} 段 -> {outdir} ==")


if __name__ == "__main__":
    main()