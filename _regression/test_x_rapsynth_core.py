# -*- coding: utf-8 -*-
"""R14 音频族回归：Qwen3-RapSynth TTS 基座封装（synthesizer）纯逻辑层

验证（不依赖 GPU / qwen_tts，直接 import 真实模块）：
  1. EMOTION_VOCAB 结构：键数与情感标签唯一性、`[emotion]X[/emotion]` 围栏自洽。
  2. EMOTION_FALLBACK 必须落在词表内（兜底可解析）。
  3. 情感→前缀解析：已知情感命中词条；未知情感回退兜底；
     前缀与文本拼接及 strip 语义等价于 synthesize 首行逻辑。
  4. TTSUnavailable 为 RuntimeError 子类（结构化异常契约）。
  5. _check() 缺件检测：任一组件缺失即抛 TTSUnavailable 且信息含组件名；
     全部就绪时不再抛。
运行：python3 /workspace/_regression/test_x_rapsynth_core.py
"""
import os, sys, tempfile, importlib.util, re

根 = "/workspace/_regression"
PASS = 0
FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")

# ---- 引入真实模块（顶层仅依赖 stdlib）----
mod_path = (根 + "/../EmoCompanion_角色挂载与情感注入工程/"
            "07_说唱合成Qwen3RapSynth/tts/synthesizer.py")
spec = importlib.util.spec_from_file_location("rapsynth_core_test", mod_path)
synth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth)

VOCAB = synth.EMOTION_VOCAB
FB = synth.EMOTION_FALLBACK

print("== EMOTION_VOCAB 结构 ==")
check("词表非空且≥6 情感", len(VOCAB) >= 6, f"n={len(VOCAB)}")
# 每个词条必须是 `[emotion]X[/emotion]` 且 X 与键同名
围栏自洽 = True
for k, v in VOCAB.items():
    exp = f"[emotion]{k}[/emotion]"
    if v != exp:
        围栏自洽 = False
        print(f"    key={k} got={v!r} exp={exp!r}")
check("情感围栏与键自洽", 围栏自洽)
# 键去重唯一
check("情感键唯一", len(set(VOCAB)) == len(VOCAB))

print("== 兜底与前缀解析 ==")
check("兜底情感在词表内", FB in VOCAB)
# 已知情感命中词条（复现 synthesize 的解析逻辑）
def 前缀解析(emotion):
    return VOCAB.get(emotion, VOCAB[FB])
for 情感 in list(VOCAB):
    check(f"已知情感解析[{情感}]", 前缀解析(情感) == VOCAB[情感])
# 未知情感回退兜底
check("未知情感回退兜底", 前缀解析("不存在的情感") == VOCAB[FB])
# 拼接语义：f"{前缀}{text}".strip()
txt = "  hello 说唱  "
拼接 = f"{前缀解析('开心')}{txt}".strip()
check("拼接 strip 语义", 拼接 == f"[emotion]开心[/emotion]{txt}".strip())

print("== TTSUnavailable 契约 ==")
check("TTSUnavailable 是 RuntimeError 子类", issubclass(synth.TTSUnavailable, RuntimeError))

print("== _check() 缺件检测 ==")
with tempfile.TemporaryDirectory() as td:
    ok_core = synth.RaSynthCore(
        base_model=td, adapter_dirs={"voice": td, "emotion": td},
        speaker_emb=os.path.join(td, "emb.pt"))
    open(ok_core.speaker_emb, "wb").close()  # 就绪态：embedding 文件真实存在
    ok_core._check()  # 全部就绪不应抛
    check("全部就绪 _check 不抛", True)

    bad_core = synth.RaSynthCore(
        base_model=os.path.join(td, "no_such_base"),
        adapter_dirs={"voice": td, "emotion": os.path.join(td, "no_emo")},
        speaker_emb=os.path.join(td, "no_emb.pt"))
    try:
        bad_core._check()
        check("缺件抛 TTSUnavailable", False, "未抛")
    except synth.TTSUnavailable as e:
        msg = str(e)
        check("缺件抛 TTSUnavailable", True)
        check("错误信息含全部缺件名",
              "Base:" in msg and "外挂[emotion]" in msg and "音色 embedding" in msg,
              f"msg={msg!r}")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)