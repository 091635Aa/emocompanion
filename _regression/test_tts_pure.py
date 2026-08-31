# -*- coding: utf-8 -*-
"""R3 离线回归：TTS 纯逻辑（无 GPU、无 torch、无 llama.exe）
覆盖：tts_gguf.resolve_style/compose_params/shape_for_style/strip_control_tokens
      emo_detect._lexicon_detect/_extract_text/detect_emotion
运行：python3 /workspace/_regression/test_tts_pure.py
"""
import sys, os, types

SERVE = "/workspace/EmoCompanion_角色挂载与情感注入工程/06_Qwen3TTS外挂/serve"
sys.path.insert(0, SERVE)

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


# tts_engine stub：emo_detect 顶部 `from tts_engine import EMOTION_VOCAB`
te = types.ModuleType("tts_engine")
te.EMOTION_VOCAB = {"开心": None, "俏皮": None, "悲伤": None, "平静": None, "兴奋": None, "撒娇": None}
sys.modules["tts_engine"] = te

# 屏蔽 tts_gguf 模块级 numpy（真实 numpy 已在沙箱装好，仅保守 stub 重定向避免版本行为差异）
import numpy as np  # noqa: E402

import tts_gguf  # noqa: E402
import emo_detect  # noqa: E402

print("== tts_gguf.resolve_style ==")
check("未知风格回退", tts_gguf.resolve_style("不存在的风格xyz") in (None, "", "default") or isinstance(tts_gguf.resolve_style("不存在的风格xyz"), str))
st = tts_gguf.resolve_style("慵懒")
check("慵懒 可解析", st is not None, f"got={st}")

print("== tts_gguf.compose_params ==")
cp = tts_gguf.compose_params("开心", "慵懒")
check("compose_params 返回 6 元组", isinstance(cp, tuple) and len(cp) == 6, f"type={type(cp)}, len={len(cp) if isinstance(cp, tuple) else 0}")

print("== tts_gguf.shape_for_style / strip_control_tokens ==")
sp = tts_gguf.shape_for_style("你好，我好。", "慵懒")
check("shape_for_style 非 None", sp is not None, f"got={sp}")
clean = tts_gguf.strip_control_tokens("哈哈[开心]你真棒(快速)〔笑〕")
check("strip_control_tokens 去控制符", "[" not in clean and "(" not in clean and "〔" not in clean, f"clean={clean!r}")

print("== emo_detect._lexicon_detect ==")
r = emo_detect._lexicon_detect("我好难过呜呜")
check("难过词命中", r.label == "悲伤", f"got={r.label}")
r2 = emo_detect._lexicon_detect("嘻嘻，人家好开心")
check("开心词命中", r2.label == "开心", f"got={r2.label}")

print("== emo_detect._extract_text（兼容 OpenAI / 04 引擎两套协议）==")
a = emo_detect._extract_text({"choices": [{"message": {"content": "OK"}}]})
check("OpenAI 协议", a == "OK", f"got={a!r}")
b = emo_detect._extract_text({"role": "assistant", "reply": "你好呀"})
check("04引擎协议", b == "你好呀", f"got={b!r}")

print("== emo_detect.detect_emotion（词典分支，无 LLM）==")
d = emo_detect.detect_emotion("我真的好悲伤，呜呜呜")
check("detect 词典分支 返回悲伤", d.label == "悲伤", f"got={d.label}")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)
