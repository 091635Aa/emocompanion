# -*- coding: utf-8 -*-
"""R8 离线回归：Qwen3RapSynth 韵律规则基线（纯规则，无学习模型/GPU）
覆盖：count_syllables / predict_lyrics（拍对齐、风格参数、BPM 钳制、
      未知风格回退、空行跳过、start_sec 单调、轮廓类型）
运行：python3 /workspace/_regression/test_x_rap_prosody.py
"""
import sys, os, math

DIR = "/workspace/EmoCompanion_角色挂载与情感注入工程/07_说唱合成Qwen3RapSynth"
sys.path.insert(0, DIR)

from prosody_model.rules import count_syllables, predict_lyrics, STYLES  # noqa: E402

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

print("== count_syllables ==")
check("中文+英文按字符计", count_syllables("我要唱好这个flow") == 10, f"got={count_syllables('我要唱好这个flow')}")
check("标点不计入", count_syllables("快，快！") == 2, f"got={count_syllables('快，快！')}")

print("== predict_lyrics 结构 ==")
lyrics = "第一行歌词\n第二行歌词\n\n第三行"
plan = predict_lyrics(lyrics, bpm=90, style="快嘴")
check("空行被跳过", len(plan.lines) == 3, f"got={len(plan.lines)}")
check("首行 index=0", plan.lines[0].index == 0)
check("start_sec 单调递增", all(plan.lines[i].start_sec < plan.lines[i+1].start_sec for i in range(len(plan.lines)-1)))

print("== 风格参数/轮廓 ==")
p_mel = predict_lyrics("A\nB\nC\nD", style="旋律说唱")
check("旋律说唱 arch 轮廓", all(l.f0_style == "arch" for l in p_mel.lines), f"got={[l.f0_style for l in p_mel.lines]}")
p_hard = predict_lyrics("A\nB", style="硬核")
check("硬核 punch 轮廓", all(l.f0_style == "punch" for l in p_hard.lines), f"got={[l.f0_style for l in p_hard.lines]}")
p_fast = predict_lyrics("A\nB", style="快嘴")
check("快嘴 flat 轮廓", all(l.f0_style == "flat" for l in p_fast.lines), f"got={[l.f0_style for l in p_fast.lines]}")

print("== 未知风格回退 + BPM 钳制 ==")
p_unk = predict_lyrics("回退", style="不存在")
check("未知风格回退快嘴", p_unk.style == "快嘴", f"got={p_unk.style}")
check("低 BPM 钳制", predict_lyrics("x", bpm=1).bpm > 0)
p_hi = predict_lyrics("x", bpm=600)
check("beat_sec>0", p_hi.beat_sec > 0)

print("== base_f0 / 相对能量 ==")
p_f0 = predict_lyrics("压轴", style="硬核", base_f0=200)
check("mean_f0 有限且>0", all(math.isfinite(l.mean_f0) and l.mean_f0 > 0 for l in p_f0.lines))

print("== 协议：get_predictor 无权重回退规则基线 ==")
import importlib
mod = importlib.import_module("prosody_model.rules")
plan2 = mod.get_predictor(weights_path="/nonexistent.pt").predict("测试", 90, "旋律说唱")
check("get_predictor 回退规则且走统一协议", plan2 is not None and len(plan2.lines) == 1, f"got={plan2}")
rpred = mod.get_predictor(weights_path="/nonexistent.pt")
check("规则回退 is_learned=False", rpred.is_learned() is False, f"got={rpred.is_learned()}")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)