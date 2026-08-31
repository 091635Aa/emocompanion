# -*- coding: utf-8 -*-
"""R6 离线回归：calibration.metrics 纯度量（speak_rate/edit_similarity/composite/_clamp）
运行：python3 /workspace/_regression/test_metrics_pure.py
"""
import sys

SERVE = "/workspace/EmoCompanion_角色挂载与情感注入工程/06_Qwen3TTS外挂/serve"
sys.path.insert(0, SERVE)

from calibration.metrics import speak_rate, edit_similarity, composite, _clamp

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}  {detail}")

print("== speak_rate ==")
check("正常语速", speak_rate(2.0, 10) == 5.0, f"got={speak_rate(2.0,10)}")
check("0秒=0", speak_rate(0, 10) == 0.0)

print("== edit_similarity ==")
check("完全相同=1", edit_similarity("你好世界", "你好世界") == 1.0)
check("空串=0", edit_similarity("", "abc") == 0.0)
s = edit_similarity("你好世界", "你好世界呀")
check("多一字 相似<1", 0 < s < 1, f"s={s}")

print("== _clamp / composite ==")
check("_clamp 下限", _clamp(-5, 0, 1) == 0)
check("_clamp 上限", _clamp(9, 0, 1) == 1)
c = composite(
    gen_pros={"duration_s": 2.0, "f0_mean": 200, "f0_std": 50, "energy_std": 0.3},
    gt_pros={"duration_s": 2.0, "f0_mean": 200, "f0_std": 50, "energy_std": 0.3},
    gen_nchars=4,
    asr_sim=1.0, gen_emb=None, gt_emb=None,
)
check("composite 返回 dict 带 parts", isinstance(c, dict) and "parts" in c, f"type={type(c)}")
check("自相似 composite 高", c["composite"] > 0.9, f"S={c['composite']}")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)