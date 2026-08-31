# -*- coding: utf-8 -*-
"""R6 离线回归：calibration/optimizer 纯逻辑（strip_tag/sample_config/basic_space）
运行：python3 /workspace/_regression/test_opt_pure.py
"""
import sys, random

SERVE = "/workspace/EmoCompanion_角色挂载与情感注入工程/06_Qwen3TTS外挂/serve"
sys.path.insert(0, SERVE)

from calibration.optimizer import strip_tag, sample_config, basic_space, UNIFIED_SPACE, DEFAULT_DEV_K, DEFAULT_ROUNDS

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}  {detail}")

print("== strip_tag ==")
check("剥 emoji 标签 保留正文", strip_tag("[emotion]开心[/emotion]你好") == "开心你好", f"got={strip_tag('[emotion]开心[/emotion]你好')!r}")
check("空安全", strip_tag(None) == "")

print("== sample_config（tuple=区间, list=choice）==")
rng = random.Random(1)
cfg = sample_config(rng, UNIFIED_SPACE)
check("采样含全部键", all(k in cfg for k in UNIFIED_SPACE), f"keys={sorted(cfg)}")
check("tuple 键为数值", isinstance(cfg["temperature"], float), f"t={cfg['temperature']!r}")
check("list 键命中空间", cfg["adapter"] in UNIFIED_SPACE["adapter"], f"a={cfg['adapter']}")

print("== basic_space（网格基线）==")
base = basic_space(UNIFIED_SPACE)
check("网格产物为非空列表", len(base) > 0, f"n={len(base)}")
check("网格固定参数", all(c["repetition_penalty"] == 1.05 for c in base))
check("adapter 只在空间内", all(c["adapter"] in UNIFIED_SPACE["adapter"] for c in base))

print("== 常量健全性 ==")
check("rounds>0", DEFAULT_ROUNDS > 0)
check("dev_k>0", DEFAULT_DEV_K > 0)

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)