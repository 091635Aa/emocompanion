# -*- coding: utf-8 -*-
"""R5 离线回归：锚点解码器纯逻辑（经验熵/计算重复率）+ torch stub 导入绕过
运行：python3 /workspace/_regression/test_anchor_pure.py
"""
import sys, os, types, math

CORE = "/workspace/KV_情感共振解码/核心"
sys.path.insert(0, CORE)
os.environ.setdefault("EMOTION_REPO_ROOT", "")

# torch stub：仅需解析 import，不运行任何 torch 算子
_t = types.ModuleType("torch")
class _F:
    def softmax(s, *a, **k): raise NotImplementedError
    def log_softmax(s, *a, **k): raise NotImplementedError
_F = _F()
_t.nn = types.SimpleNamespace(functional=_F)
_t.functional = _F
_t.no_grad = lambda *a, **k: (lambda f: f)  # 兼容 @torch.no_grad() 装饰器工厂
_t.Tensor = type("Tensor", (object,), {})
sys.modules["torch"] = _t
sys.modules["torch.nn"] = _t.nn
sys.modules["torch.nn.functional"] = _F

import 锚点解码器 as A

PASS = 0
FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}  {detail}")

print("== 经验熵 ==")
check("空列表熵=0", A.经验熵([]) == 0.0)
check("单token熵=0", A.经验熵([5]) == 0.0)
h = A.经验熵([1, 1, 2, 2])
check("两分类熵=ln2", abs(h - math.log(2)) < 1e-9, f"h={h}")

print("== 计算重复率（2-gram）==")
check("短列表取0", A.计算重复率([1]) == 0.0)
r = A.计算重复率([1, 1, 1, 1])
check("全同2-gram重复率=0.6667", r == 0.6667, f"r={r}")
r2 = A.计算重复率([1, 2, 3, 4])
check("全唯一2-gram重复率=0", r2 == 0.0, f"r2={r2}")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)