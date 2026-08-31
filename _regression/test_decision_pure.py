# -*- coding: utf-8 -*-
"""R3 离线回归：决策器/感知器纯逻辑（无 GPU、无 torch）
覆盖：目标决策器 匹配角色基调 / 架构族β因子 / VAD到锚点 / _指令到锚点 / _默认温柔 / 简易感知器
运行：python3 /workspace/_regression/test_decision_pure.py
"""
import sys, os, math

sys.path.insert(0, "/workspace/KV_情感共振解码/核心")

import numpy as np

# 移除硬编码盘符路径注入的干扰：目标决策器已支持 EMOTION_REPO_ROOT 覆盖（R2 已改）
os.environ.setdefault("EMOTION_REPO_ROOT", "")

import 目标决策器 as D

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


print("== 匹配角色基调 ==")
r = D.匹配角色基调("女友")
check("女友 命中角色名", r is not None and r != "", f"got={r}")
r2 = D.匹配角色基调("一个不存在的角色xyz")
check("未知角色 不命中", r2 is None or r2 == "", f"got={r2}")

print("== 架构族β因子 ==")
b = D.架构族β因子("Qwen2.5-3B")
check("Qwen2.5 有 β", b is not None and isinstance(b, tuple) and b[0] > 0, f"got={b}")
b0 = D.架构族β因子("完全不认识的模型名xyz")
check("未知架构 有兜底 β", b0 is not None and b0[0] > 0, f"got={b0}")

print("== 简易感知器（纯 stdlib）==")
S = D.简易感知器()
st, kw = S.测量("我今天很难过，好累啊")
check("测量返回状态", st is not None and hasattr(st, "valence"), f"got={st}")
check("难过为负效价", st.valence < 0, f"valence={getattr(st,'valence',None)}")

print("== 目标决策器 VAD→锚点 / 指令→锚点 / 默认温柔 ==")
dec = D.目标决策器(感知器=D.简易感知器(), 锚点库=None)
v = dec.VAD到锚点(D.简易情感状态(-0.6, 0.3, 0.2))
check("VAD到锚点 输出 K 维", v is not None and v.shape == (len(dec.维度),), f"shape={getattr(v,'shape',None)}")
vi = dec._指令到锚点("温柔陪伴")
check("指令到锚点 输出 K 维且归一化", vi.shape == (len(dec.维度),) and abs(float(np.linalg.norm(vi)) - 1) < 1e-6, f"norm={float(np.linalg.norm(vi)) if vi.size else 0}")
vd = D.目标决策器._默认温柔(len(dec.维度))
check("默认温柔 首维主导", vd[0] > 0 and vd[0] == float(vd.max()), f"vd0={vd[0]}, max={vd.max()}")

print("== 锚点目标字段（β 自适应限幅）==")
dec2 = D.目标决策器(感知器=D.简易感知器(), 锚点库=None, β基=0.8, β上限=2.0)
t = dec2.计算目标("我好难过啊", 指令="温柔陪伴")
check("计算目标 返回锚点目标", t is not None and hasattr(t, "v_target"), f"got={type(t)}")
beta = getattr(t, "β", None)
check("β 不越上限", beta is None or beta <= dec2.β上限 + 1e-9, f"β={beta}")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)
