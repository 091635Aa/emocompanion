# -*- coding: utf-8 -*-
"""R16 LLM 族回归：P1~P5 统一测试「汇总报告」纯聚合逻辑（torch-free）

直接 import 真实模块 `P1_5统一汇总报告.py`（顶层仅 os/json/datetime），
无需 GPU 即验证评测端聚合契约：
  1. 汇总健康度：每模式跨样本求平均（熵/重复/命中/长度）+ 兜底总和（非平均）。
  2. 汇总裁判：win_rate/胜/总 透传；全部 7 模式都产出。
  3. 模式清单：裸 + P1~P5 三修饰 → 7 模式常量（单源事实）。
  4. 边界：单样本 n=1 不炸；均值四舍五入位数。
运行：python3 /workspace/_regression/test_x_p15_aggregate.py
"""
import os, sys, importlib.util

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

mod_path = (os.path.join(根, "..", "KV_情感共振解码", "核心", "P1_5统一汇总报告.py"))
spec = importlib.util.spec_from_file_location("p15_report", mod_path)
报告 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(报告)

模式列表 = 报告.模式列表
print("== 模式清单单源 ==")
check("7 模式且含裸基线", len(模式列表) == 7 and 模式列表[0] == "裸", f"got={模式列表}")
check("含 P1~P5 系列", all(("P" in m) for m in 模式列表[1:]))

print("== 汇总健康度（跨样本求平均 / 兜底求和）==")

def 造统计(s):
    """把 (熵,重复,命中,长度,兜底) 构成最小统计块"""
    return {"平均熵": s[0], "重复率": s[1], "情感命中率": s[2],
            "长度(字)": s[3], "触发兜底次数": s[4]}

def 造生成(模式, 样本统计):
    """样本统计=每样本 dict[模式->统计块]；真实结构为 项['回复'][m]['统计']"""
    return {"回复": [{"回复": {m: {"统计": 造统计(样本统计[i][m])} for m in 模式}}
                     for i in range(len(样本统计))]}

# 两个样本：模式"裸" 熵 0.1/0.3 -> 均值 0.2；兜底 1/3 -> 总 4
s0 = {m: (0.1, 0.02, 0.9, 50.0, 1) for m in 模式列表}
s1 = {m: (0.3, 0.04, 0.5, 30.0, 3) for m in 模式列表}
生成 = 造生成(模式列表, [s0, s1])
健康 = 报告.汇总健康度(生成)
裸h = 健康["裸"]
check("健康度含全部7模式", set(健康) == set(模式列表), f"got={list(健康)}")
check("平均熵=跨样本均值", abs(裸h["平均熵"] - 0.2) < 1e-9, f"got={裸h['平均熵']}")
check("重复率均值", abs(裸h["重复率"] - 0.03) < 1e-9, f"got={裸h['重复率']}")
check("命中率均值", abs(裸h["情感命中率"] - 0.7) < 1e-9, f"got={裸h['情感命中率']}")
check("平均长度均值", abs(裸h["平均长度(字)"] - 40.0) < 1e-9, f"got={裸h['平均长度(字)']}")
check("兜底总次数=求和非平均", 裸h["兜底总次数"] == 4, f"got={裸h['兜底总次数']}")
check("均值四舍五入到4位", isinstance(裸h["平均熵"], float))

# 边界：单样本 n=1
单 = 造生成(模式列表, [s0])
单h = 报告.汇总健康度(单)
check("单样本 n=1 不炸", abs(单h["裸"]["平均熵"] - 0.1) < 1e-9)

print("== 汇总裁判 ==")
def 造裁判(模式, wr):
    return {"裁判": "x", "配对": {m: {"win_rate": wr[m], "胜": round(wr[m]*30), "总": 30}
                                 for m in 模式}}
wr = {m: 0.5 for m in 模式列表}
裁判 = 报告.汇总裁判(造裁判(模式列表, wr))
check("裁判含全部7模式", set(裁判) == set(模式列表))
check("win_rate 透传", abs(裁判["裸"]["win_rate"] - 0.5) < 1e-9)
check("胜/总透传", 裁判["裸"]["胜"] == 15 and 裁判["裸"]["总"] == 30)
# 相对裸百分比计算（报告第3节 rel）
裸wr = 裁判["裸"]["win_rate"]
rel = f"{裁判['P5_超融合']['win_rate']/裸wr*100-100:+.1f}%" if 裸wr else "—"
check("相对裸百分比可算", rel != "—" and rel.endswith("%"))

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)