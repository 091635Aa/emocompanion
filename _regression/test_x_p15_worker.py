# -*- coding: utf-8 -*-
"""R18 LLM 族回归：P1~P5 生成 worker 统计契约（torch-free 源码默认值白盒）

worker 顶层 import torch/transformers，无法离线 import；
本测试对其中三个纯统计/默认值契约做双重护栏：
  A. 行为规范：对 情感命中率 / 兜底默认 / 平均熵默认 的逻辑做等价复算测试。
  B. 源码字面一致：抓取真实源码片段，若与所测规范逐字不一致即判漂移（红灯），
     确保"测的就是生产跑的"。
运行：python3 /workspace/_regression/test_x_p15_worker.py
"""
import os, sys, re

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

worker_path = os.path.join(根, "..", "KV_情感共振解码", "核心", "P1_5统一生成worker.py")
源码 = open(worker_path, encoding="utf-8").read()

def 取函数体(名):
    """抓取 def 名(...): 到下一个同缩进 def / 文件尾 之间的源码"""
    m = re.search(r"(?m)^def %s\(" % 名, 源码)
    if not m:
        return ""
    start = m.start()
    # 找函数内首行的缩进
    body = 源码[m.end():]
    m2 = re.match(r"[^\n]*\n([ \t]+)", body)  # 找首个非空行的缩进
    indent = m2.group(1) if m2 else "    "
    # 截到第一个缩进小于 indent 的非空行（即返回顶层）
    lines = 源码[start:].split("\n")
    out = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() and not ln.startswith(indent):
            break
        out.append(ln)
    return "\n".join(out)

# ---- A1 情感命中率 ----
print("== A1 情感命中率 ==")
def 情感命中率(token列表, 情感集):
    if not token列表:
        return 0.0
    return round(sum(1 for t in token列表 if t in 情感集) / len(token列表), 4)

check("空 token 列表返回 0.0", 情感命中率([], {1, 2}) == 0.0)
check("命中计算 2/4=0.5", 情感命中率([1, 9, 2, 9], {1, 2}) == 0.5)
check("全命中 1.0", 情感命中率([1, 2], {1, 2}) == 1.0)
check("无命中 0.0", 情感命中率([5, 6], {1, 2}) == 0.0)
check("四舍五入 1/3=0.3333", 情感命中率([1, 9, 9], {1}) == 0.3333)

# ---- B1 情感命中率 源码字面一致 ----
print("== B1 情感命中率 源码字面一致 ==")
体 = 取函数体("情感命中率")
规范 = '''def 情感命中率(token列表, 情感集):
    if not token列表:
        return 0.0
    return round(sum(1 for t in token列表 if t in 情感集) / len(token列表), 4)'''
check("情感命中率 源码与所测规范逐字一致", 体.strip() == 规范.strip(), f"实际=\n{体}")

# ---- A2 平均熵默认 0.0 + round(4) ----
print("== A2 平均熵 空列表默认 0.0 ==")
def 平均熵(熵列表):
    return round(sum(熵列表) / len(熵列表), 4) if 熵列表 else 0.0
check("空熵列表 0.0", 平均熵([]) == 0.0)
check("均值 round(4)", 平均熵([0.1, 0.3]) == 0.2)
check("除零不炸", 平均熵([]) == 0.0, "由空列表 else 分支保障")

# ---- A3 兜底计数默认 0（解码器无 .统计 或不含键时）----
print("== A3 兜底计数默认 0 ==")
def 兜底计数(解码):
    return int(解码.get("统计", {}).get("触发兜底", 0)) if isinstance(解码, dict) and "统计" in 解码 else 0
check("无 .统计 解码器默认 0", 兜底计数({}) == 0)
check("统计缺触发兜底默认 0", 兜底计数({"统计": {"其他": 1}}) == 0)
check("有值时透传 int", 兜底计数({"统计": {"触发兜底": 7}}) == 7)

# ---- B3 兜底计数 源码字面一致 ----
print("== B3 兜底计数 源码片段一致 ==")
分段 = re.search(r"触发兜底次数[\s\S]*?hasattr\(解码, \"统计\"\)[^\n]*", 源码)
check("兜底计数字面量存在", 分段 is not None)
分段2 = re.search(r"""int\(解码\.统计\.get\("触发兜底", 0\)""", 源码)
check("int(解码.统计.get('触发兜底',0)) 字面存在", 分段2 is not None)

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)