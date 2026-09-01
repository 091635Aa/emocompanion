# -*- coding: utf-8 -*-
"""R8 离线回归：多模态「单源不变量」漂移检测（零 import，纯静态源码分析）

目的：加固 R6 完成的 ETD/KV 单源收敛——防止「词典/模式名/AI腔词表」被重新复制
打散到多文件导致逻辑漂移。任何把下面任一"单源事实"破坏的改动都会在此红灯。

校验的单源事实（均为确定性文件级扫描）：
  1. 中文标签映射 只在 语音合成.py 定义一次（TTS 标签单源）。
  2. AI腔身份短语表（"我是一个AI"等）只在 base_decoding_controller.py 定义，
     不再以同集散落在其它 KV 解码器（防多条影子副本）。
  3. 接口降级三级模式名 '本地'/'logprobs'/'提示' 由 接口降级.py 的 判定接口 唯一产出。

运行：python3 /workspace/_regression/test_x_single_source_drift.py
"""
import os, re, sys

根 = "/workspace"
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

def 读(rel):
    p = os.path.join(根, rel)
    return open(p, encoding="utf-8").read() if os.path.isfile(p) else ""

# ---- 1) 中文标签映射 单源 ----
print("== TTS 中文标签映射 单源 ==")
语音合成 = os.path.join("EmoCompanion—Ai智能体/EmoCompanion智能体/核心模块", "语音合成.py")
映射定义处 = [语音合成] if "中文标签映射" in 读(语音合成) else []
# 全工作区扫描其它可能定义处
for dirpath, _, files in os.walk(根):
    if "_regression" in dirpath or ".venv" in dirpath or "__pycache__" in dirpath:
        continue
    for f in files:
        if f.endswith(".py"):
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, 根)
            if rel == 语音合成:
                continue
            try:
                src = open(full, encoding="utf-8").read()
            except Exception:
                continue
            if "中文标签映射 = " in src or "中文标签映射=" in src:
                映射定义处.append(rel)
check("中文标签映射仅在 语音合成.py 定义", 映射定义处 == [语音合成], f"got={映射定义处}")

# 语音合成内仅一个定义行
n_语音 = 读(语音合成).count("中文标签映射 = {")
check("语音合成中只定义一次", n_语音 == 1, f"got={n_语音}")

# ---- 2) AI腔身份键 底座齐全 + 重复定义受控（防新增影子副本）----
print("== AI腔身份键 底座齐全 + 重复定义受控 ==")
KV核心 = "KV_情感共振解码/核心"
base = 读(os.path.join(KV核心, "base_decoding_controller.py"))
# 底座身份键（从 base 的 AI腔词表 实测抽取）
身份键 = {"AI", "助手", "人工智能", "模型", "智能", "作为一个", "作为一名", "机器人", "程序"}
缺失 = [w for w in 身份键 if ("\"" + w + "\":") not in base and ("'" + w + "':") not in base]
check("base_decoding_controller 身份键齐全", not 缺失, f"missing={缺失}")

# 统计 KV_核心 内同时定义这些身份键的 .py 文件，必须限定在已知持有者集合内
def _含身份键(src, keys):
    return [k for k in keys if ("\"" + k + "\":" in src) or ("'" + k + "':" in src)]

持有者 = []
base_keys = _含身份键(base, 身份键)
for f in sorted(os.listdir(os.path.join(根, KV核心))):
    if not f.endswith(".py"):
        continue
    rel = os.path.join(KV核心, f)
    src = 读(rel)
    ks = _含身份键(src, 身份键)
    # 仅当与底座高度重叠（≥ 底座键的 80%）才视为"重复持有者"，防误报片段引用
    if len(ks) >= int(len(base_keys) * 0.8) and f != "base_decoding_controller.py":
        重叠率 = round(len(ks) / len(base_keys), 2)
        持有者.append((f, len(ks), 重叠率))
已知持有者 = {"情感导演解码器.py"}   # 独立实现，已知存在（扩展集更全，见 R9 报告）
check("身份键重复持有者为已知集合", {f for f, _, _ in 持有者} == 已知持有者, f"got={持有者}")
for f_, n_, o_ in 持有者:
    print(f"  [info] {f_} 与底座身份键重叠率={o_}")

# ---- 2b) 重叠身份键权重一致性（防"同名不同权"静默漂移）----
print("== 重叠身份键权重一致性 ==")
情感 = 读(os.path.join(KV核心, "情感导演解码器.py"))
# 身份键 → 期望权重（两处同语义，实测一致）
键权 = {"AI": 3.0, "助手": 3.0, "人工智能": 3.0, "模型": 3.0, "智能": 2.5,
        "作为一个": 2.5, "作为一名": 2.5, "机器人": 2.0, "程序": 2.0, "model": 2.5}
不符 = []
for k, w in 键权.items():
    a = ("\"" + k + "\": " + str(w)) in base or ("'" + k + "': " + str(w)) in base
    b = ("\"" + k + "\": " + str(w)) in 情感 or ("'" + k + "': " + str(w)) in 情感
    if not (a and b):
        不符.append((k, w, a, b))
check("重叠身份键权重两处一致", not 不符, f"不符={不符}")

# ---- 3) 接口降级模式名一致性 ----
print("== 接口降级模式名 ==")
id档 = 读(os.path.join(KV核心, "接口降级.py"))
定义函数 = r"def 判定接口\("
check("判定接口 存在", bool(re.search(定义函数, id档)))
# 三个模式字面量在定义源内都出现
for m in ("'本地'", "'logprobs'", "'提示'"):
    check(f"模式 {m} 在接口降级.py 内", m in id档, f"missing={m}")

# ---- 3b) 消费者对 接口 的比较字面量不得偏离规范三元组 ----
print("== 消费者 接口 比较字面量一致性 ==")
规范 = {"本地", "提示", "logprobs"}
违规 = []
for f in sorted(os.listdir(os.path.join(根, KV核心))):
    if not f.endswith(".py"):
        continue
    src = 读(os.path.join(KV核心, f))
    for m in re.finditer(r"接口\s*==\s*['\"]([^'\"]+)['\"]", src):
        val = m.group(1)
        if val not in 规范:
            违规.append((f, val))
check("接口比较字面量均在规范三元组内", not 违规, f"违规={违规}")

# ---- 3c) 生产者侧：模式名作为 return 值仅由 接口降级.py 唯一产出 ----
print("== 生产者 模式名 return 唯一产出 ==")
产出者 = []
for f in sorted(os.listdir(os.path.join(根, KV核心))):
    if not f.endswith(".py"):
        continue
    src = 读(os.path.join(KV核心, f))
    # 判定接口 用 `return '本地'` 而非变量赋值产出模式名
    for m in re.finditer(r"return\s+['\"](本地|logprobs|提示)['\"]", src):
        产出者.append((f, m.group(1)))
# 唯一产出者应为 接口降级.py（判定接口）；若出现其它文件 return 模式名即漂移
check("模式名 return 仅由 接口降级.py 产出",
      len(产出者) > 0 and all(f == "接口降级.py" for f, _ in 产出者),
      f"产出者={产出者}")

# ---- 3d) 接口= 关键字实参（构造传参）取值统一落在规范三元组 ----
print("== 接口= 实参取值一致性 ==")
接口实参 = []
for f in sorted(os.listdir(os.path.join(根, KV核心))):
    if not f.endswith(".py"):
        continue
    src = 读(os.path.join(KV核心, f))
    for m in re.finditer(r"接口\s*=\s*['\"]([^'\"]+)['\"]", src):
        # 排除语句起始的真赋值与 `self.接口 = 接口 or 判定接口(model)` 的表达式形态
        line = src[:m.start()].rfind("\n")
        prefix = src[line + 1:m.start()] if line != -1 else ""
        if prefix.strip().startswith("接口"):
            continue  # 赋值语句（真赋值）已由 3c/3b 覆盖
        接口实参.append((f, m.group(1)))
for f, v in 接口实参:
    if v not in 规范:
        违规.append(("实参", f, v))
check("接口= 实参均在规范三元组内",
      not [v for v in 违规 if v[0] == "实参"],
      f"实参={接口实参} 违规={[v for v in 违规 if v[0]=='实参']}")

# ---- 4) 跨族情感标签单源一致性（语音合成族 ↔ RapSynth/TTS族）----
print("== 跨族情感标签单源一致性 ==")
# 语音合成.py 的中文控制标签（中文->英文）作为可扫描源
语音文件 = os.path.join("EmoCompanion—Ai智能体/EmoCompanion智能体/核心模块", "语音合成.py")
语音src = 读(语音文件)
# 抽取 `("sad", "悲伤")` 风格的三元组 (英文, 中文)
控制对 = re.findall(r"\(\s*['\"]([a-z _]+)['\"]\s*,\s*['\"]([\u4e00-\u9fff A-Za-z]+)['\"]\s*\)", 语音src)
语音中文集 = {cn for _, cn in 控制对}
# synthesizer 的 EMOTION_VOCAB 键（RapSynth/TTS 族唯一词表）
唱歌src = 读(os.path.join(
    "EmoCompanion_角色挂载与情感注入工程/07_说唱合成Qwen3RapSynth/tts", "synthesizer.py"))
情感键 = set(re.findall(r"['\"]([\u4e00-\u9fff]{2,4})['\"]\s*:\s*['\"]\[emotion\]", 唱歌src))
# 共享情感 = 两族同时出现的词（应为 悲伤 / 兴奋）
共享 = 语音中文集 & 情感键
check("跨族存在共享情感", len(共享) >= 2, f"共享={共享}")
# 已知重叠 悲伤/兴奋 必须在两族都健在（防任一族丢弃共享概念）
for 词 in ("悲伤", "兴奋"):
    check(f"共享情感[{词}]两族均在", 词 in 语音中文集 and 词 in 情感键)
print(f"  [info] 语音合成族中文标签={sorted(语音中文集)}")
print(f"  [info] RapSynth 情感键={sorted(情感键)}")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)