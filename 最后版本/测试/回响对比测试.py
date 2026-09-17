# -*- coding: utf-8 -*-
"""
回响引擎对比测试：裸生成 vs V1 语义回响生成
比较 语义熵 / 重复率 / 生成速度 / 显存占用
用法: python 测试\回响对比测试.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
项目根 = r"j:\最后版本！"
if 项目根 not in sys.path:
    sys.path.insert(0, 项目根)

模型路径 = r"l:\模型空间\Qwen2.5-0.5B-Instruct"
提示词列表 = [
    "今天心情很好，说说你开心的事。",
    "晚上一个人在家有点孤单，陪我聊聊天。",
    "讲一个让你感动的小故事。",
]
生成步数 = 80

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def 裸生成(tokenizer, 模型, 提示词, 步数=生成步数):
    输入 = tokenizer(提示词, return_tensors="pt").to("cuda:0")
    开始 = time.time()
    with torch.no_grad():
        输出 = 模型.generate(
            **输入, max_new_tokens=步数, temperature=1.0, top_p=0.9,
            do_sample=True, repetition_penalty=1.0,
        )
    耗时 = time.time() - 开始
    文本 = tokenizer.decode(输出[0][输入.input_ids.shape[1]:], skip_special_tokens=True)
    return 文本, 耗时


def 计算指标(文本):
    # 语义熵：用简单字符分布熵近似（避免重复加载模型）
    import math
    from collections import Counter
    if not 文本:
        return 0.0, 0.0
    计数 = Counter(文本)
    总 = len(文本)
    熵 = -sum((c / 总) * math.log(c / 总) for c in 计数.values())
    # 重复率：相邻 4-gram 重复占比
    n = 4
    if 总 <= n:
        return 熵, 0.0
    gram = [文本[i:i + n] for i in range(总 - n + 1)]
    出现 = Counter(gram)
    重复数 = sum(次数 - 1 for 次数 in 出现.values() if 次数 > 1)
    return round(熵, 4), round(重复数 / len(gram), 4)


def 主():
    print("=" * 66)
    print("  V1 语义回响引擎 · 裸生成 vs 回响生成 对比")
    print(f"  模型: {模型路径}（λ=0.08 γ=0.07 τ=0.09，max_new={生成步数}）")
    print("=" * 66)

    tokenizer = AutoTokenizer.from_pretrained(模型路径)

    # ── 裸生成 ──
    from transformers import AutoModelForCausalLM
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16, device_map="cuda:0").eval()
    显存前 = torch.cuda.memory_allocated() / 2**20
    裸结果 = []
    for 提示 in 提示词列表:
        文本, 耗时 = 裸生成(tokenizer, 模型, 提示)
        裸结果.append({"提示": 提示, "文本": 文本, "耗时": round(耗时, 2)})
    显存后 = torch.cuda.memory_allocated() / 2**20
    裸显存 = round(显存后 - 显存前, 1)
    del 模型
    torch.cuda.empty_cache()

    # ── V1 回响生成 ──
    from 核心引擎.推理架构 import V1推理引擎
    v1 = V1推理引擎()
    初始 = v1.初始化(模型路径, {"量化": "fp16", "max_new_tokens": 生成步数, "λ": 0.08, "γ": 0.07, "τ": 0.09})
    if not 初始.get("成功"):
        print(f"  初始化失败: {初始.get('错误')}")
        return
    回响结果 = []
    for 提示 in 提示词列表:
        输出 = v1.生成(提示)
        if 输出.get("成功"):
            回响结果.append({
                "提示": 提示,
                "文本": 输出["回复"],
                "耗时": 输出["指标"].get("耗时秒"),
                "熵": 输出["指标"].get("语义熵"),
                "重复率": 输出["指标"].get("重复率"),
            })
        else:
            回响结果.append({"提示": 提示, "错误": 输出.get("错误")})
    v1.释放()

    # ── 展示对比 ──
    裸熵总和, 回熵总和 = 0.0, 0.0
    裸重总和, 回重总和 = 0.0, 0.0
    for i, 提示 in enumerate(提示词列表):
        print("\n" + "─" * 66)
        print(f"  提示词: {提示}")
        裸 = 裸结果[i]
        回 = 回响结果[i]
        裸熵, 裸重 = 计算指标(裸["文本"])
        if "错误" in 回:
            print(f"  [裸生成] 耗时{裸['耗时']}s | 熵{裸熵} | 重复率{裸重}")
            print(f"  [回响生成] 失败: {回['错误']}")
            continue
        回熵 = 回.get("熵")
        回重 = 回.get("重复率")
        print(f"  ┌ 裸生成    耗时 {裸['耗时']}s | 语义熵 {裸熵} | 重复率 {裸重}")
        print(f"  │ 回响生成  耗时 {回['耗时']}s | 语义熵 {回熵} | 重复率 {回重}")
        熵变化 = ((回熵 - 裸熵) / 裸熵 * 100) if 裸熵 else 0
        print(f"  └ 熵变化 {熵变化:+.1f}%" + ("（回响提升表达细腻度）" if 熵变化 > 10 else ""))
        裸熵总和 += 裸熵; 回熵总和 += 回熵 or 0
        裸重总和 += 裸重; 回重总和 += 回重 or 0

    条数 = len(提示词列表)
    print("\n" + "=" * 66)
    print(f"  汇总（{条数} 条提示词）")
    print(f"  平均语义熵 : 裸 {裸熵总和/条数:.3f} → 回响 {回熵总和/条数:.3f}")
    print(f"  平均重复率 : 裸 {裸重总和/条数:.3f} → 回响 {回重总和/条数:.3f}")
    print(f"  模型显存   : 加载后 {裸显存}MB（0.5B fp16）")
    print("=" * 66)


if __name__ == "__main__":
    主()
