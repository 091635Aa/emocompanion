# -*- coding: utf-8 -*-
"""
效果演示：基座模型 vs 身份微调适配器 生成对比
用法: python 测试\效果演示.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

基座路径 = r"l:\模型空间\Qwen2.5-0.5B-Instruct"
适配器路径 = r"j:\最后版本！\数据\微调输出\测试角色_20260807153247"
提示词列表 = ["你是谁？", "我今天心情不好，安慰安慰我", "讲一件你小时候的事"]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def 生成(模型, tokenizer, 提示词, max_new=100):
    输入 = tokenizer(提示词, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        输出 = 模型.generate(
            **输入, max_new_tokens=max_new, temperature=0.8, top_p=0.9,
            do_sample=True, repetition_penalty=1.1,
        )
    文本 = tokenizer.decode(输出[0][输入.input_ids.shape[1]:], skip_special_tokens=True)
    return 文本.strip()


def 主():
    print("=" * 70)
    print("  基座模型 vs 身份微调适配器 · 生成效果对比")
    print(f"  基座: {基座路径}")
    print(f"  适配器: {适配器路径}")
    print("=" * 70)

    # ── 阶段1：基座生成 ──
    print("\n[1/2] 加载基座模型...")
    tokenizer = AutoTokenizer.from_pretrained(基座路径)
    基座模型 = AutoModelForCausalLM.from_pretrained(
        基座路径, torch_dtype=torch.float16, device_map="cuda:0"
    ).eval()
    基座结果 = []
    for 提示 in 提示词列表:
        基座结果.append(生成(基座模型, tokenizer, 提示))
    del 基座模型
    torch.cuda.empty_cache()
    print("  基座生成完成，已释放。")

    # ── 阶段2：微调适配器生成 ──
    print("\n[2/2] 加载身份微调适配器...")
    from peft import PeftModel
    模型 = AutoModelForCausalLM.from_pretrained(
        基座路径, torch_dtype=torch.float16, device_map="cuda:0"
    )
    微调模型 = PeftModel.from_pretrained(模型, 适配器路径).eval()
    微调结果 = []
    for 提示 in 提示词列表:
        微调结果.append(生成(微调模型, tokenizer, 提示))
    del 微调模型, 模型
    torch.cuda.empty_cache()

    # ── 展示对比 ──
    for i, 提示 in enumerate(提示词列表):
        print("\n" + "─" * 70)
        print(f"  提示词: {提示}")
        print(f"  ┌─ 微调前（基座）")
        for 行 in (基座结果[i][:120] or "(空)").splitlines() or ["(空)"]:
            print(f"  │ {行[:120]}")
        print(f"  └─ 微调后（身份微调）")
        for 行 in (微调结果[i][:120] or "(空)").splitlines() or ["(空)"]:
            print(f"  │ {行[:120]}")
    print("\n" + "=" * 70)
    print("  说明：0.5B 为最小测试基座，效果有限；实际使用时请选更大的定制基座（如 3B/7B/30B）。")
    print("=" * 70)


if __name__ == "__main__":
    主()
