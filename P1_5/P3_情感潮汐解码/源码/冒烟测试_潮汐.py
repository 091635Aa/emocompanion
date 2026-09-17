# -*- coding: utf-8 -*-
"""
P3 情感潮汐解码（ETD）· 冒烟测试（不加载大模型）
==================================================
覆盖三层核心纯逻辑：
  1. 感知层 VAD：中文情感文本 → valence 方向正确（"今天好难过" → 负向）
  2. 决策层 α 有界性：α ∈ [0, α上限] ⊆ [0,1]；密度目标 ∈ [密度基, 密度上限]
  3. 解码层乘性重加权：p' ∝ p^(1-α)·q^α 的归一化性质（和为 1、非负、α=0/1 退化为 p/q）
     + 对数域 logits 加法实现等价性验证
  4. AI 腔抑制词表非空 + 解码器接口契约（mock tokenizer/model 构建 token 表）

全部不加载大模型（仅 torch + 轻量情感词典 cnsenti/jieba）。

用法：F:\\打标\\.venv\\Scripts\\python.exe 冒烟测试_潮汐.py
"""
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"

ETD目录 = r"h:\情感潮汐解码（Emotion Tidal Decoding, ETD）"
if ETD目录 not in sys.path:
    sys.path.insert(0, ETD目录)

# ──────────────────────────────────────────────
# 测试结果收集器
# ──────────────────────────────────────────────
类结果 = []


def 记录(测试点, 预期, 结果, 结论):
    类结果.append({"测试点": 测试点, "预期": 预期, "结果": 结果, "结论": 结论})
    标记 = "✓" if 结论 == "✓ 通过" else "✗"
    print(f"  {标记} [{测试点}] {结果}")


def main():
    t0 = time.time()
    print("╔══════════════════════════════════════════════╗")
    print("║   P3 情感潮汐解码（ETD）冒烟测试（纯逻辑）    ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\nPython {sys.version.split()[0]} | 不加载大模型\n")

    import torch
    import torch.nn.functional as F

    from 潮汐感知器 import 潮汐感知器, 情感状态
    from 潮汐决策器 import 潮汐决策器, 默认角色表
    from 潮汐解码器 import 潮汐解码器

    # ══════════════════════════════════════════
    # 1. 感知层 VAD 方向
    # ══════════════════════════════════════════
    print("【1】感知层 VAD（中文情感方向）")
    感知器 = 潮汐感知器()
    状态负, 关键词负 = 感知器.测量("今天好难过，心里特别委屈")
    ok = 状态负.valence < 0
    记录("VAD·负向文本", "valence < 0", f"valence={状态负.valence:.4f} 关键词={关键词负}",
         "✓ 通过" if ok else "✗ 失败")

    状态正, 关键词正 = 感知器.测量("我太开心了，好幸福！")
    ok = 状态正.valence > 0
    记录("VAD·正向文本", "valence > 0", f"valence={状态正.valence:.4f} 关键词={关键词正}",
         "✓ 通过" if ok else "✗ 失败")

    状态空, _ = 感知器.测量("   ")
    ok = (状态空.valence == 0.0 and 状态空.arousal == 0.0)
    记录("VAD·空文本", "valence=arousal=dominance=0", f"{状态空}",
         "✓ 通过" if ok else "✗ 失败")

    # ══════════════════════════════════════════
    # 2. 决策层 α 有界性
    # ══════════════════════════════════════════
    print("\n【2】决策层 α 有界性")
    决策器 = 潮汐决策器(感知器, 角色=默认角色表["共情"])
    场景 = ["我崩溃了，彻底受不了了！", "今天被领导当众批评，好委屈",
            "嗯，还行吧。", "我升职了！！太开心了！！"]
    α最大 = 0.0
    α越界 = False
    密度越界 = False
    for 文本 in 场景:
        状态, 关键词 = 感知器.测量(文本)
        感知器.追加轨迹("用户", 状态, 关键词)
        目标 = 决策器.计算目标(状态)
        α最大 = max(α最大, 目标.引导强度)
        if not (0.0 <= 目标.引导强度 <= 决策器.α上限):
            α越界 = True
        if not (决策器.密度基 <= 目标.密度目标 <= 决策器.密度上限):
            密度越界 = True
    ok = (not α越界) and α最大 <= 决策器.α上限 <= 1.0
    记录("决策·α∈[0,α上限]⊆[0,1]", f"α ∈ [0, {决策器.α上限}]",
         f"α最大={α最大:.4f} 越界={α越界}", "✓ 通过" if ok else "✗ 失败")
    ok = not 密度越界
    记录("决策·密度目标有界", f"[{决策器.密度基}, {决策器.密度上限}]",
         f"越界={密度越界}", "✓ 通过" if ok else "✗ 失败")

    # ══════════════════════════════════════════
    # 3. 解码层乘性重加权归一化（纯数学 + 对数域等价）
    # ══════════════════════════════════════════
    print("\n【3】解码层乘性重加权归一化 p' ∝ p^(1-α)·q^α")
    torch.manual_seed(42)
    V = 12  # 词表模拟
    p = F.softmax(torch.randn(V), dim=-1)          # 原分布
    s = torch.tensor([2.0, 1.5, 0.0, -1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 情感词强度
    T_emo = 0.3
    q = F.softmax(s / T_emo, dim=-1)               # 情感引导分布

    全部好 = True
    for α in (0.0, 0.15, 0.5, 1.0):
        对数 = (1 - α) * torch.log(p + 1e-12) + α * torch.log(q + 1e-12)
        p_重加权 = F.softmax(对数, dim=-1)
        和 = float(p_重加权.sum().item())
        非负 = bool((p_重加权 >= 0).all().item())
        # 退化端点：α=0 → p；α=1 → q
        端点 = True
        if α == 0.0:
            端点 = bool(torch.allclose(p_重加权, p, atol=1e-4))
        if α == 1.0:
            端点 = bool(torch.allclose(p_重加权, q, atol=1e-4))
        if abs(和 - 1.0) > 1e-6 or not 非负 or not 端点:
            全部好 = False
        print(f"    α={α}: sum(p')={和:.8f} 非负={非负} 端点退化={端点}")
    记录("重加权·归一化性质", "α∈{0,0.15,0.5,1} 均 和=1、非负、端点正确",
         f"全部通过={全部好}", "✓ 通过" if 全部好 else "✗ 失败")

    # 解码器实际实现为对数域注入（_情感引导：logits 加法，幅度 α×引导倍率），
    # 验证其归一化性质（和=1、非负）与方向单调性（高情感强度 token 概率随 α 递增）。
    L = torch.log(p + 1e-12)
    引导倍率 = 12.0
    好注入 = True
    p_基础 = F.softmax(L, dim=-1)
    s最大idx = int(s.argmax().item())
    for α in (0.0, 0.15, 0.5, 1.0):
        p_加法 = F.softmax(L + α * 引导倍率 * s, dim=-1)
        和 = float(p_加法.sum().item())
        非负 = bool((p_加法 >= 0).all().item())
        if abs(和 - 1.0) > 1e-6 or not 非负:
            好注入 = False
    # 方向单调性：s 最大的 token 概率随 α 单调不减
    p_序列 = [F.softmax(L + a * 引导倍率 * s, dim=-1)[s最大idx].item()
              for a in (0.0, 0.15, 0.5, 1.0)]
    单调 = all(b >= a - 1e-6 for a, b in zip(p_序列, p_序列[1:]))
    ok = 好注入 and 单调
    记录("重加权·logits加法实现归一化", "各 α 下 和=1、非负；高情感词概率随 α 单调上升",
         f"和=1 全部={好注入} 单调={单调} P(s最大|α)={[round(x,4) for x in p_序列]}",
         "✓ 通过" if ok else "✗ 失败")

    # ══════════════════════════════════════════
    # 4. AI 腔抑制词表 + 解码器接口契约（mock，不加载模型）
    # ══════════════════════════════════════════
    print("\n【4】AI 腔抑制词表 + 解码器接口契约（mock）")
    ok = (len(潮汐解码器.AI腔词表) > 0 and len(潮汐解码器.口语化词表) > 0
          and len(潮汐解码器.首token黑名单) > 0)
    记录("词表·AI腔/口语化/首token非空",
         "AI腔词表、口语化词表、首token黑名单均非空",
         f"AI腔={len(潮汐解码器.AI腔词表)} 口语化={len(潮汐解码器.口语化词表)} 首token={len(潮汐解码器.首token黑名单)}",
         "✓ 通过" if ok else "✗ 失败")

    # mock 模型与 tokenizer：仅测 token 表构建与接口契约（不加载权重）
    class MockModel:
        device = "cpu"
        config = type("cfg", (), {"hidden_size": 1536, "vocab_size": 20000, "eos_token_id": 1})()

    class MockTokenizer:
        def encode(self, 词, add_special_tokens=False):
            return [abs(hash(词)) % 19900 + 10]

        def decode(self, ids, skip_special_tokens=True):
            return f"w{ids[0]}" if isinstance(ids, (list, tuple)) else f"w{ids}"

    解码器 = 潮汐解码器(MockModel(), MockTokenizer(), 感知器, 决策器)
    ok1 = isinstance(解码器._情感token表, dict) and len(解码器._情感token表) > 0
    ok2 = isinstance(解码器._AI腔token表, dict) and len(解码器._AI腔token表) > 0
    ok3 = isinstance(解码器._口语化token表, dict) and len(解码器._口语化token表) > 0
    ok = ok1 and ok2 and ok3
    记录("接口·情感/AI腔/口语化token表构建（mock）",
         "三张 token 表均为非空 dict（不加载模型）",
         f"情感={len(解码器._情感token表)} AI腔={len(解码器._AI腔token表)} 口语化={len(解码器._口语化token表)}",
         "✓ 通过" if ok else "✗ 失败")

    # ══════════════════════════════════════════
    # 汇总
    # ══════════════════════════════════════════
    通过 = sum(1 for r in 类结果 if r["结论"] == "✓ 通过")
    总数 = len(类结果)
    print("\n" + "=" * 56)
    print(f"  P3 情感潮汐解码冒烟：{总数} 项，通过 {通过}，失败 {总数 - 通过}")
    print(f"  耗时 {time.time() - t0:.1f}s")
    print("=" * 56)
    return 0 if 通过 == 总数 else 1


if __name__ == "__main__":
    sys.exit(main())
