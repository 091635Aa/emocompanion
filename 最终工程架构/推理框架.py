# -*- coding: utf-8 -*-
"""
推理框架 — 集成框架（动态策略 v2 + RAG + LoRA + 回响）
=====================================================
- 加载模型（复用 echo_common.加载模型）；可选挂载 LoRA（peft 热切换）
- 自动匹配推荐参数（自适应匹配.推荐参数）
- 动态策略 v2：检测情感密度，密度 > 0.15 时——
    A = 固定不动
    B = 降τ不升λ（τ→0.05，λ 不变）
    C = λ+0.02 且 τ→0.05
- RAG：FAISS 检索 top-k 文档拼接 [参考信息] 前缀（复用 build_vector_db 编码/检索思路；
  向量库缺失时回退纯回响并打印警告）
- 回响：长上下文启用时用 回响引擎.运行回响_步数衰减（λ 步数衰减），
  否则用 echo_common.运行回响；λ 用动态策略后的值，γ/τ 用匹配值，
  情感过滤器用带 τ 的包装（阈值情感过滤器）

自检（加载 1.5B 跑 2 条提示词，可接受耗时）：
    python 推理框架.py [--量化 fp16|4bit]
"""
import sys
import os

回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

rag脚本目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\04_RAG数据库创建\indexing_scripts"
if rag脚本目录 not in sys.path:
    sys.path.insert(0, rag脚本目录)

from echo_common import 加载模型, 运行回响, 创建情感过滤器
from semantic_echo_framework import (
    阈值情感过滤器, 归一化基准表,
    动态策略情感密度阈值, 动态策略τ目标值, 动态策略λ增量,
)
import 自适应匹配
import 回响引擎

# RAG 向量库路径（与 build_vector_db.py 的输出一致）
RAG向量库目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\04_RAG数据库创建\vector_db"


class 推理框架:
    """集成框架：动态策略 v2 + RAG + LoRA + 回响"""

    def __init__(self, 模型路径, 量化=None, rag=False, lora=None, 动态策略="B",
                 长上下文=False, 归一化基准=896):
        self.模型路径 = 模型路径
        self.量化 = 量化
        self.动态策略 = 动态策略
        self.长上下文 = 长上下文

        # ── 加载模型 + tokenizer ──
        self.model, self.tokenizer = 加载模型(模型路径, 量化=量化)
        self.hidden_dim = int(self.model.config.hidden_size)
        self.vocab_size = int(getattr(self.model.config, "vocab_size", 151936))

        # ── 归一化基准：默认 896，但若 hidden_dim 匹配归一化基准表则升级为表值
        #    （此时缩放因子 = 基准/hidden_dim = 1，扫描表 λ 直接生效，不做二次缩放） ──
        if 归一化基准 == 896 and self.hidden_dim in 归一化基准表:
            self.归一化基准 = 归一化基准表[self.hidden_dim]
        else:
            self.归一化基准 = 归一化基准

        # ── 推荐参数（扫描表优先，未命中用公式） ──
        参数 = 自适应匹配.推荐参数(self.hidden_dim, self.vocab_size)
        self.λ基准 = 参数["λ"]
        self.γ基准 = 参数["γ"]
        self.τ基准 = 参数["τ"]
        self.参数来源 = 参数["来源"]

        # ── RAG ──
        self.rag = bool(rag)
        self._rag检索器 = None
        self._rag警告过 = False

        # ── LoRA（热切换） ──
        self.lora路径 = None
        if lora:
            self.挂载LoRA(lora)

        # ── 惰性情感过滤器（带 τ 阈值包装，复用 semantic_echo_framework 逻辑） ──
        self._情感过滤器 = None

        print(
            f"[推理框架] {os.path.basename(模型路径)} hidden={self.hidden_dim} "
            f"vocab={self.vocab_size} | 推荐 λ={self.λ基准} γ={self.γ基准} τ={self.τ基准} "
            f"({self.参数来源}) | 归一化基准={self.归一化基准} "
            f"动态策略={动态策略} 长上下文={长上下文} RAG={self.rag}"
        )

    # ═══════════════════════════════════════════
    # 情感过滤与动态策略
    # ═══════════════════════════════════════════
    def _获取情感过滤器(self):
        """惰性创建情感过滤器（带 τ 阈值包装），避免每次生成重复加载词库"""
        if self._情感过滤器 is None:
            self._情感过滤器 = 阈值情感过滤器(创建情感过滤器(), τ=self.τ基准)
        return self._情感过滤器

    def 检测情感密度(self, text) -> float:
        """jieba 分词 + 情感过滤器命中率（复用 semantic_echo_framework 逻辑）"""
        if not text:
            return 0.0
        try:
            import jieba
            词列表 = [词 for 词 in jieba.cut(str(text)) if 词.strip()]
        except ImportError:
            # 无 jieba 时按空白切分兜底
            词列表 = [词 for 词 in str(text).split() if 词.strip()]
        if not 词列表:
            return 0.0
        过滤器 = self._获取情感过滤器()
        命中 = 过滤器.筛选(词列表, self.tokenizer)
        return len(命中) / len(词列表)

    # ═══════════════════════════════════════════
    # RAG（FAISS 检索，复用 build_vector_db 的检索器）
    # ═══════════════════════════════════════════
    def _获取RAG检索器(self):
        """惰性加载 FAISS 检索器；向量库缺失/加载失败时返回 None（回退纯回响）"""
        if self._rag检索器 is None and not self._rag警告过:
            try:
                import build_vector_db
                from pathlib import Path
                self._rag检索器 = build_vector_db.RAG检索器(Path(RAG向量库目录))
            except Exception as e:
                self._rag检索器 = False
                self._rag警告过 = True
                print(f"[RAG] 警告：向量库加载失败，回退纯回响（{e}）")
        return self._rag检索器 or None

    def _检索RAG前缀(self, prompt, top_k=2) -> str:
        """检索 top-k 文档拼接 [参考信息] 前缀；失败返回空串"""
        检索器 = self._获取RAG检索器()
        if not 检索器:
            return ""
        try:
            结果 = 检索器.检索(prompt, top_k=top_k)
            if 结果:
                print(f"[RAG] 检索到 {len(结果)} 条（top score={结果[0]['score']:.4f}）")
                return "[参考信息]" + "\n".join(d["text"] for d in 结果) + "\n"
            print("[RAG] 检索结果为空")
        except Exception as e:
            print(f"[RAG] 检索失败：{e}，回退纯回响")
        return ""

    # ═══════════════════════════════════════════
    # LoRA 热切换
    # ═══════════════════════════════════════════
    def 挂载LoRA(self, 路径):
        """挂载 peft 适配器（热切换，异常捕获打印）"""
        try:
            import peft
            self.model = peft.PeftModel.from_pretrained(self.model, 路径)
            self.lora路径 = 路径
            print(f"[LoRA] 已挂载适配器: {路径}")
        except Exception as e:
            print(f"[LoRA] 挂载失败（{e}），继续使用基座模型")

    def 卸载LoRA(self):
        """卸载 peft 适配器，回到基座模型（热切换，异常捕获打印）"""
        try:
            import peft
            if isinstance(self.model, peft.PeftModel):
                # PeftModel → LoraModel → HF 模型，解包回基座
                self.model = self.model.base_model.model
                self.lora路径 = None
                print("[LoRA] 已卸载适配器，恢复基座模型")
            else:
                print("[LoRA] 当前无适配器，无需卸载")
        except Exception as e:
            print(f"[LoRA] 卸载失败（{e}）")

    # ═══════════════════════════════════════════
    # 统一生成接口
    # ═══════════════════════════════════════════
    def 生成(self, prompt, max_new_tokens=256, 前缀="", λ覆盖=None, γ覆盖=None, τ覆盖=None,
             思考标记对=None, 复用池=None, repetition_penalty=1.0):
        """完整流程：动态策略 v2 → RAG → 回响。

        λ覆盖/γ覆盖/τ覆盖: 显式覆盖扫描表参数（任务自适应 λ 用）；None 时用扫描表/公式值。
        思考标记对: ("思考：", "\n回答：") 等——启用注入器的思考阶段（CoT），
                    思考阶段用 λ、正文阶段用 0.0 注入（原实现已支持，此处透传）。
        复用池: 跨轮持久回响池（None 则每次新建）；返回结果携带 "池" 供下轮复用。
        repetition_penalty: >1 施加重复惩罚（对齐裸模型 1.05 用）。

        返回 {文本, 平均熵, 重复率, 情感命中率, λ, γ, τ, 动态信息{启用,情感密度,决策},
              步数, 池统计, 池}
        """
        # ── 1. 动态策略 v2：密度 > 0.15 时 B=降τ不升λ / C=λ+0.02 且 τ→0.05 / A=固定 ──
        λ = self.λ基准 if λ覆盖 is None else λ覆盖
        γ = self.γ基准 if γ覆盖 is None else γ覆盖
        τ = self.τ基准 if τ覆盖 is None else τ覆盖
        动态信息 = {"启用": self.动态策略 != "A", "情感密度": None, "决策": None}
        if self.动态策略 in ("B", "C"):
            密度 = self.检测情感密度(prompt)
            动态信息["情感密度"] = round(密度, 4)
            if 密度 > 动态策略情感密度阈值:
                if self.动态策略 == "B":
                    τ = 动态策略τ目标值  # 降τ不升λ
                    动态信息["决策"] = (f"策略B：情感密度{密度:.3f}>{动态策略情感密度阈值}"
                                     f" → τ降至{τ:.3f}，λ不变")
                else:  # C
                    λ = λ + 动态策略λ增量
                    τ = 动态策略τ目标值
                    动态信息["决策"] = (f"策略C：情感密度{密度:.3f}>{动态策略情感密度阈值}"
                                     f" → λ+{动态策略λ增量} 且 τ降至{τ:.3f}")
                print(f"[动态策略] {动态信息['决策']}")
            else:
                动态信息["决策"] = (f"策略{self.动态策略}：情感密度{密度:.3f}≤{动态策略情感密度阈值}"
                                 f" → 保持默认 τ={τ:.4f}、λ={λ:.4f}")
                print(f"[动态策略] {动态信息['决策']}")
        else:
            动态信息["决策"] = "策略A：固定不动"

        # ── 2. RAG 前缀注入 ──
        最终前缀 = 前缀 or ""
        if self.rag:
            rag前缀 = self._检索RAG前缀(prompt)
            if rag前缀:
                最终前缀 = rag前缀 + 最终前缀

        # ── 3. 情感过滤器（带 τ 阈值包装） ──
        过滤器 = self._获取情感过滤器()
        过滤器.τ = τ

        # ── 4. 回响：长上下文启用时用 λ 步数衰减，否则用 echo_common.运行回响 ──
        if self.长上下文:
            if 复用池 is not None:
                print("[推理框架] 警告：长上下文路径暂不支持复用池，忽略")
            结果 = 回响引擎.运行回响_步数衰减(
                self.model, self.tokenizer, prompt, lam=λ, gamma=γ,
                情感过滤器实例=过滤器, 保留策略="衰减", 滑动窗口=3,
                max_new_tokens=max_new_tokens, 归一化基准=self.归一化基准,
                repetition_penalty=1.0, 衰减起始步=256, 终点λ比例=0.3,
                前缀=最终前缀)
        else:
            结果 = 运行回响(
                self.model, self.tokenizer, prompt, lam=λ, gamma=γ,
                情感过滤器实例=过滤器, 保留策略="衰减", 滑动窗口=3,
                max_new_tokens=max_new_tokens, 前缀=最终前缀,
                归一化基准=self.归一化基准, repetition_penalty=repetition_penalty,
                思考标记对=思考标记对, pool=复用池)

        # ── 5. 汇总返回 ──
        池统计 = 结果.get("池统计") or {}
        return {
            "文本": 结果.get("文本", ""),
            "平均熵": 结果.get("平均熵", 0.0),
            "重复率": 结果.get("重复率", 0.0),
            "情感命中率": max(0.0, min(1.0, 池统计.get("情感命中率", 0.0))),  # 命中率口径修复：clip 到 [0,1]（池统计命中时不自增总检查数，可能 >1）
            "λ": round(λ, 6),
            "γ": round(γ, 6),
            "τ": round(τ, 6),
            "动态信息": 动态信息,
            "步数": 结果.get("步数", 0),
            "池统计": 池统计,
            "池": 结果.get("池"),
        }


# ════════════════════════════════════════════════════════
# 自检：加载 1.5B 模型跑 2 条提示词（可接受耗时）
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="推理框架自检（1.5B）")
    parser.add_argument("--量化", choices=["fp16", "4bit"], default="fp16")
    parser.add_argument("--模型", default=r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    量化值 = None if args.量化 == "fp16" else args.量化
    print("=" * 60)
    print(f"推理框架自检：{args.模型}（量化={args.量化}）")
    print("=" * 60)
    t0 = time.time()
    框架 = 推理框架(args.模型, 量化=量化值, rag=False, lora=None,
                   动态策略="B", 长上下文=False)
    print(f"模型加载耗时 {time.time() - t0:.1f}s")

    自检提示词 = ["今天的中标消息让我兴奋得睡不着", "我想了解一下这个产品的功能"]
    for 提示 in 自检提示词:
        t1 = time.time()
        try:
            结果 = 框架.生成(提示, max_new_tokens=128)
            print("-" * 60)
            print(f"提示词: {提示}")
            print(f"回复  : {结果['文本'][:120]}")
            print(f"平均熵: {结果['平均熵']:.4f} | 重复率: {结果['重复率']:.4f} | "
                  f"情感命中率: {结果['情感命中率']:.4f} | 步数: {结果['步数']}")
            print(f"λ={结果['λ']} γ={结果['γ']} τ={结果['τ']} | "
                  f"动态: {结果['动态信息']['决策']} | 耗时 {time.time() - t1:.1f}s")
        except Exception as e:
            import traceback
            print(f"生成失败: {e}")
            traceback.print_exc()
    print("=" * 60)
    print("推理框架自检完成")
