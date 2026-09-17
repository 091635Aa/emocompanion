# -*- coding: utf-8 -*-
"""
V 通用架构 — 语义回响 + LoRA + RAG + 记忆 四层推理引擎
=====================================================
来源：f:\\最终工程架构\\推理框架.py 与 自适应匹配.py（参考适配），
     并引入"语义回响多模型对照实验"的 架构族因子 结论。

四层（按需激活 + 优雅回退）：
1. 语义回响：回响池 + 回响注入器（复用 V1 注入器封装）
2. LoRA：peft.PeftModel 挂载适配器（缺失/失败回退基座模型）
3. RAG：简单文档库（默认 数据/记忆库）检索拼接 [参考信息] 前缀，无库回退
4. 记忆：记忆外挂注入最近相关记忆 [长期记忆] 前缀，跨会话生效

每层失败自动回退并记录到 回退记录；层状态 反映各层激活情况。
"""

import os
import re
import time

from .V1架构 import (
    回响注入器,
    推荐参数,
    计算语义熵,
    计算重复率,
    _加载模型与分词器,
    _当前显存MB,
)
from .回响池 import 语义回响池
from .记忆外挂 import 记忆外挂


# ══════════════════════════════════════════════════
# 一、架构族因子（语义回响多模型对照实验结论）
# ══════════════════════════════════════════════════

架构族因子 = {
    "Qwen3小": 0.3,   # Qwen3 ≤ 1.7B：小模型敏感，注入减弱
    "Qwen3大": 0.6,   # Qwen3 ≥ 4B
    "gemma": 0.7,
    "SmolLM": 0.5,
    "Phi": 0.8,
    "Qwen2.5": 1.0,   # 基准
}


def 推断架构族(模型名: str = "", 参数量亿=None):
    """按模型名/参数量推断架构族。

    返回:
        (族名, 因子)。未知架构族返回 ("通用", 1.0)。
    """
    名 = (模型名 or "").lower()
    if "smollm" in 名:
        return "SmolLM", 0.5
    if "gemma" in 名:
        return "gemma", 0.7
    if "phi" in 名:
        return "Phi", 0.8
    if "qwen3" in 名:
        if 参数量亿 is not None and float(参数量亿) <= 1.7:
            return "Qwen3小", 0.3
        m = re.search(r"(\d+(?:\.\d+)?)\s*b", 名)
        if m and float(m.group(1)) <= 1.7:
            return "Qwen3小", 0.3
        return "Qwen3大", 0.6
    if "qwen" in 名:
        return "Qwen2.5", 1.0
    return "通用", 1.0


def 通用注入参数(hidden_dim, 模型名: str = "", 参数量亿=None, 量化: str = "fp16") -> dict:
    """通用架构注入参数：λ = 推荐λ × 架构族因子 × 量化因子（4bit ×0.75）。

    参数:
        hidden_dim: 模型隐藏层维度。
        模型名: 模型名称/路径（用于推断架构族）。
        参数量亿: 可选参数量（亿），用于区分 Qwen3 大小。
        量化: "fp16" / "4bit"。

    返回:
        {"λ", "γ", "τ", "架构族", "架构族因子", "量化因子", "来源"}。
    """
    基础 = 推荐参数(hidden_dim)
    族, 因子 = 推断架构族(模型名, 参数量亿)
    量化因子 = 0.75 if str(量化).lower() in ("4bit", "qlora", "bitsandbytes") else 1.0
    return {
        "λ": round(基础["λ"] * 因子 * 量化因子, 6),
        "γ": 基础["γ"],
        "τ": 基础["τ"],
        "架构族": 族,
        "架构族因子": 因子,
        "量化因子": 量化因子,
        "来源": 基础["来源"],
    }


# ══════════════════════════════════════════════════
# 二、默认参数合并
# ══════════════════════════════════════════════════

def _默认参数(参数: dict = None) -> dict:
    """合并用户参数与通用架构默认参数（默认值来自 系统配置.json 推理节）。"""
    from ..配置管理 import 获取配置项
    默认 = {
        "架构": "V通用架构",
        # λ/γ/τ 为 None 时在 初始化 中按 架构族因子 自动计算
        "λ": None,
        "γ": None,
        "τ": None,
        "max_new_tokens": 获取配置项("推理.最大新Token", 256),
        "last_n_layers": 4,
        "投影种子": 42,
        "量化": "fp16",
        "温度": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "池大小上限": 1024,
        # 四层开关（默认全开）
        "启用回响": True,
        "启用LoRA": True,
        "启用RAG": True,
        "启用记忆": True,
        # 附加配置
        "LoRA适配器路径": "",
        "RAG检索库": "",
        "记忆外挂路径": "",
        "参数量亿": None,
    }
    if 参数:
        默认.update({k: v for k, v in 参数.items() if v is not None})
    return 默认


# ══════════════════════════════════════════════════
# 三、通用推理引擎
# ══════════════════════════════════════════════════

class 通用推理引擎:
    """V 通用架构推理引擎：语义回响 + LoRA + RAG + 记忆四层，按需激活 + 优雅回退。

    使用方式
    --------
    >>> 引擎 = 通用推理引擎()
    >>> 引擎.初始化("数据/模型库/qwen2.5-1.5b", {"启用RAG": False})
    >>> 结果 = 引擎.生成("你好")
    """

    def __init__(self) -> None:
        self.模型 = None
        self.分词器 = None
        self.回响池: 语义回响池 = None
        self.注入器: 回响注入器 = None
        self.记忆外挂: 记忆外挂 = None
        self._RAG记忆: 记忆外挂 = None
        self.参数: dict = {}
        self.层状态 = {"回响": "未激活", "LoRA": "未激活", "RAG": "未激活", "记忆": "未激活"}
        self.回退记录: list = []
        self.hidden_dim = 0
        self.显存占用MB = 0.0

    # ──────────────────────────────────────────────
    # 初始化（四层按需激活 + 优雅回退）
    # ──────────────────────────────────────────────

    def 初始化(self, 模型路径: str, 参数: dict = None) -> dict:
        """加载模型与推理配置（λ/γ/τ 等参数取配置项 推理.* 或自动推荐）。

        参数:
            模型路径: 基座/微调产出模型绝对路径。
            参数: 推理参数字典，结构示例：
                {"架构": "V通用架构", "λ": 0.08, "γ": 0.07, "τ": 0.09,
                 "启用回响": True, "启用LoRA": True, "启用RAG": True, "启用记忆": True,
                 "LoRA适配器路径": "", "RAG检索库": "", "记忆外挂路径": ""}

        返回:
            {"成功": bool, "状态": "就绪", "模型路径": ..., "层状态": ...,
             "回退记录": ..., "显存占用MB": ..., "提示": ""}
            失败时返回 {"成功": False, "错误": ...}。
        """
        # 幂等：先清理旧资源
        self.释放()

        参数 = _默认参数(参数)
        量化 = str(参数.get("量化", "fp16"))

        模型, 分词器, 错误 = _加载模型与分词器(模型路径, 量化)
        if 错误:
            return {"成功": False, "错误": 错误}

        try:
            hidden_dim = int(模型.config.hidden_size)
        except Exception as e:
            return {"成功": False, "错误": f"模型配置读取失败：{e}"}

        self.模型 = 模型
        self.分词器 = 分词器
        self.hidden_dim = hidden_dim
        self.参数 = 参数
        self.回退记录 = []
        self.层状态 = {"回响": "未激活", "LoRA": "未激活", "RAG": "未激活", "记忆": "未激活"}

        # λ/γ/τ：用户未给时按 架构族因子 自动计算（通用注入 λ）
        λ = 参数.get("λ")
        γ = 参数.get("γ")
        τ = 参数.get("τ")
        if λ is None or γ is None or τ is None:
            模型名 = os.path.basename(模型路径.rstrip("/\\"))
            注入参数 = 通用注入参数(hidden_dim, 模型名, 参数.get("参数量亿"), 量化)
            λ = 注入参数["λ"]
            γ = 注入参数["γ"]
            τ = 注入参数["τ"]
        self.参数["λ"], self.参数["γ"], self.参数["τ"] = λ, γ, τ

        # ── 第二层：LoRA（挂载适配器，失败回退基座） ──
        if 参数.get("启用LoRA", True) and 参数.get("LoRA适配器路径"):
            try:
                from peft import PeftModel
                self.模型 = PeftModel.from_pretrained(self.模型, 参数["LoRA适配器路径"])
                self.层状态["LoRA"] = "激活"
            except ImportError:
                self.层状态["LoRA"] = "回退"
                self.回退记录.append({"层": "LoRA", "原因": "未安装 peft", "动作": "使用基座模型"})
            except Exception as e:
                self.层状态["LoRA"] = "回退"
                self.回退记录.append({"层": "LoRA", "原因": str(e), "动作": "使用基座模型"})
        else:
            self.层状态["LoRA"] = "未启用" if not 参数.get("启用LoRA", True) else "未配置适配器"

        # ── 第一层：语义回响（核心） ──
        if 参数.get("启用回响", True):
            try:
                self.回响池 = 语义回响池(
                    hidden_dim=hidden_dim,
                    max_pool_size=int(参数.get("池大小上限", 1024)),
                    decay_gamma=float(γ),
                )
                self.注入器 = 回响注入器(
                    self.模型, self.回响池,
                    lambda_strength=float(λ),
                    uncertainty_threshold=float(τ),
                    projection_seed=int(参数.get("投影种子", 42)),
                    last_n_layers=int(参数.get("last_n_layers", 4)),
                )
                self.层状态["回响"] = "激活"
            except Exception as e:
                self.层状态["回响"] = "回退"
                self.回退记录.append({"层": "回响", "原因": str(e), "动作": "使用裸模型生成"})
        else:
            self.层状态["回响"] = "未启用"

        # ── 第四层：记忆外挂（跨会话记忆） ──
        if 参数.get("启用记忆", True):
            try:
                self.记忆外挂 = 记忆外挂(目录=参数.get("记忆外挂路径") or None)
                self.层状态["记忆"] = "激活"
            except Exception as e:
                self.层状态["记忆"] = "回退"
                self.回退记录.append({"层": "记忆", "原因": str(e), "动作": "不注入记忆"})
        else:
            self.层状态["记忆"] = "未启用"

        # ── 第三层：RAG（简单文档库，无库回退） ──
        if 参数.get("启用RAG", True):
            rag目录 = 参数.get("RAG检索库") or None
            if rag目录 and not os.path.isdir(rag目录):
                self.层状态["RAG"] = "回退"
                self.回退记录.append({"层": "RAG", "原因": f"检索库不存在：{rag目录}", "动作": "不注入参考信息"})
            else:
                try:
                    self._RAG记忆 = 记忆外挂(目录=rag目录)
                    self.层状态["RAG"] = "激活"
                except Exception as e:
                    self.层状态["RAG"] = "回退"
                    self.回退记录.append({"层": "RAG", "原因": str(e), "动作": "不注入参考信息"})
        else:
            self.层状态["RAG"] = "未启用"

        self.显存占用MB = _当前显存MB()
        return {
            "成功": True,
            "状态": "就绪",
            "模型路径": 模型路径,
            "hidden_dim": hidden_dim,
            "λ": float(λ),
            "γ": float(γ),
            "τ": float(τ),
            "量化": 量化,
            "层状态": dict(self.层状态),
            "回退记录": list(self.回退记录),
            "显存占用MB": self.显存占用MB,
            "提示": "",
        }

    # ──────────────────────────────────────────────
    # 生成（构建提示词 → 生成 → 指标 + 层状态）
    # ──────────────────────────────────────────────

    def 生成(self, 提示词: str, 角色名: str = None,
             记忆开关: bool = True, 记忆外挂实例=None) -> dict:
        """执行一次推理生成，输出回复、指标与各层激活状态。

        流程：注入记忆/检索结果构建提示词 → 回响注入器生成 → 汇总。

        返回:
            {"成功": True, "回复": ..., "指标": {...}, "层状态": {...},
             "回退记录": [...], "激活层": [...]}
        """
        if self.模型 is None:
            try:
                import transformers  # noqa: F401
            except ImportError:
                return {
                    "成功": False,
                    "错误": "缺少 transformers 库，请先安装：\n"
                            "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers",
                }
            return {"成功": False, "错误": "引擎尚未初始化，请先调用 初始化(模型路径, 参数)"}

        开始 = time.time()
        参数 = self.参数
        外挂 = 记忆外挂实例 or self.记忆外挂

        # ── 构建提示词：记忆层 + RAG 层 ──
        前缀块 = []
        if 记忆开关:
            if self.层状态.get("记忆") == "激活" and 外挂 is not None:
                try:
                    记忆前缀 = 外挂.构建前缀(提示词, 前N=5, 角色名=角色名)
                    if 记忆前缀:
                        前缀块.append(记忆前缀)
                except Exception as e:
                    self.层状态["记忆"] = "回退"
                    self.回退记录.append({"层": "记忆", "原因": str(e), "动作": "本次不注入记忆"})
            if self.层状态.get("RAG") == "激活" and self._RAG记忆 is not None:
                try:
                    rag结果 = self._RAG记忆.检索相关(提示词, 前N=2, 角色名=角色名)
                    if rag结果:
                        rag前缀 = "[参考信息]\n" + "\n".join(
                            m.get("内容", "") for m in rag结果) + "\n"
                        前缀块.append(rag前缀)
                except Exception as e:
                    self.回退记录.append({"层": "RAG", "原因": str(e), "动作": "本次不注入参考信息"})

        if 前缀块:
            最终提示词 = "\n".join(前缀块) + str(提示词)
        else:
            最终提示词 = str(提示词)

        # ── 编码 ──
        try:
            输入ids = self.分词器(最终提示词, return_tensors="pt").input_ids.to(str(self.模型.device))
        except Exception as e:
            return {"成功": False, "错误": f"提示词编码失败：{e}"}

        熵列表 = []

        def 收集(步: int, logits) -> None:
            try:
                熵列表.append(计算语义熵(logits))
            except Exception:
                pass

        # ── 生成（优先回响注入器；回响层回退时用裸生成） ──
        try:
            if self.注入器 is not None:
                输出ids = self.注入器.生成(
                    输入ids,
                    max_new_tokens=int(参数.get("max_new_tokens", 256)),
                    temperature=float(参数.get("温度", 1.0)),
                    top_p=float(参数.get("top_p", 0.9)),
                    top_k=int(参数.get("top_k", 50)),
                    tokenizer=self.分词器,
                    logits_callback=收集,
                )
            else:
                from .V1架构 import _裸生成
                输出ids = _裸生成(
                    self.模型, 输入ids,
                    max_new_tokens=int(参数.get("max_new_tokens", 256)),
                    temperature=float(参数.get("温度", 1.0)),
                    top_p=float(参数.get("top_p", 0.9)),
                    top_k=int(参数.get("top_k", 50)),
                    logits_callback=收集,
                )
        except Exception as e:
            return {"成功": False, "错误": f"生成失败：{e}"}

        回复ids = 输出ids[:, 输入ids.shape[1]:]
        try:
            回复 = self.分词器.decode(回复ids[0], skip_special_tokens=True).strip()
        except Exception:
            回复 = ""

        指标 = {
            "语义熵": round(sum(熵列表) / len(熵列表), 4) if 熵列表 else 0.0,
            "重复率": 计算重复率(回复),
            "池大小": self.回响池.大小 if self.回响池 is not None else 0,
            "质心范数": round(float(self.回响池.计算质心().norm().item()), 4)
                if self.回响池 is not None else 0.0,
            "耗时秒": round(time.time() - 开始, 3),
            "显存MB": _当前显存MB(),
        }
        self.显存占用MB = 指标["显存MB"]
        return {
            "成功": True,
            "回复": 回复,
            "指标": 指标,
            "层状态": dict(self.层状态),
            "回退记录": list(self.回退记录),
            "激活层": [k for k, v in self.层状态.items() if v == "激活"],
        }

    # ──────────────────────────────────────────────
    # 释放
    # ──────────────────────────────────────────────

    def 释放(self) -> dict:
        """释放模型、注入器与显存。"""
        try:
            if self.注入器 is not None:
                self.注入器._移除钩子()
        except Exception:
            pass
        self.注入器 = None
        self.回响池 = None
        self.模型 = None
        self.分词器 = None
        self.记忆外挂 = None
        self._RAG记忆 = None
        self.显存占用MB = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return {"成功": True, "提示": "已释放"}

    @staticmethod
    def 推荐参数(hidden_dim) -> dict:
        """类方法版参数推荐（与 V1 相同扫描表/公式）。"""
        return 推荐参数(hidden_dim)
