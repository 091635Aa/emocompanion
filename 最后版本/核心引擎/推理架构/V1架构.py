# -*- coding: utf-8 -*-
"""
V1 架构 — 简单语义回响推理引擎
==============================
来源：语义回响项目（复制适配，i:\\Desktop\\语义回响\\semantic_echo\\采样处理器.py 的 回响注入器）
     与 i:\\Desktop\\语义回响\\semantic_echo\\回响评估器.py 的 计算语义熵，
     以及 f:\\最终工程架构\\自适应匹配.py 的 λ/γ/τ 扫描表与公式。

包含：
- 参数推荐：推荐参数(hidden_dim)（扫描表优先，未命中用公式）
- 指标计算：计算语义熵 / 计算重复率
- 核心引擎：回响注入器（forward hook 捕获 hidden_state、随机静态投影、注入 logits、生成循环）
- 推理引擎：V1推理引擎（加载模型 → 回响池 + 注入器 → 生成 → 指标）

依赖：torch / math / typing；transformers 仅在运行时按需导入（缺失时返回友好错误）。
"""

import os
import math
import time
from typing import Optional, Callable, List, Dict

import torch
import torch.nn.functional as F

from .回响池 import 语义回响池


# ══════════════════════════════════════════════════
# 一、λ/γ/τ 参数推荐（扫描表优先，未命中用公式）
# ══════════════════════════════════════════════════

扫描表 = {
    896: (0.50, 0.05, 0.10),   # 0.5B
    1536: (0.08, 0.07, 0.09),  # 1.5B
    2048: (0.10, 0.08, 0.06),  # 3B
    3584: (0.06, 0.12, 0.05),  # 7B
}

基准hidden_dim = 896


def 公式λ(hidden_dim: float) -> float:
    """λ 分段公式 — 回响注入强度（大模型自动减弱）。"""
    if hidden_dim >= 2048:
        return 0.28 * (基准hidden_dim / hidden_dim)
    return 0.5 * (基准hidden_dim / hidden_dim) ** 1.5


def 公式γ(hidden_dim: float) -> float:
    """γ = 0.05 × (hidden_dim / 896) ** 0.5 — 回响池衰减系数。"""
    return 0.05 * (hidden_dim / 基准hidden_dim) ** 0.5


def 公式τ(hidden_dim: float) -> float:
    """τ = 0.10 × (hidden_dim / 896) ** (-0.5) — 情感筛选强度阈值。"""
    return 0.10 * (hidden_dim / 基准hidden_dim) ** (-0.5)


def 推荐参数(hidden_dim) -> dict:
    """返回 {λ, γ, τ, 来源}：优先匹配扫描表（hidden_dim 相等即用），未命中时用公式。

    参数
    ----
    hidden_dim : int   模型隐藏层维度
    """
    hidden_dim = int(hidden_dim)
    if hidden_dim in 扫描表:
        λ, γ, τ = 扫描表[hidden_dim]
        return {"λ": float(λ), "γ": float(γ), "τ": float(τ), "来源": "扫描表"}
    return {
        "λ": round(公式λ(hidden_dim), 4),
        "γ": round(公式γ(hidden_dim), 4),
        "τ": round(公式τ(hidden_dim), 4),
        "来源": "公式",
    }


# ══════════════════════════════════════════════════
# 二、指标计算
# ══════════════════════════════════════════════════

def 计算语义熵(logits: torch.Tensor) -> float:
    """
    计算单 token 位置的语义熵。

    定义：H = -Σ P(w_i) ln P(w_i)，其中 P = softmax(logits)。

    来源：语义回响项目 回响评估器.py（复制适配）。
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits 必须是 torch.Tensor，收到 {type(logits)}")

    logits_dim = logits.dim()
    if logits_dim not in (1, 2):
        raise ValueError(f"logits 维度必须为 1 或 2，收到 {logits_dim} 维")

    if logits_dim == 2:
        if logits.shape[0] != 1:
            raise ValueError(
                f"2 维 logits 的 batch 维度必须为 1，收到 shape={tuple(logits.shape)}"
            )
        logits = logits.squeeze(0)

    if logits.shape[0] == 0:
        raise ValueError("logits 的 vocab_size 不能为 0")

    # 转 float32 再计算：模型以 fp16 加载时，极小概率在 fp16 下溢为 0，
    # 而 1e-12 在 fp16 中不可表示，导致 log(0) = -inf、0 * -inf = NaN。
    logits = logits.float()
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum().item()
    return entropy


def 计算重复率(文本, n: int = 2) -> float:
    """相邻 n-gram 重复比例：重复出现的 n-gram 造成的冗余占比（自实现）。

    定义：重复率 = 重复次数超出首次的部分之和 / 全部 n-gram 数量。
    """
    if not 文本:
        return 0.0
    文本 = str(文本)
    总长 = len(文本) - n + 1
    if 总长 <= 0:
        return 0.0
    ngram列表 = [文本[i:i + n] for i in range(总长)]
    出现次数: Dict[str, int] = {}
    for g in ngram列表:
        出现次数[g] = 出现次数.get(g, 0) + 1
    重复数 = 0
    for 次数 in 出现次数.values():
        if 次数 > 1:
            重复数 += 次数 - 1
    return round(重复数 / len(ngram列表), 4)


def _当前显存MB() -> float:
    """返回当前已分配显存（MB）；无 CUDA 或查询失败时返回 0。"""
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
    except Exception:
        pass
    return 0.0


# ══════════════════════════════════════════════════
# 三、模型加载（transformers 缺失时返回友好错误）
# ══════════════════════════════════════════════════

def _加载模型与分词器(模型路径: str, 量化: str = "fp16"):
    """加载因果语言模型与分词器。

    加载顺序：4bit（可选）→ fp16→cuda:0 → 失败降级 CPU。

    返回:
        (模型, 分词器, 错误)：成功时 错误 为 None；失败时 模型/分词器 为 None。
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return (
            None, None,
            "缺少 transformers 库，无法加载模型。请先安装：\n"
            "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers",
        )
    if not 模型路径 or not os.path.isdir(模型路径):
        return None, None, f"模型路径不存在：{模型路径}"
    try:
        # ── 4bit 量化（bitsandbytes，失败自动回退 fp16） ──
        if str(量化).lower() in ("4bit", "qlora", "bitsandbytes"):
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
                if torch.cuda.is_available():
                    量化配置 = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    模型 = AutoModelForCausalLM.from_pretrained(
                        模型路径, quantization_config=量化配置, device_map="auto")
                    分词器 = AutoTokenizer.from_pretrained(模型路径)
                    return 模型, 分词器, None
            except ImportError:
                pass  # bitsandbytes 未装 → 回退 fp16
            except Exception:
                pass  # 4bit 加载失败 → 回退 fp16
        # ── fp16 → cuda:0，失败降级 CPU ──
        try:
            模型 = AutoModelForCausalLM.from_pretrained(模型路径, torch_dtype=torch.float16)
            if torch.cuda.is_available():
                模型 = 模型.to("cuda:0")
        except Exception:
            模型 = AutoModelForCausalLM.from_pretrained(模型路径)
        分词器 = AutoTokenizer.from_pretrained(模型路径)
        return 模型, 分词器, None
    except Exception as e:
        return None, None, f"模型加载失败：{e}"


# ══════════════════════════════════════════════════
# 四、回响注入器（来源：采样处理器.py，复制适配）
# ══════════════════════════════════════════════════

def _定位最后一层(model) -> torch.nn.Module:
    """根据模型架构自动定位最后一层 Transformer 层。

    Raises
    ------
    ValueError
        如果模型架构不属于已知模式
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[-1]
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h[-1]
    if (
        hasattr(model, 'model')
        and hasattr(model.model, 'decoder')
        and hasattr(model.model.decoder, 'layers')
    ):
        return model.model.decoder.layers[-1]
    raise ValueError(
        f"无法自动定位模型 {type(model).__name__} 的最后一层。"
        f"支持：LLaMA/Qwen/Mistral/GPT-2/OPT/BLOOM"
    )


class 回响注入器:
    """
    语义回响注入器 — 核心推理增强模块。

    使用方式
    --------
    >>> pool = 语义回响池(hidden_dim=model.config.hidden_size)
    >>> injector = 回响注入器(model, pool, lambda_strength=0.08)
    >>> output_ids = injector.生成(input_ids, max_new_tokens=256)
    """

    def __init__(
        self,
        model,
        echo_pool: 语义回响池,
        lambda_strength: float = 0.08,
        uncertainty_threshold: float = 0.09,
        projection_seed: int = 42,
        last_n_layers: int = 4,
        思考标记对: tuple = ("", ""),
        思考阶段λ: Optional[float] = None,
        正文阶段λ: float = 0.0,
        启用注入: bool = True,
        启用捕获: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model
            HuggingFace Transformers 兼容的预训练模型
        echo_pool : 语义回响池
            共享的回响池实例
        lambda_strength : float
            注入偏置的强度系数 λ
        uncertainty_threshold : float
            不确定性权重低于此阈值时不入池（避免噪声积累）τ
        projection_seed : int
            随机投影矩阵的固定种子
        last_n_layers : int
            取最后 N 层的 hidden_state 平均作为"语义场"向量
        思考标记对 : tuple[str, str]
            思考阶段边界标记，如 ("<think>", "</think>")；为空时行为与无思考一致
        思考阶段λ : Optional[float]
            思考阶段的注入强度，为 None 时与 lambda_strength 相同
        正文阶段λ : float
            正文阶段的注入强度，为 0.0 时正文阶段不注入
        启用注入 : bool
            为 False 时注入偏置直接返回原 logits（回响层关闭）
        启用捕获 : bool
            为 False 时不将 hidden_state 写入回响池（回响层关闭）
        """
        self.model = model
        self.pool = echo_pool
        self.lambda_strength = lambda_strength
        self.uncertainty_threshold = uncertainty_threshold
        self.last_n_layers = last_n_layers
        self.启用注入 = 启用注入
        self.启用捕获 = 启用捕获

        self.思考标记对 = 思考标记对
        self.思考阶段λ = 思考阶段λ
        self.正文阶段λ = 正文阶段λ

        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        self.device = model.device

        self.当前hidden_state: Optional[torch.Tensor] = None

        self.当前阶段: str = "思考"
        self.已解码文本: str = ""

        # 随机静态投影矩阵
        self._初始化投影(projection_seed)

        # 注册 forward hook
        self._钩子列表: List = []
        self._注册钩子()

    # ──────────────────────────────────────────────
    # 投影矩阵初始化
    # ──────────────────────────────────────────────

    def _初始化投影(self, seed: int) -> None:
        """
        创建固定的随机投影矩阵：hidden_dim → vocab_size

        使用 Kaiming 均匀初始化缩放因子，保证投影后输出的
        方差 ≈ 输入方差，避免注入偏置过大或过小。
        """
        rng = torch.Generator()
        rng.manual_seed(seed)

        scale = math.sqrt(2.0 / self.hidden_dim)
        self.投影矩阵 = torch.randn(
            self.hidden_dim, self.vocab_size,
            generator=rng,
            dtype=torch.float32,
        ) * scale

        self.投影矩阵.requires_grad_(False)

    # ──────────────────────────────────────────────
    # Forward Hook 注册
    # ──────────────────────────────────────────────

    def _注册钩子(self) -> None:
        """
        在模型最后 N 层注册 forward hook，捕获每步的 hidden_state。

        标准 Decoder-only 架构（model.model.layers）对最后 last_n_layers 层
        各注册一个钩子，取输出平均；否则退化为只捕获最后一层。
        """
        try:
            if (
                hasattr(self.model, 'model')
                and hasattr(self.model.model, 'layers')
            ):
                所有层 = self.model.model.layers
                目标层数 = min(self.last_n_layers, len(所有层))
                起始索引 = len(所有层) - 目标层数
                self.目标层索引 = list(range(起始索引, len(所有层)))

                for idx in self.目标层索引:
                    handle = 所有层[idx].register_forward_hook(self._创建多层钩子(idx))
                    self._钩子列表.append(handle)
            else:
                最后一层 = _定位最后一层(self.model)
                handle = 最后一层.register_forward_hook(self._单层钩子)
                self._钩子列表.append(handle)
        except Exception as e:
            raise ValueError(f"钩子注册失败：{e}")

    def _单层钩子(self, module, inputs, output) -> None:
        """单层钩子：直接从输出中提取最后一个位置的 hidden_state（1D 向量）"""
        if isinstance(output, tuple):
            hs = output[0][0, -1, :]
        else:
            hs = output[0, -1, :]
        self.当前hidden_state = hs.detach().clone()

    def _创建多层钩子(self, layer_idx: int) -> Callable:
        """
        多层钩子工厂：为指定层创建钩子，收集所有目标层输出后取平均。
        """
        def hook(module, inputs, output) -> None:
            if isinstance(output, tuple):
                hs = output[0][0, -1, :]
            else:
                hs = output[0, -1, :]

            if not hasattr(self, '_层输出缓存'):
                self._层输出缓存: Dict[int, torch.Tensor] = {}
            self._层输出缓存[layer_idx] = hs.detach()

            if len(self._层输出缓存) == len(self.目标层索引):
                向量列表 = [self._层输出缓存[i] for i in sorted(self.目标层索引)]
                self.当前hidden_state = torch.stack(向量列表).mean(dim=0)
                self._层输出缓存.clear()

        return hook

    def _移除钩子(self) -> None:
        """移除所有注册的 forward hook"""
        for handle in self._钩子列表:
            handle.remove()
        self._钩子列表.clear()

    # ──────────────────────────────────────────────
    # 阶段切换检测
    # ──────────────────────────────────────────────

    def _检测阶段切换(self, 新token_id: int, tokenizer=None) -> bool:
        """
        检测是否进入了新阶段（思考→正文），通过检测已生成文本中
        是否出现思考结束标记来判断。
        """
        新文本 = tokenizer.decode([新token_id]) if tokenizer else ""
        self.已解码文本 += 新文本

        if (
            self.思考标记对
            and self.思考标记对 != ("", "")
            and self.当前阶段 == "思考"
            and self.思考标记对[1] in self.已解码文本
        ):
            self.当前阶段 = "正文"
            return True

        return False

    # ──────────────────────────────────────────────
    # 核心操作：注入 + 捕获
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 注入偏置(self, logits: torch.Tensor) -> torch.Tensor:
        """
        将回响池质心通过随机投影映射到 logits 空间，作为偏置注入。

        根据当前阶段选择不同的 λ 强度系数：
        - 思考阶段：使用 思考阶段λ（若不为 None）或 lambda_strength
        - 正文阶段：使用 正文阶段λ（为 0 时不注入）
        """
        if not self.启用注入:
            return logits

        if self.当前阶段 == "正文" and self.正文阶段λ == 0.0:
            return logits

        if self.pool.是否为空:
            return logits

        质心 = self.pool.计算质心().to(self.device)

        if self.当前阶段 == "思考" and self.思考阶段λ is not None:
            有效λ = self.思考阶段λ
        elif self.当前阶段 == "正文":
            有效λ = self.正文阶段λ
        else:
            有效λ = self.lambda_strength

        偏置 = 质心 @ self.投影矩阵.to(self.device)
        偏置 = 偏置 * 有效λ

        return logits + 偏置.unsqueeze(0)

    @torch.no_grad()
    def 捕获回响(self, logits: torch.Tensor, tokenizer=None) -> None:
        """
        将当前 hidden_state 存入回响池（带不确定性阈值筛选）。
        """
        if not self.启用捕获:
            return
        if self.当前hidden_state is None:
            return

        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max().item()
        不确定性 = 1.0 - max_prob

        if 不确定性 <= self.uncertainty_threshold:
            return

        self.pool.添加(self.当前hidden_state, 不确定性)

    # ──────────────────────────────────────────────
    # 自定义生成循环
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 生成(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        logits_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        tokenizer=None,
        轮次回调: Optional[Callable[[int, object], None]] = None,
    ) -> torch.Tensor:
        """
        带语义回响的自回归生成循环。

        每步执行：前向 → 注入 → 捕获 → 采样 → 推进

        Parameters
        ----------
        input_ids : torch.Tensor
            shape=(1, seq_len)，初始 prompt token ID 序列
        max_new_tokens : int
            最大新生成 token 数
        temperature : float
            采样温度，>1 更随机，<1 更确定
        top_p : float
            nucleus sampling 累积概率阈值
        top_k : int
            top-k 采样保留的候选数
        repetition_penalty : float
            重复惩罚系数（>1 抑制重复）
        eos_token_id : Optional[int]
            结束标记 ID，为 None 时从模型配置获取
        logits_callback : Optional[Callable[[int, torch.Tensor], None]]
            可选回调函数，每步生成后调用，传入 (步数, logits)
        tokenizer
            可选 tokenizer，用于思考阶段边界检测
        轮次回调 : Optional[Callable[[int, object], None]]
            可选回调函数，每步推进后调用，传入 (当前步数, self.pool)

        Returns
        -------
        torch.Tensor
            shape=(1, total_len)，完整生成的 token ID 序列
        """
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id

        past_key_values: Optional[tuple] = None
        已生成 = input_ids.clone()
        已生成token集合: set = set()

        for 步 in range(max_new_tokens):
            模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
            outputs = self.model(
                模型输入,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # 重复惩罚
            if repetition_penalty != 1.0:
                for token_id in 已生成token集合:
                    logits[0, token_id] /= repetition_penalty

            # (1) 注入：将回响池偏置加到 logits
            logits = self.注入偏置(logits)

            # 外部日志回调（用于实验记录/熵计算）
            if logits_callback is not None:
                logits_callback(步, logits)

            # (2) 捕获：将当前 hidden_state 存入回响池
            self.捕获回响(logits, tokenizer=tokenizer)

            # 温度缩放
            logits = logits / temperature

            # Top-p (nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove,
                )
                logits[indices_to_remove] = float('-inf')

            # Top-k 过滤
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            下一个token = torch.multinomial(probs, num_samples=1)

            # (3) 检测阶段切换（思考→正文）
            self._检测阶段切换(下一个token.item(), tokenizer)

            已生成 = torch.cat([已生成, 下一个token], dim=-1)
            已生成token集合.add(下一个token.item())

            # (4) 推进回响池步数
            self.pool.推进()

            # (5) 外部轮次回调（用于滑动窗口策略）
            if 轮次回调 is not None:
                轮次回调(self.pool.当前步数, self.pool)

            if 下一个token.item() == eos_token_id:
                break

        return 已生成

    def __del__(self) -> None:
        """清理注册的 hook，防止内存泄漏"""
        if hasattr(self, '_钩子列表'):
            self._移除钩子()


# ══════════════════════════════════════════════════
# 五、裸生成（回响层关闭/回退时的兜底生成循环）
# ══════════════════════════════════════════════════

@torch.no_grad()
def _裸生成(
    模型,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    eos_token_id: Optional[int] = None,
    logits_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """无回响注入的简单自回归生成循环（回响层回退时使用）。"""
    if eos_token_id is None:
        eos_token_id = getattr(模型.config, "eos_token_id", None)

    past_key_values: Optional[tuple] = None
    已生成 = input_ids.clone()

    for 步 in range(max_new_tokens):
        模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
        outputs = 模型(模型输入, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = getattr(outputs, "past_key_values", None)

        if logits_callback is not None:
            logits_callback(步, logits)

        logits = logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove,
            )
            logits[indices_to_remove] = float('-inf')

        if top_k > 0:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            threshold = top_k_values[:, -1].unsqueeze(-1)
            logits[logits < threshold] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        下一个token = torch.multinomial(probs, num_samples=1)

        已生成 = torch.cat([已生成, 下一个token], dim=-1)

        if eos_token_id is not None and 下一个token.item() == eos_token_id:
            break

    return 已生成


# ══════════════════════════════════════════════════
# 六、默认参数合并
# ══════════════════════════════════════════════════

def _默认参数(参数: Optional[dict]) -> dict:
    """合并用户参数与默认参数（默认值来自 系统配置.json 推理节）。"""
    from ..配置管理 import 获取配置项
    默认 = {
        "架构": "V1简单回响",
        "λ": 获取配置项("推理.默认λ", 0.08),
        "γ": 获取配置项("推理.默认γ", 0.07),
        "τ": 获取配置项("推理.默认τ", 0.09),
        "max_new_tokens": 获取配置项("推理.最大新Token", 256),
        "last_n_layers": 4,
        "投影种子": 42,
        "量化": "fp16",
        "温度": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "池大小上限": 1024,
    }
    if 参数:
        默认.update({k: v for k, v in 参数.items() if v is not None})
    return 默认


# ══════════════════════════════════════════════════
# 七、V1 推理引擎
# ══════════════════════════════════════════════════

class V1推理引擎:
    """V1 简单回响推理引擎：语义回响池 + 回响注入器。

    使用方式
    --------
    >>> 引擎 = V1推理引擎()
    >>> 引擎.初始化("数据/模型库/qwen2.5-1.5b", {"λ": 0.08})
    >>> 结果 = 引擎.生成("你好")
    """

    def __init__(self) -> None:
        self.模型 = None
        self.分词器 = None
        self.回响池: Optional[语义回响池] = None
        self.注入器: Optional[回响注入器] = None
        self.参数: dict = {}
        self.设备 = "cpu"
        self.hidden_dim = 0
        self.显存占用MB = 0.0

    def 初始化(self, 模型路径: str, 参数: Optional[dict] = None) -> dict:
        """加载模型与回响池配置。

        参数:
            模型路径: 基座/微调产出模型绝对路径。
            参数: 推理参数字典，含 λ/γ/τ、max_new_tokens、last_n_layers、
                  投影种子、量化（fp16/4bit）等。

        返回:
            {"成功": bool, "状态": "就绪", "模型路径": ..., "回响池条目数": ..., "提示": ...}
            失败时返回 {"成功": False, "错误": ...}。
        """
        # 幂等：先清理旧资源
        self.释放(清理模型=False)

        参数 = _默认参数(参数)
        量化 = str(参数.get("量化", "fp16"))

        模型, 分词器, 错误 = _加载模型与分词器(模型路径, 量化)
        if 错误:
            return {"成功": False, "错误": 错误}

        try:
            hidden_dim = int(模型.config.hidden_size)
            vocab_size = int(getattr(模型.config, "vocab_size", 151936))
        except Exception as e:
            return {"成功": False, "错误": f"模型配置读取失败：{e}"}

        self.模型 = 模型
        self.分词器 = 分词器
        self.hidden_dim = hidden_dim
        self.设备 = str(模型.device)
        self.参数 = 参数

        # 回响池
        try:
            self.回响池 = 语义回响池(
                hidden_dim=hidden_dim,
                max_pool_size=int(参数.get("池大小上限", 1024)),
                decay_gamma=float(参数.get("γ", 0.07)),
            )
        except Exception as e:
            return {"成功": False, "错误": f"回响池创建失败：{e}"}

        # 注入器
        try:
            self.注入器 = 回响注入器(
                模型, self.回响池,
                lambda_strength=float(参数.get("λ", 0.08)),
                uncertainty_threshold=float(参数.get("τ", 0.09)),
                projection_seed=int(参数.get("投影种子", 42)),
                last_n_layers=int(参数.get("last_n_layers", 4)),
            )
        except Exception as e:
            return {"成功": False, "错误": f"回响注入器创建失败：{e}"}

        self.显存占用MB = _当前显存MB()
        return {
            "成功": True,
            "状态": "就绪",
            "模型路径": 模型路径,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab_size,
            "回响池条目数": self.回响池.大小,
            "λ": float(参数.get("λ", 0.08)),
            "γ": float(参数.get("γ", 0.07)),
            "τ": float(参数.get("τ", 0.09)),
            "量化": 量化,
            "显存占用MB": self.显存占用MB,
            "提示": "",
        }

    def 生成(self, 提示词: str, 角色名: Optional[str] = None,
             记忆开关: bool = True, 记忆外挂实例=None) -> dict:
        """执行一次推理生成，输出回复与指标。

        参数:
            提示词: 用户输入文本。
            角色名: 可选，用于记忆注入时按角色过滤。
            记忆开关: 是否注入相关记忆到提示词（需提供 记忆外挂实例）。
            记忆外挂实例: 可选 记忆外挂 实例（跨会话记忆）。

        返回:
            {"成功": True, "回复": ..., "指标": {"语义熵", "重复率", "池大小",
             "质心范数", "耗时秒", "显存MB"}}
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

        # 记忆注入（跨会话生效）
        最终提示词 = str(提示词)
        if 记忆开关 and 记忆外挂实例 is not None:
            try:
                前缀 = 记忆外挂实例.构建前缀(提示词, 前N=5, 角色名=角色名)
                if 前缀:
                    最终提示词 = 前缀 + 最终提示词
            except Exception:
                pass  # 记忆注入失败不影响生成

        try:
            输入ids = self.分词器(最终提示词, return_tensors="pt").input_ids.to(self.设备)
        except Exception as e:
            return {"成功": False, "错误": f"提示词编码失败：{e}"}

        熵列表: List[float] = []

        def 收集(步: int, logits: torch.Tensor) -> None:
            try:
                熵列表.append(计算语义熵(logits))
            except Exception:
                pass

        try:
            输出ids = self.注入器.生成(
                输入ids,
                max_new_tokens=int(参数.get("max_new_tokens", 256)),
                temperature=float(参数.get("温度", 1.0)),
                top_p=float(参数.get("top_p", 0.9)),
                top_k=int(参数.get("top_k", 50)),
                tokenizer=self.分词器,
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
            "池大小": self.回响池.大小,
            "质心范数": round(float(self.回响池.计算质心().norm().item()), 4),
            "耗时秒": round(time.time() - 开始, 3),
            "显存MB": _当前显存MB(),
        }
        self.显存占用MB = 指标["显存MB"]
        return {"成功": True, "回复": 回复, "指标": 指标}

    def 释放(self, 清理模型: bool = True) -> dict:
        """释放模型、注入器与显存。

        参数:
            清理模型: 为 True 时连模型/分词器一起释放。
        """
        try:
            if self.注入器 is not None:
                self.注入器._移除钩子()
        except Exception:
            pass
        self.注入器 = None
        self.回响池 = None
        if 清理模型:
            self.模型 = None
            self.分词器 = None
            self.显存占用MB = 0.0
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return {"成功": True, "提示": "已释放"}

    @staticmethod
    def 推荐参数(hidden_dim) -> dict:
        """类方法版参数推荐（与模块级 推荐参数 等价）。"""
        return 推荐参数(hidden_dim)
