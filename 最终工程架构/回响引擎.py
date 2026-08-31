# -*- coding: utf-8 -*-
"""
回响引擎 — 带 λ 步数衰减的回响生成引擎
========================================
在 echo_common.运行回响 基础上扩展"每步 λ 调度"：
前 衰减起始步 步使用固定 λ，之后线性衰减至 λ × 终点λ比例（在生成结束时到达）。

实现要点
--------
- 复用 echo_common（加载模型 / 计算语义熵 / 计算重复率 / 既有补丁）
- 创建 回响注入器 后，在其 轮次回调 中设置 injector.当前λ = λ调度函数(当前步数)
- monkey-patch 注入偏置：优先读取 injector.当前λ（存在时），否则走原阶段逻辑；
  与原 _快注入偏置（GPU 投影矩阵缓存）补丁兼容——先保存原方法再包装
- 提供独立 λ调度函数 便于离线测试

自测（不加载模型）：
    python 回响引擎.py --自测
"""
import sys
import torch

回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

# 触发 echo_common 的既有补丁（向量化质心 / GPU投影缓存 / Peft兼容钩子）
import echo_common
from echo_common import 计算语义熵, 计算重复率

from semantic_echo.回响池 import 语义回响池
from semantic_echo.采样处理器 import 回响注入器


# ════════════════════════════════════════════════════════
# λ 步数调度（独立函数，便于离线自测）
# ════════════════════════════════════════════════════════
def λ调度函数(步数, lam, 起始步, 终点比例, 总步数):
    """每步 λ 调度：前 起始步 步保持 lam，之后线性衰减到 lam*终点比例（总步数处到达）。

    参数
    ----
    步数 : int      当前生成步（轮次回调收到的 pool.当前步数）
    lam : float     初始 λ
    起始步 : int    衰减起始步（此前 λ 恒定）
    终点比例 : float 终点 λ 占初始 λ 的比例（0~1），如 0.3 表示衰减到 0.3λ
    总步数 : int    计划生成总步数（max_new_tokens）

    返回
    ----
    float 当前步应使用的 λ
    """
    if 步数 <= 起始步 or 总步数 <= 起始步:
        return lam
    已衰减比例 = (步数 - 起始步) / (总步数 - 起始步)
    return lam * (1 - (1 - 终点比例) * 已衰减比例)


# ════════════════════════════════════════════════════════
# 注入偏置补丁：优先读取 injector.当前λ（λ 步数衰减模式），否则走原阶段逻辑
# 与 echo_common 的 _快注入偏置（GPU 投影矩阵缓存）兼容：先保存原方法再包装
# ════════════════════════════════════════════════════════
_原注入偏置 = 回响注入器.注入偏置  # 此刻已被 echo_common 替换为 _快注入偏置


def _步数调度注入偏置(self, logits):
    """λ 步数衰减专用注入偏置。

    当注入器存在 当前λ 属性（步数衰减模式）时，无视阶段直接以 当前λ 注入
    （数值等价于原 _快注入偏置 的快速路径）；否则委托原 注入偏置 走原阶段逻辑。
    """
    if hasattr(self, "当前λ"):
        if self.当前λ == 0.0 or self.pool.是否为空:
            return logits
        质心 = self.pool.计算质心().to(self.device)
        if not hasattr(self, "_投影矩阵GPU缓存"):
            # fp16 缓存：量化模型 compute_dtype=fp16，显存减半（与 _快注入偏置 一致）
            self._投影矩阵GPU缓存 = self.投影矩阵.to(self.device, dtype=torch.float16)
        偏置 = 质心.to(self._投影矩阵GPU缓存.dtype) @ self._投影矩阵GPU缓存
        return logits + (偏置 * self.当前λ).unsqueeze(0)
    return _原注入偏置(self, logits)


回响注入器.注入偏置 = _步数调度注入偏置
print("[回响引擎] 已应用 λ 步数调度注入偏置补丁（优先读 injector.当前λ）")


# ════════════════════════════════════════════════════════
# 带 λ 步数衰减的回响生成
# ════════════════════════════════════════════════════════
def 运行回响_步数衰减(model, tokenizer, prompt, lam, gamma, 情感过滤器实例=None,
                    保留策略="衰减", 滑动窗口=3, max_new_tokens=2048, 归一化基准=None,
                    repetition_penalty=1.0, 衰减起始步=256, 终点λ比例=0.3, 前缀="",
                    progress_callback=None):
    """带 λ 步数衰减的语义回响生成（逻辑与 echo_common.运行回响 一致，仅 λ 改为每步调度）。

    前 衰减起始步 步用 lam，之后线性衰减到 lam*终点λ比例（生成结束时到达）。
    通过 轮次回调 在每步推进后设置 injector.当前λ = λ调度函数(当前步数)。

    progress_callback : 可选回调 (当前步数, 总步数)，每步推进后调用（供长上下文实验
        后台监控打印进度，如每 256 步一行）。

    返回 dict：
        {文本, 熵列表, 平均熵, 步数, 重复率, 池统计(最终大小/有效温度/质心范数/情感命中率/NaNInf),
         前缀, token列表(生成 token ids，供区间统计)}
    """
    hidden_dim = model.config.hidden_size
    if 归一化基准 is not None:
        lam = lam * 归一化基准 / hidden_dim
        print(f"  [λ归一化] {lam:.4f} (hidden_dim={hidden_dim}, 基准={归一化基准})")
    pool = 语义回响池(hidden_dim=hidden_dim, decay_gamma=gamma,
                     保留策略=保留策略, 滑动窗口大小=滑动窗口)
    if hasattr(model, "_echo_injector"):
        try:
            model._echo_injector._移除钩子()
        except Exception:
            pass
    injector = 回响注入器(
        model, pool, lambda_strength=lam, projection_seed=42, last_n_layers=1,
        情感过滤器实例=情感过滤器实例,
        思考标记对=("", ""),
        思考阶段λ=lam,
        正文阶段λ=0.0,
    )
    model._echo_injector = injector
    # 初始 λ：第 0 步注入即用 lam；轮次回调随后逐步更新
    injector.当前λ = lam

    # 每步推进后更新当前λ（轮次回调在 pool.推进 之后调用，收到推进后的步数）
    def 步进回调(当前步数, _池):
        injector.当前λ = λ调度函数(当前步数, lam, 衰减起始步, 终点λ比例, max_new_tokens)
        if progress_callback is not None:
            progress_callback(当前步数, max_new_tokens)

    完整prompt = 前缀 + prompt
    输入ids = tokenizer(完整prompt, return_tensors="pt").to(model.device).input_ids
    熵列表 = []

    def logits_cb(step, logits):
        熵列表.append(计算语义熵(logits))

    try:
        with torch.no_grad():
            输出ids = injector.生成(
                输入ids, max_new_tokens=max_new_tokens,
                temperature=1.0, top_p=0.9, top_k=50,
                repetition_penalty=repetition_penalty,
                logits_callback=logits_cb, tokenizer=tokenizer,
                轮次回调=步进回调,
            )
    finally:
        injector._移除钩子()

    pre_len = 输入ids.shape[1]
    生成ids = 输出ids[0][pre_len:]
    文本 = tokenizer.decode(生成ids, skip_special_tokens=True)
    质心 = pool.计算质心()
    池统计 = {
        "最终大小": pool.大小, "有效温度": pool.计算有效温度(),
        "质心范数": round(质心.norm().item(), 6),
        "情感命中率": round(pool.情感命中率, 4),
        "NaN/Inf检查": bool(torch.isfinite(质心).all()),
    }
    return {
        "文本": 文本, "熵列表": 熵列表,
        "平均熵": sum(熵列表) / len(熵列表) if 熵列表 else 0.0,
        "步数": len(熵列表), "重复率": 计算重复率(生成ids.tolist()),
        "池统计": 池统计, "前缀": 前缀[:60],
        "token列表": 生成ids.tolist(),
    }


# ════════════════════════════════════════════════════════
# 离线自测
# ════════════════════════════════════════════════════════
def 自测λ调度():
    """离线自测：不加载模型，打印 λ 调度曲线验证单调衰减、256 步后从 λ 降至 0.3λ"""
    lam = 0.10
    起始步 = 256
    终点比例 = 0.3
    总步数 = 2048
    采样步 = [0, 128, 256, 384, 512, 1024, 2048]
    print("=" * 64)
    print("λ 步数调度自测（lam=0.10, 起始步=256, 终点比例=0.3, 总步数=2048）")
    print("-" * 64)
    数值 = []
    for 步 in 采样步:
        v = λ调度函数(步, lam, 起始步, 终点比例, 总步数)
        数值.append(v)
        print(f"  步 {步:>5} → λ = {v:.6f}")
    单调 = all(数值[i] >= 数值[i + 1] for i in range(len(数值) - 1))
    前段恒定 = all(abs(v - lam) < 1e-12 for v in 数值[:3])      # 0/128/256 均保持 lam
    终点达标 = abs(数值[-1] - lam * 终点比例) < 1e-12           # 2048 = 0.3λ
    print("-" * 64)
    print(f"单调非增:            {'通过' if 单调 else '失败'}")
    print(f"起始段(0~256)保持λ:  {'通过' if 前段恒定 else '失败'}")
    print(f"终点(2048)=0.3λ:     {'通过' if 终点达标 else '失败'}")
    print(f"曲线: {数值[0]:.4f} → {数值[-1]:.4f}（总降幅 {(1 - 数值[-1] / 数值[0]) * 100:.1f}%）")
    print("=" * 64)
    return 单调 and 前段恒定 and 终点达标


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="回响引擎（带 λ 步数衰减）")
    parser.add_argument("--自测", action="store_true", help="离线自测 λ 调度曲线（不加载模型）")
    args = parser.parse_args()
    if args.自测:
        通过 = 自测λ调度()
        sys.exit(0 if 通过 else 1)
    print("用法: python 回响引擎.py --自测")
