# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · 韵律参数预测器（含规则先验基线）

输入：歌词（按行）+ BPM + 风格；输出：每行音节的 起始拍 / F0 轮廓 / 时长缩放 / 能量。

- `predict_lyrics()` 是统一出口（Phase 2 学习模型可遵守同一 `ProsodyPlan` 协议替换之）。
- `styles` 内置三档风格模板（快嘴 / 旋律说唱 / 硬核），对应任务书要求的 3 类样例。
- 规则基线刻意简单、可解释，作为学习模型增益的对照（Task 3.1）。
"""
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

STYLES = ("快嘴", "旋律说唱", "硬核")


def count_syllables(line: str) -> int:
    """粗略字数作为音节代理（中文 1 字 ≈ 1 音节）。"""
    import re
    return len(re.findall(r"[^\s，。,.！？!?、]", line))


@dataclass
class LinePlan:
    index: int
    text: str
    syllables: int
    start_sec: float      # 相对整首起点的起始时间（拍对齐后）
    duration_sec: float   # 目标时长（含字间停连）
    mean_f0: float        # Hz，整句平均音高（相对默认）
    f0_style: str         # contour 类型：flat / arch / punch / learned
    energy: float         # 相对能量系数（0.5 轻 ~ 1.6 重）
    jump: int             # 由前一拍跳进的拍程（用于 pitch shift 换算）
    syllable_f0_delta: Optional[List[float]] = None  # 每音节半音偏移（学习模型）：供逐音节 F0 轮廓注入


@dataclass
class ProsodyPlan:
    bpm: float
    style: str
    beat_sec: float
    lines: List[LinePlan] = field(default_factory=list)

    def total_seconds(self) -> float:
        return max(len(self.lines), 1) * self.beat_sec * 4  # 以 4 拍小节下限


def _style_param(style: str):
    """风格→采样密度与音高/能量参数。"""
    if style == "快嘴":
        return dict(spb=0.5)      # 半拍字（16 分音符感）, 密集, 平直 F0
    if style == "硬核":
        return dict(spb=0.5)
    return dict(spb=1.0)          # 旋律说唱：每拍一字，起伏

def predict_lyrics(lyrics: str, bpm: float = 90.0,
                   style: str = "快嘴", base_f0: float = 180.0) -> ProsodyPlan:
    """解析多行歌词 → 拍对齐 ProsodyPlan。行间以 '\\n' 分隔。"""
    if style not in STYLES:
        style = "快嘴"
    beat_sec = 60.0 / max(bpm, 20.0)
    spb = _style_param(style)["spb"]
    plan = ProsodyPlan(bpm=bpm, style=style, beat_sec=beat_sec)

    t = 0.0
    for i, raw in enumerate(lyrics.splitlines()):
        line = raw.strip()
        if not line:
            continue
        n = count_syllables(line)
        n = max(n, 1)
        # 目标时长 = 音节数 × 每音节拍程 × 单拍时长；至少半拍兜底
        dur = n * spb * beat_sec
        jump = 1 if style != "旋律说唱" else 2

        # 音高：风格相关轮廓
        if style == "旋律说唱":        # 拱形（起-升-降）
            phase = float(i % 4)
            mean_f0 = base_f0 * (1.0 + 0.18 * math.sin(math.pi * phase / 3.0))
            contour = "arch"
        elif style == "硬核":          # 下探 + 中重音
            mean_f0 = base_f0 * (0.86 if i % 2 == 0 else 1.02)
            contour = "punch"
        else:                          # 快嘴：平直近朗读，速率优先
            mean_f0 = base_f0 * (1 + 0.03 * (i % 2))
            contour = "flat"

        energy = {"快嘴": 1.15, "旋律说唱": 0.95, "硬核": 1.35}[style]
        plan.lines.append(LinePlan(
            index=i, text=line, syllables=n,
            start_sec=t, duration_sec=dur,
            mean_f0=mean_f0, f0_style=contour, energy=energy,
            jump=jump,
        ))
        # 下一行从下一拍开始（句间一拍休止），硬核刻意留空拍更狠
        t = t + dur + (0.5 * beat_sec if style != "硬核" else 1.0 * beat_sec)
    return plan


# ---------- Phase 2 学习模型统一接口（替代规则基线，遵守同一协议） ----------
class ProsodyPredictorBase:
    """学习型预测器应实现同接口。16GB 下 batch=1+grad_accum，输出与 LinePlan 兼容。"""

    def predict(self, lyrics: str, bpm: float, style: str,
                base_f0: float = 180.0) -> ProsodyPlan:
        raise NotImplementedError

    def is_learned(self) -> bool:
        return False


_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "weights", "prosody_lstm.pt")


def get_predictor(weights_path: str = _DEFAULT_WEIGHTS):
    """自动选型：若存在学习模型权重则返回 LearnedProsodyPredictor，否则退回规则基线。"""
    if os.path.isfile(weights_path):
        try:
            from .learned import LearnedProsodyPredictor
            return LearnedProsodyPredictor(weights_path)
        except Exception as e:  # 权重/依赖异常时稳定回退，不阻塞管道
            print(f"[warn] 学习模型加载失败回退规则基线: {e}")
    class _RuleAdapter(ProsodyPredictorBase):
        def predict(self, lyrics, bpm, style, base_f0=180.0):
            return predict_lyrics(lyrics, bpm=bpm, style=style, base_f0=base_f0)
        def is_learned(self):
            return False
    return _RuleAdapter()