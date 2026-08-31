# -*- coding: utf-8 -*-
"""解码控制基类（共享单一实现）

抽取同源脚本（潮汐解码器 / 混合注入器 / 混合锚点器）中大量复制的六类
"解码控制"逻辑，收敛为单一实现，消除逻辑代码的重复与已证实的漂移风险：

  - AI 腔抑制（v3/v4 三通道 + 首 token 黑名单 + 短语感知 + decode 级过滤）
  - 身份暴露硬拦截（v6：我是/作为 + 身份名词 → -inf）
  - 句子边界硬停止（v6：短回复更似真人）
  - 口语化引导（v5：抬升语气词）
  - 长度收尾（v5：超目标长度促句尾标点/EOS）
  - 在线退化兜底（共享 2-gram 重复率框架；子类可覆写 触发条件/处置动作）

子类职责：
  1. 在自身 __init__ 中设置 model / tokenizer / vocab_size，并调用
     super().__init__(...)（或多继承时显式调用本类 __init__）注入控制参数。
  2. 需要差异化时覆盖类常量词表（AI腔词表 / 首token黑名单 / 身份打开器列表 /
     身份名词列表 / 口语化词表）。
  3. 生成循环中按需调用 _AI腔抑制 / _身份暴露拦截 / _口语化引导 / _长度收尾 /
     _句子停止 / _兜底监测。
"""
import re
from typing import Dict, List, Tuple

import torch


class 解码控制基类:
    """六类解码控制逻辑的单一实现（强度/长度/标点可配置，词表可覆盖）"""

    # ──────────────────────────────────────────────
    # v3/v4 AI 腔 token 表（含短语感知 / 首 token 黑名单）
    # ──────────────────────────────────────────────

    # AI 身份词/正式书面语（LLM-Judge 低分主因：57% 回复含 AI 腔）
    AI腔词表 = {
        # 强抑制（AI 身份暴露）
        "AI": 3.0, "助手": 3.0, "人工智能": 3.0, "模型": 3.0, "智能": 2.5,
        # 中抑制（服务性套话）
        "作为": 2.5, "提供": 2.0, "帮助": 2.0, "用户": 2.0, "服务": 2.0,
        "抱歉": 2.5, "对不起": 2.5, "请告诉我": 2.0, "很高兴": 2.0,
        "请问": 1.5, "回答": 1.5, "信息": 1.5, "相关": 1.5, "方面": 1.5,
        "如果": 1.5, "需要": 1.5, "可以": 1.0, "能够": 1.5,
        # 弱抑制（正式书面语）
        "一些": 1.0, "以下": 1.5, "总之": 1.5, "以上": 1.5, "首先": 1.0,
        "其次": 1.0, "最后": 1.0, "根据": 1.5, "建议": 1.0, "内容": 1.0,
        # v4 新增：短语感知补充（"语言模型/文本生成/随时/功能/设计/目标/主要"等绕行词）
        "语言": 2.0, "文本": 1.5, "生成": 1.5, "随时": 1.5, "功能": 1.5,
        "设计": 1.5, "目标": 1.5, "主要": 1.0,
        # v4.2 身份暴露复合/替代词（句中"我是一个程序/机器人/model/作为一个"）
        "作为一个": 2.5, "作为一名": 2.5, "机器人": 2.0, "程序": 2.0,
        "model": 2.5, "虚拟": 1.5, "语料": 1.5,
        # v4.3 跨模型：Qwen3 的  thinking 思考标记（单 token，禁止模型"思考"）
        " thinking": 4.0, "think": 2.0,
    }

    # v4 首 token 黑名单：真人回复从不以 AI 身份词/服务套话开头
    # 权重用于第一 token 的强抑制（抑制 = 强度 × 6 × 权重）
    首token黑名单 = {
        "作为": 3.0, "我是": 3.0, "我的": 2.5, "抱歉": 3.0, "对不起": 3.0,
        "请": 2.0, "如果": 1.5, "根据": 1.5, "AI": 3.0, "助手": 3.0,
        "模型": 3.0, "语言": 2.0, "文本": 1.5, "很高兴": 2.0, "非常": 1.0,
        "Q": 3.0, "作为一个": 3.0, "作为一名": 3.0, "机器人": 2.0,
        "程序": 2.0, "model": 2.5, " thinking": 4.0,
    }

    # v4 短语感知前缀：近文本若正处这些前缀之后，抑制 ×3（打"作为AI/语言模型"等长短语）
    _短语前缀 = [
        "作为", "我是", "我的", "语言", "随时", "请", "提供", "帮助",
        "无法", "抱歉", "对不起", "AI", "文本", "目标", "功能", "设计",
        "智能", "生成",
    ]

    # v4.1 首 token decode 级过滤：任何候选 token decode 后以这些开头 → 直接 -inf
    # 覆盖"作为一个/作为一名/如果你/我很"等复合 token（单 token 词表管不到的）
    _AI腔开头 = re.compile(
        r"^(作为|抱歉|对不起|不好意思|我是|我的|我很|我无法|我不能|如果|请|"
        r"根据|很高兴|AI|助手|模型|语言|文本|智能|Q|I\b|非常|很遗憾)"
    )

    # ──────────────────────────────────────────────
    # v6 身份暴露硬拦截（decode 级）
    # ──────────────────────────────────────────────
    # 60 样本实测：混合输出仍有"我是一个大型语言处理系统/数字人/AI模型"类
    # 身份暴露（10% 样本，全部拿 2 分）→ 在"我是/作为…"刚成形时对身份名词直接 -inf。
    # 只拦截紧邻组合，正常句子（"我是真的…"/"作为纪念"）不受影响。
    身份打开器列表 = [
        "我是一个正在运行", "我是一个正在", "我是一个", "我是一台",
        "我是24小时", "我是基于", "我是", "作为一个", "作为",
        "我的功能", "我的目的", "我被设计", "我的存在",
        # v6.3：更多身份开启器
        "我的出现", "我的使命", "我的作用", "我的回答",
    ]
    身份名词列表 = [
        "AI", "人工智能", "助手", "模型", "语言", "系统", "程序", "机器人",
        "数字", "虚拟", "计算机", "预训练", "算法", "智能体", "语料",
        "自然语言", "大语言", "语言处理", "聊天机器人", "在线", "正在运行",
        "大型", "大规模", "大規模", "人工", "运行", "24小时",
        # v6.1：多 token 绕行短语的早期阻断（只紧跟在"我是/作为"后触发，安全）
        "基于", "正在", "数据", "框架", "处理", "规模", "训练",
        # v6.3：绕行身份名词补充（"为人类解决问题/知识库/无意识的存在/阿里巴巴开发"）
        "人类", "人们", "用户", "知识库", "数据库", "网络", "存在", "无意识",
        "编程", "逻辑", "服务", "推荐", "解答", "阿里巴巴", "物理",
    ]

    # ──────────────────────────────────────────────
    # v5 口语化词表（真人式俏皮语气）
    # ──────────────────────────────────────────────
    口语化词表 = {
        # 强（真人式俏皮语气）
        "哈哈": 1.5, "嘻嘻": 1.5, "啦": 1.2, "呀": 1.2, "嘛": 1.2, "哟": 1.2,
        "～": 1.2, "嘿嘿": 1.2, "嗯嗯": 1.2, "哎呀": 1.2,
        # 中（日常语气）
        "啊": 1.0, "呢": 1.0, "哦": 0.8, "嗯": 0.8, "哎": 0.8, "哇": 0.8,
        "吧": 0.8, "吗": 0.8, "哈": 0.7, "耶": 1.0,
        # 弱（口头禅/感叹）
        "咦": 0.6, "唉": 0.6, "呗": 0.6, "咯": 0.6, "嘿": 0.6, "呵呵": 0.6,
        "呀呼": 0.8, "噗": 0.8,
    }

    def __init__(
        self,
        *args,
        AI腔抑制强度: float = 2.0,
        口语化强度: float = 1.0,
        身份拦截: bool = True,
        目标长度: int = 34,
        句子停止: bool = True,
        最长句数: int = 2,
        最短字数: int = 12,
        最大字数: int = 90,
        退化窗口: int = 40,
        兜底阈值: float = 0.6,
        句尾标点: Tuple[str, ...] = ("。", "！", "？", "；"),
        **kwargs,
    ) -> None:
        """注入解码控制参数。

        args/kwargs：容忍多继承 MRO 链中其它基类（如 回响注入器）__init__
        内部调用 super().__init__(...) 时透传过来的无关参数，避免 TypeError。
        """
        _ = (args, kwargs)
        self.AI腔抑制强度 = AI腔抑制强度
        self.口语化强度 = 口语化强度
        self.身份拦截 = 身份拦截
        self.目标长度 = 目标长度
        self.句子停止 = 句子停止
        self.最长句数 = 最长句数
        self.最短字数 = 最短字数
        self.最大字数 = 最大字数
        self.退化窗口 = 退化窗口
        self.兜底阈值 = 兜底阈值
        self.句尾标点 = 句尾标点
        # 兜底状态（供 _兜底监测 使用）
        self._兜底计数 = 0

    # ──────────────────────────────────────────────
    # token 表构建（词表来自类常量，子类可覆盖词表）
    # ──────────────────────────────────────────────

    def _构建AI腔token表(self) -> Dict[int, float]:
        """
        将 AI 腔词映射到 token 空间（只登记单 token 词）。
        token_id → 抑制权重（越大压得越狠）。
        """
        表: Dict[int, float] = {}
        for 词, 权重 in self.AI腔词表.items():
            if not 词:
                continue
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 权重
        return 表

    def _构建首token表(self) -> Dict[int, float]:
        表: Dict[int, float] = {}
        for 词, 权重 in self.首token黑名单.items():
            if not 词:
                continue
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 权重
        return 表

    def _构建口语化token表(self) -> Dict[int, float]:
        表: Dict[int, float] = {}
        for 词, 权重 in self.口语化词表.items():
            if not 词:
                continue
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 权重
        return 表

    def _构建句尾标点表(self) -> Dict[int, float]:
        """句尾标点（促收尾）：。！？； 单 token（标点列表来自 句尾标点 属性）"""
        表: Dict[int, float] = {}
        for 词 in self.句尾标点:
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 1.0
        return 表

    # ──────────────────────────────────────────────
    # AI 腔抑制（v4 三通道）
    # ──────────────────────────────────────────────

    def _AI腔抑制(self, logits: torch.Tensor) -> torch.Tensor:
        """
        v4 三通道 AI 腔抑制：
        (1) 首 token 黑名单：第一 token 绝不以"作为/我是/抱歉/AI"开头（×6 强压）
        (2) 短语感知：近文本若正处 AI 腔短语前缀之后（如"作为"+"AI"），抑制 ×3
        (3) 基础抑制：全部 AI 腔 token 施加 -强度×权重 偏置
        """
        if self.AI腔抑制强度 <= 0 or not self._AI腔token表:
            return logits
        强度 = self.AI腔抑制强度

        # (1) 首 token 黑名单（生成第一个新 token 时，_已生成token列表 为空）
        if not self._已生成token列表:
            for tid, 权重 in self._首token表.items():
                if tid < self.vocab_size:
                    logits[0, tid] -= 强度 * 6.0 * 权重
            # v4.1 decode 级过滤：扫 top-300 候选，AI 腔开头一律 -inf
            topk = torch.topk(logits, k=min(300, self.vocab_size), dim=-1)
            for v, tid in zip(topk.values[0], topk.indices[0]):
                txt = self.tokenizer.decode([tid.item()], skip_special_tokens=True)
                if txt and self._AI腔开头.match(txt):
                    logits[0, tid] = float('-inf')
            return logits

        # (2) 短语感知：解码最近 3 个 token，若构成 AI 腔短语前缀 → 抑制 ×3
        倍数 = 1.0
        近文 = "".join(
            self.tokenizer.decode([t], skip_special_tokens=True)
            for t in self._已生成token列表[-3:]
        )
        for 前缀 in self._短语前缀:
            idx = 近文.rfind(前缀)
            if idx != -1 and len(近文) - idx <= 4:
                倍数 = 3.0
                break

        # (3) 基础抑制
        for tid, 权重 in self._AI腔token表.items():
            if tid < self.vocab_size:
                logits[0, tid] -= 强度 * 权重 * 倍数
        return logits

    # ──────────────────────────────────────────────
    # v6 身份暴露硬拦截（decode 级）
    # ──────────────────────────────────────────────

    def _身份暴露拦截(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.身份拦截 or not self._生成文本:
            return logits
        尾部 = self._生成文本[-10:]
        # v6.3：开启器出现在尾部（最后10字）即触发（原 endswith 会漏"我是为人类…"类绕行）
        if not any(o in 尾部 for o in self.身份打开器列表):
            return logits
        topk = torch.topk(logits, k=min(80, self.vocab_size), dim=-1)
        for _v, tid in zip(topk.values[0], topk.indices[0]):
            候选 = self.tokenizer.decode([tid.item()], skip_special_tokens=True)
            if any(n in 候选 for n in self.身份名词列表):
                logits[0, tid] = float('-inf')
        return logits

    # ──────────────────────────────────────────────
    # v6 句子边界硬停止（短回复）
    # ──────────────────────────────────────────────
    # 真人回复均值 32 字（中位 20），1.5B 混合动辄 60+ 字长文——长度是最大破绽。
    # v6 改为在完整句子边界后硬停止：2 句且 ≥ 最短字数 → 停；超 3 句必停；超 最大字数 强制停。
    def _句子停止(self) -> bool:
        if not self.句子停止 or not self._生成文本:
            return False
        if len(self._生成文本) >= self.最短字数 and self._句子数 >= self.最长句数:
            return True
        if self._句子数 >= self.最长句数 + 1:
            return True
        if len(self._生成文本) >= self.最大字数:
            return True
        return False

    # ──────────────────────────────────────────────
    # v5 人味儿引导（口语化 + 长度收尾）
    # ──────────────────────────────────────────────

    def _口语化引导(self, logits: torch.Tensor) -> torch.Tensor:
        """
        v5：对语气词 token 施加正偏置，让输出更口语化、更像真人聊天。
        真人回复高频用"啦/呀/嘛/哈/～"等语气词；AI 回复偏书面。
        """
        if self.口语化强度 <= 0 or not self._口语化token表:
            return logits
        强度 = self.口语化强度
        for tid, 权重 in self._口语化token表.items():
            if tid < self.vocab_size:
                logits[0, tid] += 强度 * 权重
        return logits

    def _长度收尾(self, logits: torch.Tensor) -> torch.Tensor:
        """
        v5：真人回复均值 32 字，AI 回复 79 字——长度就是最大破绽。
        已生成文本超过 目标长度 后，对句尾标点/EOS 施加正偏置促进收尾。
        """
        if self._生成文本 and len(self._生成文本) < self.目标长度:
            return logits
        if not self._句尾标点表:
            return logits
        超长 = len(self._生成文本) - self.目标长度
        强度 = min(2.5, max(0.5, 超长 * 0.15))
        for tid in self._句尾标点表:
            if tid < self.vocab_size:
                logits[0, tid] += 强度
        eos = self.model.config.eos_token_id
        if eos is not None and eos < self.vocab_size:
            logits[0, eos] += 强度 * 0.5
        return logits

    # ──────────────────────────────────────────────
    # 在线退化兜底（共享框架）
    # ──────────────────────────────────────────────

    def _兜底监测(self) -> None:
        """
        在线退化兜底（共享框架）：
        - 共享：最近 退化窗口 个 token 的 2-gram 重复率计算 + _兜底计数 管理；
        - 可扩展：子类可覆写 _兜底_触发（检测条件）与 _兜底_处置（衰减/归零动作）。
        """
        最近 = self._已生成token列表[-self.退化窗口:]
        if len(最近) < 8:
            return
        重复率 = self._兜底_重复率(最近)
        if self._兜底_触发(重复率, 最近):
            self._兜底计数 += 1
            self._兜底_处置(重复率, self._兜底计数)
        else:
            self._兜底计数 = 0

    @staticmethod
    def _兜底_重复率(最近: List[int]) -> float:
        """共享：最近 token 的 2-gram 重复率（越大越退化）"""
        if len(最近) < 2:
            return 0.0
        唯一数 = len(set(tuple(最近[i:i + 2]) for i in range(len(最近) - 1)))
        return 1.0 - 唯一数 / (len(最近) - 1)

    def _兜底_触发(self, 重复率: float, 最近: List[int]) -> bool:
        """默认（潮汐解码器/混合注入器版）：重复率超阈值即退化。"""
        return 重复率 > self.兜底阈值

    def _兜底_处置(self, 重复率: float, 触发次数: int) -> None:
        """默认（混合注入器版）：统计计数 + α 衰减；连续 3 次仍退化 → α 归零。"""
        self.统计["触发兜底"] += 1
        if 触发次数 >= 3:
            self.当前α = 0.0
        else:
            self.当前α *= 0.5
