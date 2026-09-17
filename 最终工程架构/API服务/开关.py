# -*- coding: utf-8 -*-
"""
开关管理 — 运行时开关（前端可热切换）
======================================
- 启用API  ：对外 OpenAI 兼容接口（/v1/*）开关
- 启用RAG  ：None=跟随模型注册配置；True/False=强制开关（热切换 框架.rag）
- 启用LoRA ：同上（热挂载/卸载 peft 适配器）
- 启用记忆 ：超长期记忆系统 挂载开关（写入 + 检索注入）
- 动态策略 ：None=跟随注册配置；A/B/C=强制热切换 框架.动态策略
全局唯一实例：开关 = 开关管理()
"""
import threading


class 开关管理:
    def __init__(self):
        self._锁 = threading.Lock()
        self.启用API = True
        self.启用RAG = None       # None | True | False
        self.启用LoRA = None
        self.启用记忆 = False
        self.动态策略 = None      # None | "A" | "B" | "C"

    def 值(self):
        with self._锁:
            return {
                "启用API": self.启用API,
                "启用RAG": self.启用RAG,
                "启用LoRA": self.启用LoRA,
                "启用记忆": self.启用记忆,
                "动态策略": self.动态策略,
            }

    def 设(self, 名称, 值):
        """名称: API|RAG|LoRA|记忆|策略；值: True/False 或 A/B/C"""
        with self._锁:
            if 名称 == "API":
                self.启用API = bool(值)
            elif 名称 == "RAG":
                self.启用RAG = True if 值 else False
            elif 名称 == "LoRA":
                self.启用LoRA = True if 值 else False
            elif 名称 == "记忆":
                self.启用记忆 = bool(值)
            elif 名称 == "策略":
                if 值 not in ("A", "B", "C"):
                    raise ValueError("动态策略必须是 A/B/C")
                self.动态策略 = 值
            else:
                raise ValueError(f"未知开关: {名称}")
        return self.值()

    # 实际生效值（开关优先，否则跟随模型注册配置）
    def RAG实际(self, 描述):
        if self.启用RAG is None:
            return bool(描述["rag"])
        return self.启用RAG

    def LoRA实际(self, 描述):
        if self.启用LoRA is None:
            return bool(描述["lora"])
        return self.启用LoRA

    def 策略实际(self, 描述):
        if self.动态策略 is None:
            return 描述["动态策略"]
        return self.动态策略

    def 记忆实际(self):
        return self.启用记忆


开关 = 开关管理()
