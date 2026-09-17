# -*- coding: utf-8 -*-
"""
流程编排 — 全流程状态机 + 生成历史
====================================
节点固定顺序：
    ① 注册 → ② 解析 → ③ 适配 → ④ 加载 → ⑤ 生成 → ⑥ 测试 → ⑦ [定制]打标
每节点含 {状态(待执行|执行中|完成|失败|跳过), 开始时间, 耗时, 详情}。
全局唯一实例：流程 = 流程编排()
"""
import time
import threading
from datetime import datetime

节点顺序 = ["注册", "解析", "适配", "加载", "生成", "测试", "打标"]


def _当前时间():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class 流程编排:
    """全流程状态机：节点状态 + 生成历史（最近 100 条）"""

    def __init__(self):
        self._锁 = threading.Lock()
        self._节点 = {}
        self._生成历史 = []
        self.当前流程 = "标准"  # 标准 | 定制
        self.当前模型名 = None
        self.重置()

    def 重置(self, 流程="标准", 模型名=None):
        """清空节点状态（注册新模型/切换流程时调用）"""
        with self._锁:
            self.当前流程 = 流程
            if 模型名 is not None:
                self.当前模型名 = 模型名
            for 名 in 节点顺序:
                self._节点[名] = {
                    "节点": 名, "状态": "待执行",
                    "开始时间": None, "耗时": None, "详情": "",
                }
            # 定制流程默认标记 打标 节点为可执行（标准流程打标节点执行时自动切换流程）
            if 流程 == "定制":
                self._节点["打标"]["详情"] = "定制模型流程包含打标环节"

    def 开始(self, 名):
        """标记节点进入执行中"""
        with self._锁:
            if 名 in self._节点:
                self._节点[名]["状态"] = "执行中"
                self._节点[名]["开始时间"] = _当前时间()
                self._节点[名]["耗时"] = None
                self._节点[名]["详情"] = ""

    def 完成(self, 名, 详情=""):
        """标记节点完成并记录耗时"""
        with self._锁:
            if 名 not in self._节点:
                return
            开始 = self._节点[名]["开始时间"]
            耗时 = None
            if 开始:
                try:
                    t0 = datetime.strptime(开始, "%Y-%m-%d %H:%M:%S")
                    耗时 = round((datetime.now() - t0).total_seconds(), 2)
                except Exception:
                    pass
            self._节点[名]["状态"] = "完成"
            self._节点[名]["耗时"] = 耗时
            if 详情:
                self._节点[名]["详情"] = 详情

    def 失败(self, 名, 详情=""):
        with self._锁:
            if 名 in self._节点:
                self._节点[名]["状态"] = "失败"
                self._节点[名]["详情"] = 详情 or "执行失败"

    def 跳过(self, 名, 详情=""):
        with self._锁:
            if 名 in self._节点:
                self._节点[名]["状态"] = "跳过"
                self._节点[名]["详情"] = 详情 or "当前流程不需要"

    def 记录生成(self, 条目):
        """追加一条生成记录（保留最近 100 条）"""
        with self._锁:
            self._生成历史.append(条目)
            self._生成历史 = self._生成历史[-100:]

    def 获取(self):
        """返回节点列表 + 生成历史 + 当前流程/模型"""
        with self._锁:
            return {
                "当前流程": self.当前流程,
                "当前模型名": self.当前模型名,
                "节点": [dict(v) for v in self._节点.values()],
                "生成历史": list(self._生成历史),
            }


流程 = 流程编排()
