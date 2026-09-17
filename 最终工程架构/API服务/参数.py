# -*- coding: utf-8 -*-
"""
参数管理 — 生成参数直观调整（λ / γ / τ 覆盖）
================================================
覆盖推荐参数（扫描表/公式自动算出的基准值），实现前端可调：
- λ 回响注入强度：调大更有"灵性"但易重复坍缩，调小更稳定但可能平庸
- γ 回响池衰减  ：调大历史影响消退快，调小长程记忆更久
- τ 情感筛选阈值：调大只保留高情感词，调小保留更多词
None = 跟随推荐值。全局唯一实例：参数 = 参数管理()
"""
import threading


class 参数管理:
    def __init__(self):
        self._锁 = threading.Lock()
        self.覆盖 = {"λ": None, "γ": None, "τ": None}

    def 状态(self):
        """当前覆盖状态"""
        with self._锁:
            return dict(self.覆盖)

    def 设(self, 名称, 值):
        """设置覆盖值；传 None 恢复跟随推荐"""
        with self._锁:
            if 名称 not in self.覆盖:
                raise ValueError(f"未知参数: {名称}（可选 λ/γ/τ）")
            if 值 is None:
                self.覆盖[名称] = None
            else:
                v = float(值)
                if not (0.0 < v <= 1.0):
                    raise ValueError(f"{名称} 取值范围 (0, 1]")
                self.覆盖[名称] = round(v, 6)
        return dict(self.覆盖)

    def 重置(self):
        with self._锁:
            self.覆盖 = {"λ": None, "γ": None, "τ": None}
        return dict(self.覆盖)

    def 应用到框架(self, 框架, 推荐):
        """把覆盖值应用到已加载框架（覆盖优先，否则用推荐基准）"""
        if 框架 is None:
            return
        覆盖 = self.状态()
        框架.λ基准 = 覆盖["λ"] if 覆盖["λ"] is not None else 推荐["λ"]
        框架.γ基准 = 覆盖["γ"] if 覆盖["γ"] is not None else 推荐["γ"]
        框架.τ基准 = 覆盖["τ"] if 覆盖["τ"] is not None else 推荐["τ"]


参数 = 参数管理()
