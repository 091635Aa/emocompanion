# -*- coding: utf-8 -*-
"""
监控 — 运行时监控（大屏数据源）
================================
聚合：日志环形缓冲（最近 300 条）、生成记录（token 数 / token每秒 /
稳定度 / 情感统计）、GPU 显存、系统内存（ctypes，零依赖）。
全局唯一实例：监控 = 监控器()
"""
import os
import time
import ctypes
import threading
from collections import deque
from datetime import datetime


class _内存状态(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


# 情感维度 → 常见提示词（轻量本地映射，避免重依赖导入；与 echo_common.测试提示词 一致）
测试提示词表 = {
    "开心": ["你今天真好看", "终于等到你了，我好开心", "今天的中标消息让我兴奋得睡不着"],
    "悲伤": ["一切都结束了", "他走了，再也不会回来了", "我好像再也找不到活下去的意义了"],
    "愤怒": ["你凭什么这么说我", "这个结果简直是荒谬至极", "我受够了你们的欺骗和背叛"],
    "中性": ["今天天气不错", "我想了解一下这个产品的功能", "请问地铁站怎么走"],
    "复杂混合": ["虽然赢了比赛，但我最好的朋友受伤了", "我爱我的工作，但是工资真的太低了",
                 "你给了我这么多帮助，我却没办法回报你"],
}


def 推断情感维度(提示词):
    for 维度, 列表 in 测试提示词表.items():
        if 提示词 in 列表:
            return 维度
    return "待定"


class 监控器:
    def __init__(self, 日志上限=300):
        self._锁 = threading.Lock()
        self.日志 = deque(maxlen=日志上限)
        self.生成记录 = deque(maxlen=100)   # {时间, 步数, 耗时, 重复率, 情感命中率, 情感密度, 情感维度, 提示词}
        self.总token数 = 0
        self.服务启动时间 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.gpu忙碌 = False       # 模型加载/生成/卸载期间置 True，避免并发 torch 调用崩溃

    # ── 日志 ──
    def 记录日志(self, 消息, 级别="INFO"):
        with self._锁:
            self.日志.append({
                "时间": datetime.now().strftime("%H:%M:%S"),
                "级别": 级别,
                "消息": str(消息)[:300],
            })

    def 取日志(self, 数量=200, 过滤=None):
        with self._锁:
            列表 = list(self.日志)
        列表 = 列表[-数量:]
        if 过滤:
            列表 = [x for x in 列表 if 过滤 in x["消息"] or 过滤 in x["级别"]]
        return 列表[::-1]  # 最新在前

    # ── 生成统计 ──
    def 记录生成(self, 提示词, 结果):
        """记录一次生成：token 数、耗时、重复率、情感指标、维度"""
        步数 = int(结果.get("步数") or 0)
        耗时 = float(结果.get("耗时") or 0)
        with self._锁:
            self.总token数 += 步数
            self.生成记录.append({
                "时间": datetime.now().strftime("%H:%M:%S"),
                "步数": 步数,
                "耗时": 耗时,
                "重复率": 结果.get("重复率", 0.0),
                "情感命中率": 结果.get("情感命中率", 0.0),
                "情感密度": (结果.get("动态信息") or {}).get("情感密度"),
                "情感维度": 推断情感维度(提示词),
                "提示词": str(提示词)[:40],
            })

    def token每秒(self, 窗口=10):
        """滑动窗口内 token/秒（最近 窗口 条生成）"""
        with self._锁:
            记录 = list(self.生成记录)[-窗口:]
        if not 记录:
            return 0.0
        总token = sum(r["步数"] for r in 记录)
        总耗时 = sum(r["耗时"] for r in 记录)
        if 总耗时 <= 0:
            return 0.0
        return round(总token / 总耗时, 2)

    def 稳定度(self):
        """稳定度：基于最近 10 条生成的平均重复率 → 100*(1-重复率)，夹取 [0,100]"""
        with self._锁:
            记录 = list(self.生成记录)[-10:]
        if not 记录:
            return 100.0
        平均重复率 = sum(r["重复率"] for r in 记录) / len(记录)
        return round(max(0.0, min(100.0, 100 * (1 - 平均重复率))), 1)

    def 情感统计(self):
        """最近记录的情感维度分布 + 最近一次 命中率/密度"""
        with self._锁:
            记录 = list(self.生成记录)
        分布 = {}
        for r in 记录:
            分布[r["情感维度"]] = 分布.get(r["情感维度"], 0) + 1
        最近 = 记录[-1] if 记录 else None
        return {
            "维度分布": 分布,
            "最近命中率": round(最近["情感命中率"], 4) if 最近 else 0.0,
            "最近密度": round(最近["情感密度"], 4) if 最近 and 最近["情感密度"] is not None else None,
            "最近维度": 最近["情感维度"] if 最近 else None,
            "样本数": len(记录),
        }

    # ── 资源占用 ──
    def 记录GPU忙碌(self, 忙碌):
        """模型加载/生成/卸载期间置 True，阻止其他线程查询显存（防原生崩溃）"""
        with self._锁:
            self.gpu忙碌 = bool(忙碌)

    def 显存(self):
        if self.gpu忙碌:
            return None
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "显存MB": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
                    "总显存MB": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 0),
                    "模型名": torch.cuda.get_device_name(0),
                }
        except Exception:
            pass
        return None

    def 系统内存(self):
        """Windows GlobalMemoryStatusEx → {总MB, 可用MB, 占用%}"""
        try:
            s = _内存状态()
            s.dwLength = ctypes.sizeof(_内存状态)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
                总 = s.ullTotalPhys / 1024 / 1024
                可用 = s.ullAvailPhys / 1024 / 1024
                return {"总MB": round(总, 0), "可用MB": round(可用, 0),
                        "占用%": int(s.dwMemoryLoad)}
        except Exception:
            pass
        return None

    # ── 汇总 ──
    def 摘要(self):
        显存 = self.显存()
        内存 = self.系统内存()
        return {
            "服务启动时间": self.服务启动时间,
            "总token数": self.总token数,
            "生成次数": len(self.生成记录),
            "token每秒": self.token每秒(),
            "稳定度": self.稳定度(),
            "情感统计": self.情感统计(),
            "显存": 显存,
            "系统内存": 内存,
            "日志尾部": self.取日志(数量=30),
        }


监控 = 监控器()
