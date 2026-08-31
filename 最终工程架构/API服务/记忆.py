# -*- coding: utf-8 -*-
"""
超长期记忆系统 — 跨会话长期记忆（写入 / 检索 / 注入）
======================================================
- 记忆库：数据\\记忆库\\记忆_{时间戳}.json（每条独立文件，防并发写坏）
- 每条记忆：{id, 时间, 提示词, 回复, 情感维度, 情感命中率, 情感密度,
            平均熵, 重复率, 摘要}
- 检索：情感维度命中 + 字符 n-gram 相似度加权（零重依赖，jieba 缺失可运行）
- 注入：检索结果拼 [长期记忆] 前缀传入 推理框架.生成(前缀=...)
全局唯一实例：记忆 = 记忆系统()
"""
import os
import json
import time
import threading
from datetime import datetime

数据目录 = r"f:\最终工程架构\数据"
记忆库目录 = os.path.join(数据目录, "记忆库")


def 推断情感维度(提示词):
    """轻量维度映射（与 echo_common.测试提示词 一致），避免重依赖导入"""
    测试提示词表 = {
        "开心": ["你今天真好看", "终于等到你了，我好开心", "今天的中标消息让我兴奋得睡不着"],
        "悲伤": ["一切都结束了", "他走了，再也不会回来了", "我好像再也找不到活下去的意义了"],
        "愤怒": ["你凭什么这么说我", "这个结果简直是荒谬至极", "我受够了你们的欺骗和背叛"],
        "中性": ["今天天气不错", "我想了解一下这个产品的功能", "请问地铁站怎么走"],
        "复杂混合": ["虽然赢了比赛，但我最好的朋友受伤了", "我爱我的工作，但是工资真的太低了",
                     "你给了我这么多帮助，我却没办法回报你"],
    }
    for 维度, 列表 in 测试提示词表.items():
        if 提示词 in 列表:
            return 维度
    return "待定"


def _ngram(文本, n=2):
    """字符 n-gram 集合，用于相似度计算"""
    文本 = str(文本).strip()
    return {文本[i:i + n] for i in range(max(0, len(文本) - n + 1))}


def _相似度(a, b):
    """Jaccard n-gram 相似度"""
    A, B = _ngram(a), _ngram(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


class 记忆系统:
    def __init__(self):
        self._锁 = threading.Lock()
        self._缓存 = None   # 记忆列表缓存（文件过多时加速检索）
        self.写入数 = 0
        self.检索数 = 0
        os.makedirs(记忆库目录, exist_ok=True)

    def _全部记忆(self, 强制=False):
        if self._缓存 is not None and not 强制:
            return self._缓存
        列表 = []
        for 文件 in os.listdir(记忆库目录):
            if 文件.startswith("记忆_") and 文件.endswith(".json"):
                try:
                    with open(os.path.join(记忆库目录, 文件), encoding="utf-8") as f:
                        列表.append(json.load(f))
                except Exception:
                    continue
        列表.sort(key=lambda x: x.get("时间", ""))
        self._缓存 = 列表
        return 列表

    def 记录(self, 提示词, 结果):
        """写入一条记忆（回复为空则不写）"""
        文本 = (结果.get("文本") or "").strip()
        if not 文本:
            return None
        条目 = {
            "id": f"记忆_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}",
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "提示词": str(提示词)[:200],
            "回复": 文本[:500],
            "情感维度": 推断情感维度(提示词),
            "情感命中率": round(结果.get("情感命中率", 0.0), 4),
            "情感密度": round((结果.get("动态信息") or {}).get("情感密度") or 0.0, 4),
            "平均熵": round(结果.get("平均熵", 0.0), 4),
            "重复率": round(结果.get("重复率", 0.0), 4),
            "摘要": 文本[:80],
        }
        with self._锁:
            输出路径 = os.path.join(记忆库目录, f"{条目['id']}.json")
            with open(输出路径, "w", encoding="utf-8") as f:
                json.dump(条目, f, ensure_ascii=False, indent=2)
            self.写入数 += 1
            self._缓存 = None
        return 条目

    def 检索(self, 提示词, top_k=3):
        """按 情感维度命中 + n-gram 相似度 加权检索 top-k 记忆"""
        self.检索数 += 1
        记忆列表 = self._全部记忆()
        if not 记忆列表:
            return []
        维度 = 推断情感维度(提示词)
        计分 = []
        for m in 记忆列表:
            相似 = _相似度(提示词, m["提示词"])
            维度分 = 0.5 if m["情感维度"] == 维度 else 0.0
            质量 = max(0.0, 1.0 - m["重复率"]) * 0.3
            得分 = 相似 * 1.0 + 维度分 + 质量
            if 得分 > 0:
                计分.append((得分, m))
        计分.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in 计分[:top_k]]

    def 构建前缀(self, 提示词, top_k=3):
        """检索并拼接 [长期记忆] 前缀；无结果返回空串"""
        记忆列表 = self.检索(提示词, top_k=top_k)
        if not 记忆列表:
            return ""
        行 = []
        for m in 记忆列表:
            行.append(f"- [{m['情感维度']}] {m['提示词']} → {m['摘要']}")
        return "[长期记忆]" + "\n".join(行) + "\n"

    def 状态(self):
        记忆列表 = self._全部记忆()
        分布 = {}
        for m in 记忆列表:
            分布[m["情感维度"]] = 分布.get(m["情感维度"], 0) + 1
        return {
            "目录": 记忆库目录,
            "总条数": len(记忆列表),
            "写入数": self.写入数,
            "检索数": self.检索数,
            "维度分布": 分布,
        }

    def 列表(self, 数量=50):
        return self._全部记忆()[-数量:][::-1]

    def 清除(self):
        """清空记忆库（保留目录）"""
        with self._锁:
            for 文件 in os.listdir(记忆库目录):
                if 文件.endswith(".json"):
                    try:
                        os.remove(os.path.join(记忆库目录, 文件))
                    except Exception:
                        pass
            self._缓存 = []
            self.写入数 = 0
        return True


记忆 = 记忆系统()
