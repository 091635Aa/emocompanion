# -*- coding: utf-8 -*-
"""
记忆外挂 — 跨会话长期记忆（读写 / 检索 / 注入）
==============================================
来源：f:\\最终工程架构\\API服务\\记忆.py（参考适配，改为按角色分文件存储）

- 记忆库目录：<项目根>/数据/记忆库（可自定义目录）
- 落盘格式：<角色名>_记忆_<时间戳>.json（含 时间戳/内容/标签/角色名）
- 目录不存在时自动创建；每条记忆独立文件，防并发写坏
- 检索：jieba 分词（可用则用）n-gram 相似度 + 标签/角色匹配加权
- 注入：检索结果拼 [长期记忆] 前缀，供生成时注入提示词（跨会话生效）
"""

import os
import json
import threading
from datetime import datetime

from ..配置管理 import 解析路径

# 默认记忆库目录（惰性解析，指向项目根 数据/记忆库）
_默认记忆库目录 = None


def _获取默认目录() -> str:
    global _默认记忆库目录
    if _默认记忆库目录 is None:
        _默认记忆库目录 = 解析路径("数据/记忆库")
    return _默认记忆库目录


class 记忆外挂:
    """跨会话记忆外挂：读写记忆文件、检索相关记忆、构建注入前缀。"""

    def __init__(self, 目录: str = None) -> None:
        """
        参数:
            目录: 记忆库目录（绝对路径）；为 None 时使用 <项目根>/数据/记忆库。
        """
        self.目录 = 目录 or _获取默认目录()
        self._锁 = threading.Lock()
        self.写入数 = 0
        self.检索数 = 0
        # 目录不存在自动创建
        os.makedirs(self.目录, exist_ok=True)

    # ──────────────────────────────────────────────
    # 读写
    # ──────────────────────────────────────────────

    def _全部记忆(self, 角色名: str = None) -> list:
        """读取库内全部记忆条目（可选按角色名过滤），按时间戳排序。"""
        列表 = []
        try:
            文件名列表 = os.listdir(self.目录)
        except OSError:
            return 列表
        for 文件名 in 文件名列表:
            if not (文件名.endswith(".json") and "_记忆_" in 文件名):
                continue
            try:
                with open(os.path.join(self.目录, 文件名), "r", encoding="utf-8") as f:
                    条目 = json.load(f)
                if 角色名 and 条目.get("角色名") != 角色名:
                    continue
                列表.append(条目)
            except Exception:
                continue
        列表.sort(key=lambda x: x.get("时间戳", ""))
        return 列表

    def 添加记忆(self, 角色名: str, 内容: str, 标签: str = "") -> str:
        """写入一条记忆，落盘为 <角色名>_记忆_<时间戳>.json。

        参数:
            角色名: 角色名称（用于文件命名与按角色过滤）。
            内容: 记忆正文。
            标签: 可选标签（如情感维度/关键词）。

        返回:
            str：写入文件的绝对路径。
        """
        时间戳 = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        条目 = {
            "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "内容": str(内容),
            "标签": str(标签),
            "角色名": str(角色名),
        }
        文件名 = f"{角色名}_记忆_{时间戳}.json"
        文件路径 = os.path.join(self.目录, 文件名)
        with self._锁:
            with open(文件路径, "w", encoding="utf-8") as f:
                json.dump(条目, f, ensure_ascii=False, indent=2)
            self.写入数 += 1
        return 文件路径

    def 读取全部(self) -> list:
        """返回全部记忆条目（list[dict]）。"""
        return self._全部记忆()

    # ──────────────────────────────────────────────
    # 检索
    # ──────────────────────────────────────────────

    def _分词(self, 文本) -> list:
        """jieba 可用时用 jieba 分词；否则用字符 2-gram 兜底。"""
        try:
            import jieba
            return [词 for 词 in jieba.cut(str(文本)) if 词.strip()]
        except ImportError:
            文本 = str(文本)
            return [文本[i:i + 2] for i in range(max(0, len(文本) - 1))]

    def _相似度(self, a, b) -> float:
        """Jaccard 相似度（基于分词集合）。"""
        A, B = set(self._分词(a)), set(self._分词(b))
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def 检索相关(self, 查询: str, 前N: int = 5, 角色名: str = None) -> list:
        """按 内容相似度 + 标签命中 + 角色匹配 加权检索相关记忆。

        参数:
            查询: 用户当前输入（提示词）。
            前N: 返回条数上限。
            角色名: 可选，仅检索该角色的记忆。

        返回:
            list[dict]：按相关度降序的记忆条目。
        """
        self.检索数 += 1
        全部 = self._全部记忆(角色名=角色名)
        if not 全部:
            return []
        计分 = []
        for m in 全部:
            相似 = self._相似度(查询, m.get("内容", ""))
            标签分 = 0.3 if (m.get("标签") and str(m.get("标签")) in str(查询)) else 0.0
            角色分 = 0.2 if (角色名 and m.get("角色名") == 角色名) else 0.0
            得分 = 相似 + 标签分 + 角色分
            if 得分 > 0:
                计分.append((得分, m))
        计分.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in 计分[:前N]]

    def 构建前缀(self, 查询: str, 前N: int = 5, 角色名: str = None) -> str:
        """检索相关记忆并拼接 [长期记忆] 前缀；无结果返回空串（跨会话注入用）。"""
        结果 = self.检索相关(查询, 前N=前N, 角色名=角色名)
        if not 结果:
            return ""
        行 = []
        for m in 结果:
            角色 = m.get("角色名", "")
            标签 = m.get("标签", "")
            行.append(f"- [{角色}] {m.get('内容', '')}（标签：{标签}）")
        return "[长期记忆]\n" + "\n".join(行) + "\n"

    # ──────────────────────────────────────────────
    # 状态
    # ──────────────────────────────────────────────

    def 状态(self) -> dict:
        """返回记忆库状态概览。"""
        全部 = self._全部记忆()
        角色分布 = {}
        for m in 全部:
            r = m.get("角色名", "")
            角色分布[r] = 角色分布.get(r, 0) + 1
        return {
            "目录": self.目录,
            "总条数": len(全部),
            "写入数": self.写入数,
            "检索数": self.检索数,
            "角色分布": 角色分布,
        }
