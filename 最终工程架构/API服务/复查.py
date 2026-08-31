# -*- coding: utf-8 -*-
"""
复查 — 打标二次复查系统
========================
打标工具输出 标注任务_{批次名}.json/.csv 后，人工在此复查：
- 修改 情感维度 / 质量评分(1-5) / 标注状态(待标注|已标注|返工) / 复查备注
- 保存回写 JSON + CSV（保留 平均熵/重复率/情感命中率/λ/γ/τ/模型 等原始字段）
- 复查统计：各状态数量 / 复查进度
全局唯一实例：复查 = 复查系统()
"""
import os
import csv
import json
import glob
import threading
from datetime import datetime

打标输出目录 = r"f:\打标"

状态选项 = ["待标注", "已标注", "返工"]


class 复查系统:
    def __init__(self):
        self._锁 = threading.Lock()
        self.当前批次 = None

    # ── 任务发现 ──
    def 任务列表(self):
        """f:\打标 下 标注任务_*.json 清单（含复查统计）"""
        列表 = []
        for 文件 in sorted(glob.glob(os.path.join(打标输出目录, "标注任务_*.json"))):
            批次名 = os.path.basename(文件)[len("标注任务_"):-5]
            try:
                with open(文件, encoding="utf-8") as f:
                    数据 = json.load(f)
                列表.append({
                    "批次名": 批次名,
                    "路径": 文件,
                    "条目数": 数据.get("条目数", len(数据.get("条目", []))),
                    "时间戳": 数据.get("时间戳"),
                    **self._统计(数据.get("条目", [])),
                })
            except Exception as e:
                列表.append({"批次名": 批次名, "路径": 文件, "条目数": 0, "错误": str(e)})
        return 列表

    def _统计(self, 条目列表):
        from collections import Counter
        c = Counter(条.get("标注状态", "待标注") for 条 in 条目列表)
        总 = max(1, len(条目列表))
        return {
            "待标注数": c.get("待标注", 0),
            "已标注数": c.get("已标注", 0),
            "返工数": c.get("返工", 0),
            "复查进度%": round(c.get("已标注", 0) / 总 * 100, 1),
        }

    # ── 读取 / 保存 ──
    def 加载批次(self, 批次名):
        """读取 标注任务_{批次名}.json，返回完整数据"""
        路径 = os.path.join(打标输出目录, f"标注任务_{批次名}.json")
        if not os.path.exists(路径):
            raise ValueError(f"标注任务不存在: {批次名}")
        with open(路径, encoding="utf-8") as f:
            数据 = json.load(f)
        self.当前批次 = 批次名
        return 数据

    def 保存批次(self, 批次名, 条目列表):
        """人工复查后回写 JSON + CSV（保留原始指标字段）"""
        if not 条目列表:
            raise ValueError("条目列表为空")
        路径 = os.path.join(打标输出目录, f"标注任务_{批次名}.json")
        if not os.path.exists(路径):
            raise ValueError(f"标注任务不存在: {批次名}")
        数据 = json.load(open(路径, encoding="utf-8"))
        数据["条目"] = 条目列表
        数据["条目数"] = len(条目列表)
        数据["复查时间"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(路径, "w", encoding="utf-8") as f:
            json.dump(数据, f, ensure_ascii=False, indent=2)

        # CSV 回写（utf-8-sig，Excel 可读）
        表头 = ["提示词", "情感维度", "质量评分", "标注状态", "平均熵", "重复率",
                "情感命中率", "λ", "γ", "τ", "回复文本", "模型", "量化", "时间戳",
                "复查备注", "复查时间"]
        csv路径 = os.path.join(打标输出目录, f"标注任务_{批次名}.csv")
        with open(csv路径, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=表头, extrasaction="ignore")
            w.writeheader()
            for 条 in 条目列表:
                w.writerow({k: 条.get(k, "") for k in 表头})
        return {"已保存": True, "条目数": len(条目列表), "路径": [路径, csv路径]}

    def 统计(self, 批次名):
        数据 = self.加载批次(批次名)
        return {"批次名": 批次名, "条目数": len(数据.get("条目", [])),
                **self._统计(数据.get("条目", []))}


复查 = 复查系统()
