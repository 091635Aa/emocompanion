# -*- coding: utf-8 -*-
"""
打标服务 — 定制模型流程：生成回复批次 + 输出标注任务
====================================================
复用已加载模型（避免二次加载占显存）逐条生成，收集
{提示词, 回复文本, 平均熵, 重复率, 情感命中率, λ, γ, τ}，
调用 打标工具.输出标注任务 写出 f:\\打标\\标注任务_{批次名}.json/.csv。
后台线程执行，前端轮询 打标状态()。
"""
import sys
import os
import time
import threading
from datetime import datetime

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

import 打标工具
from 模型管理 import 管理器
from 流程编排 import 流程
from 监控 import 监控

数据目录 = r"f:\最终工程架构\数据"
默认提示词文件 = os.path.join(数据目录, "全流程_提示词.txt")
打标输出目录 = r"f:\打标"


class 打标服务:
    def __init__(self):
        self._锁 = threading.Lock()
        self.运行中 = False
        self.批次名 = None
        self.进度 = 0
        self.总数 = 0
        self.当前提示 = ""
        self.结果 = None   # {状态, 输出文件, 错误, 完成时间}
        self.历史 = []     # 最近 20 个已完成任务

    def 启动(self, 模型名, 提示词集=None, 批次名="批次1", 最大token=128):
        """启动后台打标；提示词集缺省时读取 数据\\全流程_提示词.txt"""
        if self.运行中:
            raise RuntimeError("已有打标任务进行中")
        if 管理器.加载状态 != "已加载":
            raise RuntimeError("模型未加载，请先加载模型")
        if 模型名 and 管理器.已加载模型名 != 模型名:
            raise RuntimeError(f"当前加载模型为 {管理器.已加载模型名}，与请求模型不一致")
        if not 提示词集:
            if not os.path.exists(默认提示词文件):
                raise RuntimeError(f"提示词文件不存在: {默认提示词文件}")
            提示词集 = 打标工具.读取提示词文件(默认提示词文件)
        提示词集 = [str(p).strip() for p in 提示词集 if str(p).strip()]
        if not 提示词集:
            raise RuntimeError("提示词集为空")

        self.运行中 = True
        self.批次名 = 批次名
        self.进度 = 0
        self.总数 = len(提示词集)
        self.当前提示 = ""
        self.结果 = None
        线程 = threading.Thread(
            target=self._打标线程,
            args=(模型名, 提示词集, 批次名, int(最大token)),
            daemon=True)
        线程.start()
        return {"总数": self.总数, "批次名": 批次名}

    def _打标线程(self, 模型名, 提示词集, 批次名, 最大token):
        流程.开始("打标")
        try:
            结果列表 = []
            for i, 提示 in enumerate(提示词集):
                self.当前提示 = 提示
                self.进度 = i
                print(f"[打标服务] {i + 1}/{len(提示词集)}: {提示[:30]}")
                try:
                    结果 = 管理器.生成(模型名, 提示, 最大token=最大token)
                    结果列表.append({
                        "提示词": 提示,
                        "回复文本": 结果["文本"],
                        "平均熵": round(结果["平均熵"], 4),
                        "重复率": round(结果["重复率"], 4),
                        "情感命中率": round(max(0.0, min(1.0, 结果["情感命中率"])), 4),
                        "λ": 结果["λ"], "γ": 结果["γ"], "τ": 结果["τ"],
                    })
                except Exception as e:
                    print(f"[打标服务] 生成失败: {e}")
                    结果列表.append({
                        "提示词": 提示, "回复文本": f"[生成失败] {e}",
                        "平均熵": 0.0, "重复率": 0.0, "情感命中率": 0.0,
                        "λ": 0.0, "γ": 0.0, "τ": 0.0,
                    })
                self.进度 = i + 1

            json路径, csv路径 = 打标工具.输出标注任务(
                结果列表, 输出目录=打标输出目录, 批次名=批次名,
                模型路径=管理器.获取描述(模型名)["路径"] if 管理器.获取描述(模型名) else "",
                量化=管理器.获取描述(模型名)["量化"] if 管理器.获取描述(模型名) else None)
            self.结果 = {
                "状态": "完成", "输出文件": [json路径, csv路径],
                "完成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "条目数": len(结果列表),
            }
            self.历史.append({"批次名": 批次名, "条目数": len(结果列表),
                              "完成时间": self.结果["完成时间"],
                              "输出文件": [json路径, csv路径]})
            self.历史 = self.历史[-20:]
            流程.完成("打标", f"{len(结果列表)} 条 → {json路径}")
            监控.记录日志(f"打标完成 {批次名} | {len(结果列表)} 条 → {json路径}")
        except Exception as e:
            self.结果 = {"状态": "失败", "输出文件": [], "错误": str(e)}
            self.历史.append({"批次名": 批次名, "条目数": 0, "错误": str(e),
                              "完成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            self.历史 = self.历史[-20:]
            流程.失败("打标", str(e))
            监控.记录日志(f"打标失败 {批次名}: {e}", "ERROR")
            print(f"[打标服务] 失败: {e}")
        finally:
            self.运行中 = False

    def 状态(self):
        return {
            "运行中": self.运行中,
            "批次名": self.批次名,
            "进度": self.进度,
            "总数": self.总数,
            "当前提示": self.当前提示,
            "结果": self.结果,
            "输出目录": 打标输出目录,
            "历史": list(self.历史),
        }


打标 = 打标服务()
