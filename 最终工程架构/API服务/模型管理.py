# -*- coding: utf-8 -*-
"""
模型管理 — 模型注册 / 解析 / 加载 / 卸载 / 生成 / 状态（单例）
================================================================
- 注册模型：校验路径 → 解析 config.json（hidden_size/vocab_size）→
  自动计算推荐参数（自适应匹配.推荐参数）→ 生成模型描述文件（模型文件生成）
  数据\\模型库\\{模型名}.json
- 加载模型：后台线程实例化 推理框架（自动完成 加载+推荐参数+LoRA 挂载），
  加载成功后可配置自动触发测试（通过 自动测试回调 注入，避免循环导入）
- 单 GPU 串行：加载/卸载/生成 共用一把锁（非阻塞获取，忙碌即抛 忙碌异常）
"""
import sys
import os
import json
import time
import threading
from datetime import datetime

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

import 推理框架 as 推理框架模块
import 自适应匹配
from 流程编排 import 流程
from 监控 import 监控
from 开关 import 开关
from 记忆 import 记忆
from 参数 import 参数

数据目录 = r"f:\最终工程架构\数据"
模型库目录 = os.path.join(数据目录, "模型库")
模型空间目录 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"

量化映射 = {"fp16": None, "4bit": "4bit"}
归一化基准表 = 推理框架模块.归一化基准表  # hidden_dim → 基准


class 忙碌异常(Exception):
    """GPU 忙碌（正在加载/生成/卸载）"""
    pass


class 模型管理器:
    def __init__(self):
        self._锁 = threading.Lock()       # 保护 GPU 串行操作
        self.框架 = None                  # 已加载的 推理框架 实例
        self.已加载模型名 = None
        self.加载状态 = "未加载"          # 未加载|加载中|已加载|失败
        self.加载错误 = None
        self.加载耗时 = None
        self.加载开始时间 = None
        self.生成中 = False
        self.自动测试回调 = None          # 由 主程序 注入：lambda 模型名: 激活测试(...)
        self.推荐参数 = None              # 已加载模型的推荐基准 {λ, γ, τ}
        os.makedirs(模型库目录, exist_ok=True)

    # ═══════════════════════════════════════════
    # 模型信息解析与扫描
    # ═══════════════════════════════════════════
    def 解析模型信息(self, 路径):
        """读取 config.json → {hidden_size, vocab_size}"""
        config路径 = os.path.join(路径, "config.json")
        if not os.path.isdir(路径) or not os.path.exists(config路径):
            raise ValueError(f"无效模型路径（缺少 config.json）: {路径}")
        with open(config路径, encoding="utf-8") as f:
            配置 = json.load(f)
        hidden_size = int(配置.get("hidden_size") or 配置.get("d_model") or 0)
        vocab_size = int(配置.get("vocab_size") or 配置.get("n_vocab") or 0)
        if not hidden_size:
            raise ValueError(f"无法解析 hidden_size: {路径}")
        return {"hidden_size": hidden_size, "vocab_size": vocab_size}

    def 扫描可用模型(self):
        """扫描模型空间目录 + 已注册路径 → 可用模型列表"""
        结果 = []
        已见路径 = set()
        for 目录 in [模型空间目录] + self._全部已注册路径():
            if not os.path.isdir(目录):
                continue
            for 项 in os.listdir(目录):
                模型路径 = os.path.join(目录, 项)
                if 模型路径 in 已见路径:
                    continue
                config路径 = os.path.join(模型路径, "config.json")
                if not os.path.exists(config路径):
                    continue
                try:
                    信息 = self.解析模型信息(模型路径)
                except Exception:
                    continue
                已见路径.add(模型路径)
                结果.append({
                    "模型名": 项,
                    "路径": 模型路径,
                    "hidden_size": 信息["hidden_size"],
                    "vocab_size": 信息["vocab_size"],
                    "已注册": bool(self.获取描述(项)),
                })
        结果.sort(key=lambda x: x["模型名"])
        return 结果

    def _全部已注册路径(self):
        路径集合 = set()
        for 文件 in os.listdir(模型库目录):
            if 文件.endswith(".json"):
                try:
                    with open(os.path.join(模型库目录, 文件), encoding="utf-8") as f:
                        描述 = json.load(f)
                    if 描述.get("路径"):
                        路径集合.add(os.path.dirname(描述["路径"]))
                except Exception:
                    pass
        return list(路径集合)

    # ═══════════════════════════════════════════
    # 模型注册（模型文件生成）
    # ═══════════════════════════════════════════
    def 注册模型(self, 模型名, 路径, 类型="标准", 量化="fp16", 动态策略="B",
                 rag=False, lora=None, 长上下文=False, 自动测试=True):
        """校验路径 → 解析 → 自动计算参数 → 生成模型描述文件"""
        路径 = os.path.abspath(路径)
        if not 模型名 or not 路径:
            raise ValueError("模型名与路径不能为空")
        信息 = self.解析模型信息(路径)
        hidden_size = 信息["hidden_size"]
        vocab_size = 信息["vocab_size"]

        # 自动适配：扫描表优先 / 公式兜底
        参数 = 自适应匹配.推荐参数(hidden_size, vocab_size)
        归一化基准 = 归一化基准表.get(hidden_size, 896)

        描述 = {
            "模型名": 模型名,
            "路径": 路径,
            "类型": 类型,                # 标准 | 定制
            "量化": 量化,                # fp16 | 4bit
            "动态策略": 动态策略,        # A | B | C
            "rag": bool(rag),
            "lora": lora or None,
            "长上下文": bool(长上下文),
            "自动测试": bool(自动测试),
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "参数": {
                "λ": 参数["λ"], "γ": 参数["γ"], "τ": 参数["τ"],
                "来源": 参数["来源"],
            },
            "归一化基准": 归一化基准,
            "注册时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # 覆盖写：重新注册以新配置为准
        输出路径 = os.path.join(模型库目录, f"{模型名}.json")
        with open(输出路径, "w", encoding="utf-8") as f:
            json.dump(描述, f, ensure_ascii=False, indent=2)

        # 流程状态更新（注册 → 解析 → 适配 节点）
        流程.重置(流程="定制" if 类型 == "定制" else "标准", 模型名=模型名)
        流程.完成("注册", f"模型文件已生成: {输出路径}")
        流程.完成("解析", f"hidden_size={hidden_size} vocab_size={vocab_size}")
        流程.完成("适配", f"推荐 λ={参数['λ']} γ={参数['γ']} τ={参数['τ']}（{参数['来源']}）")
        return 描述

    # ═══════════════════════════════════════════
    # 模型库读写
    # ═══════════════════════════════════════════
    def 获取描述(self, 模型名):
        输出路径 = os.path.join(模型库目录, f"{模型名}.json")
        if not os.path.exists(输出路径):
            return None
        try:
            with open(输出路径, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def 读取模型库(self):
        """返回全部已注册模型描述列表"""
        列表 = []
        for 文件 in sorted(os.listdir(模型库目录)):
            if 文件.endswith(".json"):
                描述 = self.获取描述(文件[:-5])
                if 描述:
                    列表.append(描述)
        return 列表

    # ═══════════════════════════════════════════
    # 模型加载 / 卸载（后台线程）
    # ═══════════════════════════════════════════
    def 加载(self, 模型名):
        """启动后台加载线程；返回后轮询 状态()"""
        if self.加载状态 == "加载中":
            raise 忙碌异常("模型加载中，请稍候")
        描述 = self.获取描述(模型名)
        if not 描述:
            raise ValueError(f"模型未注册: {模型名}")
        if self.加载状态 == "已加载" and self.已加载模型名 == 模型名:
            return {"已加载": True, "模型名": 模型名}
        self.加载状态 = "加载中"
        self.加载错误 = None
        self.加载耗时 = None
        self.加载开始时间 = time.time()
        监控.记录GPU忙碌(True)   # 同步置位，阻止其他线程在加载启动瞬间触碰 torch
        线程 = threading.Thread(target=self._加载线程, args=(描述,), daemon=True)
        线程.start()
        return {"已加载": False, "模型名": 模型名}

    def _加载线程(self, 描述):
        流程.开始("加载")
        监控.记录日志(f"开始加载模型: {描述['模型名']}（量化={描述['量化']}）")
        try:
            if not self._锁.acquire(blocking=False):
                self.加载状态 = "失败"
                self.加载错误 = "GPU 忙碌（其他任务进行中）"
                流程.失败("加载", self.加载错误)
                监控.记录GPU忙碌(False)
                return
            try:
                量化值 = 量化映射.get(描述["量化"], None)
                self.框架 = 推理框架模块.推理框架(
                    描述["路径"], 量化=量化值, rag=描述["rag"], lora=描述["lora"],
                    动态策略=描述["动态策略"], 长上下文=描述["长上下文"])
            finally:
                self._锁.release()
                监控.记录GPU忙碌(False)
            self.已加载模型名 = 描述["模型名"]
            self.加载状态 = "已加载"
            self.加载耗时 = round(time.time() - self.加载开始时间, 2)
            参数 = self.框架.参数来源
            流程.完成("加载", f"耗时 {self.加载耗时}s | 推荐参数来源: {参数}")
            # 记录推荐基准参数（供参数覆盖恢复）
            self.推荐参数 = {"λ": self.框架.λ基准, "γ": self.框架.γ基准, "τ": self.框架.τ基准}
            # 应用运行时开关（RAG / 动态策略 / LoRA 热切换）
            self.应用开关(描述)
            监控.记录日志(
                f"模型加载完成 {描述['模型名']} | {self.加载耗时}s | "
                f"显存 {self.显存MB() or 0} MB")
            # 自动测试（由 主程序 注入回调，避免循环导入）
            if 描述["自动测试"] and self.自动测试回调:
                try:
                    self.自动测试回调(描述["模型名"])
                except Exception as e:
                    print(f"[模型管理] 自动测试触发失败: {e}")
        except Exception as e:
            import traceback
            self.加载状态 = "失败"
            self.加载错误 = str(e)
            流程.失败("加载", str(e))
            监控.记录日志(f"模型加载失败: {e}", "ERROR")
            print(f"[模型管理] 模型加载失败: {e}")
            traceback.print_exc()

    def 加载并等待(self, 模型名, 超时=1800):
        """同步加载（OpenAI 兼容端点用）：等待后台加载完成或失败"""
        self.加载(模型名)
        开始 = time.time()
        while self.加载状态 in ("加载中",):
            if time.time() - 开始 > 超时:
                raise TimeoutError("模型加载超时")
            time.sleep(2)
        if self.加载状态 == "失败":
            raise RuntimeError(self.加载错误 or "模型加载失败")
        return self.已加载模型名

    def 卸载(self):
        """释放 GPU 显存（串行锁保护）"""
        if self.加载状态 == "加载中":
            raise 忙碌异常("模型加载中，无法卸载")
        if not self._锁.acquire(blocking=False):
            raise 忙碌异常("模型正在生成中，无法卸载")
        try:
            if self.框架 is not None:
                del self.框架
                self.框架 = None
            import torch
            if torch.cuda.is_available():
                监控.记录GPU忙碌(True)
                torch.cuda.empty_cache()
                监控.记录GPU忙碌(False)
        finally:
            self._锁.release()
        self.已加载模型名 = None
        self.加载状态 = "未加载"
        self.加载耗时 = None
        self.加载错误 = None
        监控.记录日志("模型已卸载，显存已释放")
        return True

    # ═══════════════════════════════════════════
    # 运行时开关（热切换）
    # ═══════════════════════════════════════════
    def 显存MB(self):
        try:
            import torch
            if torch.cuda.is_available():
                return round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        except Exception:
            pass
        return None

    def 应用开关(self, 描述=None):
        """把开关管理的实际值应用到已加载框架（RAG / 动态策略 / LoRA）"""
        if self.框架 is None:
            return {"已应用": False}
        描述 = 描述 or (self.获取描述(self.已加载模型名) if self.已加载模型名 else None)
        if 描述 is None:
            return {"已应用": False}
        # RAG 标志热切换（推理框架.生成 内读取 self.rag）
        self.框架.rag = 开关.RAG实际(描述)
        # 动态策略热切换（框架.生成 内读取 self.动态策略）
        self.框架.动态策略 = 开关.策略实际(描述)
        # LoRA 热挂载/卸载
        lora路径 = 描述.get("lora")
        lora开 = 开关.LoRA实际(描述) and bool(lora路径)
        is_peft = False
        try:
            import peft
            is_peft = isinstance(self.框架.model, peft.PeftModel)
        except ImportError:
            pass
        if lora开 and not is_peft:
            self.框架.挂载LoRA(lora路径)
            监控.记录日志(f"LoRA 已挂载: {lora路径}")
        elif not lora开 and is_peft:
            self.框架.卸载LoRA()
            监控.记录日志("LoRA 已卸载，恢复基座模型")
        监控.记录日志(
            f"开关已应用 | RAG={self.框架.rag} 策略={self.框架.动态策略} "
            f"LoRA={'开' if lora开 else '关'} 记忆={'开' if 开关.启用记忆 else '关'}")
        return {"已应用": True}

    def 切换开关(self, 名称, 值):
        """切换运行时开关并应用到已加载框架；名称 ∈ API|RAG|LoRA|记忆|策略"""
        开关.设(名称, 值)
        结果 = self.应用开关()
        监控.记录日志(f"开关切换: {名称}={值}")
        return {**开关.值(), **结果}

    # ═══════════════════════════════════════════
    # 参数调整（λ / γ / τ 直观覆盖）
    # ═══════════════════════════════════════════
    def 调整参数(self, 名称, 值):
        """设置参数覆盖（值=None 恢复跟随推荐）"""
        参数.设(名称, 值)
        监控.记录日志(f"参数调整: {名称}={参数.状态()[名称]}")
        return 参数.状态()

    def 重置参数(self):
        参数.重置()
        监控.记录日志("参数已重置为推荐基准")
        return 参数.状态()

    def 参数信息(self):
        """推荐基准 + 当前覆盖 + 生效值"""
        覆盖 = 参数.状态()
        推荐 = self.推荐参数 or {"λ": None, "γ": None, "τ": None}
        return {
            "推荐": 推荐,
            "覆盖": 覆盖,
            "生效": {
                "λ": 覆盖["λ"] if 覆盖["λ"] is not None else 推荐["λ"],
                "γ": 覆盖["γ"] if 覆盖["γ"] is not None else 推荐["γ"],
                "τ": 覆盖["τ"] if 覆盖["τ"] is not None else 推荐["τ"],
            },
            "说明": {
                "λ": "回响注入强度：调大更有灵性但易重复，调小更稳定但可能平庸",
                "γ": "回响池衰减：调大历史影响消退快，调小长程记忆更久",
                "τ": "情感筛选阈值：调大只保留高情感词，调小保留更多词",
            },
        }

    # ═══════════════════════════════════════════
    # 生成（标准流程核心）
    # ═══════════════════════════════════════════
    def 生成(self, 模型名, 提示词, 最大token=128):
        """完整流程生成：记忆注入 → 动态策略 → RAG → 回响，返回全量指标"""
        if not 提示词 or not str(提示词).strip():
            raise ValueError("提示词不能为空")
        if self.加载状态 != "已加载":
            raise 忙碌异常("模型未加载，请先加载模型")
        if 模型名 and self.已加载模型名 != 模型名:
            raise ValueError(f"当前加载模型为 {self.已加载模型名}，与请求模型不一致")
        if not self._锁.acquire(blocking=False):
            raise 忙碌异常("模型正在生成或加载中")
        self.生成中 = True
        监控.记录GPU忙碌(True)   # 生成期间禁止其他线程查 torch
        try:
            t0 = time.time()
            # 超长期记忆：检索注入前缀（开关开启时）
            前缀 = ""
            if 开关.启用记忆:
                前缀 = 记忆.构建前缀(提示词, top_k=3)
            # 确保开关实际值（RAG 标志/动态策略）生效
            描述 = self.获取描述(self.已加载模型名)
            if 描述:
                self.框架.rag = 开关.RAG实际(描述)
                self.框架.动态策略 = 开关.策略实际(描述)
            # 应用参数覆盖（λ/γ/τ：覆盖优先，否则推荐基准）
            if self.推荐参数:
                参数.应用到框架(self.框架, self.推荐参数)
            结果 = self.框架.生成(提示词, max_new_tokens=int(最大token), 前缀=前缀)
            结果["提示词"] = 提示词
            结果["耗时"] = round(time.time() - t0, 2)
            结果["模型"] = self.已加载模型名
            结果["记忆注入"] = bool(前缀)
            # 监控：token 统计 / 稳定度 / 情感 / 日志
            监控.记录生成(提示词, 结果)
            监控.记录日志(
                f"生成完成 [{self.已加载模型名}] {str(提示词)[:24]} → "
                f"{结果['步数']} token / {结果['耗时']}s（{监控.token每秒()} token/s）"
                f" | 重复率 {结果['重复率']:.3f} | 记忆={'开' if 前缀 else '关'}")
            # 超长期记忆：写入
            if 开关.启用记忆:
                记忆.记录(提示词, 结果)
            流程.记录生成(结果)
            流程.完成("生成", f"步数 {结果['步数']} | 耗时 {结果['耗时']}s")
            return 结果
        finally:
            self.生成中 = False
            监控.记录GPU忙碌(False)
            self._锁.release()

    # ═══════════════════════════════════════════
    # 状态汇总
    # ═══════════════════════════════════════════
    def 状态(self):
        """服务状态 + 技术参数 + GPU 显存"""
        显存信息 = 监控.显存()
        描述 = self.获取描述(self.已加载模型名) if self.已加载模型名 else None
        return {
            "加载状态": self.加载状态,
            "已加载模型名": self.已加载模型名,
            "加载耗时": self.加载耗时,
            "加载错误": self.加载错误,
            "生成中": self.生成中,
            "显存MB": 显存信息["显存MB"] if 显存信息 else None,
            "总显存MB": 显存信息["总显存MB"] if 显存信息 else None,
            "模型描述": 描述,
        }


管理器 = 模型管理器()
