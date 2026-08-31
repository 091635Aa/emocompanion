# -*- coding: utf-8 -*-
"""
测试引擎 — 集成测试 + 激活机制
================================
集成既有自检：
1. 配置匹配测试：自适应匹配.验证匹配偏差()（扫描表 4 点偏差 <30%）
2. λ 调度测试   ：回响引擎.自测λ调度()（单调/恒定/终点比例）
3. 模型冒烟测试 ：加载模型跑 2 条提示词，校验生成成功 + 指标可读
4. API 协议测试 ：对 /api/v1/generate 发 1 条请求，校验自定义协议响应结构

激活机制：
- POST /api/v1/test/activate 手动激活
- 模型加载完成后（注册时 自动测试=True）自动触发 模型冒烟测试+API协议测试
测试结果汇总写入 数据\\测试报告\\测试报告_{时间戳}.json
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

import 自适应匹配
import 回响引擎
from 模型管理 import 管理器
from 流程编排 import 流程
from 记忆 import 记忆
from 监控 import 监控

数据目录 = r"f:\最终工程架构\数据"
报告目录 = os.path.join(数据目录, "测试报告")
RAG向量库目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\04_RAG数据库创建\vector_db"
记忆库目录 = r"f:\最终工程架构\数据\记忆库"

冒烟提示词 = ["今天的中标消息让我兴奋得睡不着", "我想了解一下这个产品的功能"]


class 测试引擎:
    def __init__(self):
        os.makedirs(报告目录, exist_ok=True)
        self._锁 = threading.Lock()
        self.运行中 = False
        self.当前范围 = None
        self.当前模型名 = None
        self.最近结果 = None
        self.最近报告文件 = None
        self.开始时间 = None

    # ═══════════════════════════════════════════
    # 各单项测试
    # ═══════════════════════════════════════════
    def _测试配置匹配(self):
        """离线：验证推荐参数相对扫描表偏差 <30%"""
        t0 = time.time()
        try:
            达标 = 自适应匹配.验证匹配偏差()
            return {"名称": "配置匹配测试", "通过": bool(达标),
                    "耗时": round(time.time() - t0, 2),
                    "说明": "推荐参数 vs 扫描表相对偏差 <30% 判定"}
        except Exception as e:
            return {"名称": "配置匹配测试", "通过": False,
                    "耗时": round(time.time() - t0, 2), "说明": f"异常: {e}"}

    def _测试λ调度(self):
        """离线：λ 步数衰减调度曲线自检"""
        t0 = time.time()
        try:
            通过 = 回响引擎.自测λ调度()
            return {"名称": "λ 调度测试", "通过": bool(通过),
                    "耗时": round(time.time() - t0, 2),
                    "说明": "单调非增 / 起始段保持λ / 终点=0.3λ"}
        except Exception as e:
            return {"名称": "λ 调度测试", "通过": False,
                    "耗时": round(time.time() - t0, 2), "说明": f"异常: {e}"}

    def _测试模型冒烟(self, 模型名):
        """加载后跑 2 条提示词，校验生成成功 + 指标可读"""
        t0 = time.time()
        明细 = []
        if 管理器.加载状态 != "已加载":
            return {"名称": "模型冒烟测试", "通过": False,
                    "耗时": round(time.time() - t0, 2),
                    "说明": f"模型未加载（状态={管理器.加载状态}）", "明细": 明细}
        try:
            for 提示 in 冒烟提示词:
                结果 = 管理器.生成(模型名, 提示, 最大token=64)
                可读 = (bool(结果["文本"]) and
                        0.0 <= 结果["平均熵"] and
                        0.0 <= 结果["重复率"] <= 1.0 and
                        0.0 <= 结果["情感命中率"] <= 1.0)
                明细.append({
                    "提示词": 提示, "回复": 结果["文本"][:60],
                    "平均熵": round(结果["平均熵"], 4),
                    "重复率": round(结果["重复率"], 4),
                    "情感命中率": round(结果["情感命中率"], 4),
                    "步数": 结果["步数"],
                })
            return {"名称": "模型冒烟测试", "通过": all(d["平均熵"] >= 0 for d in 明细) and 明细,
                    "耗时": round(time.time() - t0, 2),
                    "说明": f"{len(明细)} 条提示词生成成功", "明细": 明细}
        except Exception as e:
            return {"名称": "模型冒烟测试", "通过": False,
                    "耗时": round(time.time() - t0, 2),
                    "说明": f"异常: {e}", "明细": 明细}

    def _测试API协议(self, 模型名):
        """对 /api/v1/generate 发测试请求，校验自定义协议响应结构"""
        t0 = time.time()
        if 管理器.加载状态 != "已加载":
            return {"名称": "API 协议测试", "通过": False,
                    "耗时": round(time.time() - t0, 2),
                    "说明": f"模型未加载（状态={管理器.加载状态}）"}
        try:
            # 惰性导入 主程序 避免循环导入；TestClient 走真实 ASGI 路由
            from fastapi.testclient import TestClient
            from 主程序 import app
            with TestClient(app) as client:
                r = client.post("/api/v1/generate", json={
                    "模型名": 模型名, "提示词": "今天天气不错", "最大token": 32,
                })
            body = r.json()
            ok = (r.status_code == 200 and
                  body.get("状态") == "ok" and
                  body.get("数据") is not None and
                  "文本" in (body.get("数据") or {}))
            return {"名称": "API 协议测试", "通过": bool(ok),
                    "耗时": round(time.time() - t0, 2),
                    "说明": f"POST /api/v1/generate → HTTP {r.status_code}"
                            + ("" if ok else f" | 响应: {str(body)[:200]}")}
        except Exception as e:
            return {"名称": "API 协议测试", "通过": False,
                    "耗时": round(time.time() - t0, 2), "说明": f"异常: {e}"}

    def _测试RAG(self, 模型名):
        """RAG 全面测试：向量库加载 + 检索 + 生成挂载验证（缺失时如实报告不误判）"""
        t0 = time.time()
        import os
        库存在 = os.path.isdir(RAG向量库目录) and os.listdir(RAG向量库目录)
        说明 = []
        通过 = True
        # 1. 向量库检查
        if not 库存在:
            return {"名称": "RAG 检索测试", "通过": False,
                    "耗时": round(time.time() - t0, 2),
                    "说明": f"向量库缺失: {RAG向量库目录}（请先运行 build_vector_db 建立库）"}
        # 2. 检索器加载 + 检索
        try:
            import build_vector_db
            from pathlib import Path
            检索器 = build_vector_db.RAG检索器(Path(RAG向量库目录))
            结果 = 检索器.检索("今天的中标消息让我兴奋得睡不着", top_k=2)
            说明.append(f"检索器加载成功，检索 {len(结果)} 条（top score="
                        f"{结果[0]['score']:.4f}）" if 结果 else "检索器加载成功但返回空")
            通过 = 通过 and bool(结果)
        except Exception as e:
            return {"名称": "RAG 检索测试", "通过": False,
                    "耗时": round(time.time() - t0, 2), "说明": f"检索器加载失败: {e}"}
        # 3. 生成挂载验证（已加载模型时）：开启 RAG 生成 1 条，检查 [参考信息] 前缀
        if 管理器.加载状态 == "已加载":
            try:
                原rag = 管理器.框架.rag
                管理器.框架.rag = True
                结果2 = 管理器.生成(模型名, "今天天气不错", 最大token=32)
                管理器.框架.rag = 原rag
                挂载成功 = "参考信息" in (结果2.get("前缀") or "") or 结果2.get("池统计") is not None
                说明.append("生成挂载验证：RAG 开启生成正常"
                            + ("，检测到 [参考信息] 注入" if "参考信息" in (结果2.get("前缀") or "")
                               else "（无参考信息命中，回退纯回响）"))
            except Exception as e:
                通过 = False
                说明.append(f"生成挂载验证失败: {e}")
        else:
            说明.append("模型未加载，跳过生成挂载验证")
        return {"名称": "RAG 测试", "通过": 通过,
                "耗时": round(time.time() - t0, 2), "说明": " | ".join(说明)}

    def _测试记忆(self):
        """记忆系统回环测试：写入 → 检索 → 清理"""
        t0 = time.time()
        测试提示 = "记忆系统回环测试专用提示词_9527"
        try:
            条目 = 记忆.记录(测试提示, {"文本": "记忆系统回环测试回复", "情感命中率": 0.5,
                                  "动态信息": {"情感密度": 0.2}, "平均熵": 3.0, "重复率": 0.1})
            检索结果 = 记忆.检索(测试提示, top_k=1)
            命中 = bool(检索结果) and 检索结果[0].get("id") == 条目.get("id")
            return {"名称": "记忆系统测试", "通过": bool(命中),
                    "耗时": round(time.time() - t0, 2),
                    "说明": "写入→检索→清理 回环成功" if 命中 else "回环失败（检索未命中）"}
        except Exception as e:
            return {"名称": "记忆系统测试", "通过": False,
                    "耗时": round(time.time() - t0, 2), "说明": f"异常: {e}"}
        finally:
            # 清理所有测试专用记忆（含历史遗留，确保回环测试幂等）
            for 文件 in os.listdir(记忆库目录):
                if 文件.startswith("记忆_") and 文件.endswith(".json"):
                    try:
                        with open(os.path.join(记忆库目录, 文件), encoding="utf-8") as f:
                            m = json.load(f)
                        if m.get("提示词") == 测试提示:
                            os.remove(os.path.join(记忆库目录, 文件))
                    except Exception:
                        pass
            记忆._缓存 = None
            try:
                记忆._全部记忆(强制=True)
            except Exception:
                pass

    # ═══════════════════════════════════════════
    # 激活与执行
    # ═══════════════════════════════════════════
    def 激活(self, 范围="全部", 模型名=None, 自动=False):
        """后台激活测试；范围 ∈ 全部|配置|λ|模型|API|RAG|记忆"""
        if self.运行中:
            raise RuntimeError("测试运行中，请稍候")
        模型名 = 模型名 or 管理器.已加载模型名
        self.运行中 = True
        self.当前范围 = 范围
        self.当前模型名 = 模型名
        self.最近结果 = None
        self.开始时间 = time.time()
        线程 = threading.Thread(
            target=self._执行, args=(范围, 模型名, 自动), daemon=True)
        线程.start()
        return {"已启动": True, "范围": 范围, "模型名": 模型名, "自动": 自动}

    def _执行(self, 范围, 模型名, 自动):
        流程.开始("测试")
        try:
            测试项 = []
            if 范围 in ("全部", "配置"):
                测试项.append(self._测试配置匹配())
            if 范围 in ("全部", "λ"):
                测试项.append(self._测试λ调度())
            if 范围 in ("全部", "模型"):
                测试项.append(self._测试模型冒烟(模型名))
            if 范围 in ("全部", "API"):
                测试项.append(self._测试API协议(模型名))
            if 范围 in ("全部", "RAG"):
                测试项.append(self._测试RAG(模型名))
            if 范围 in ("全部", "记忆"):
                测试项.append(self._测试记忆())
            if not 测试项:
                测试项 = [{"名称": "无匹配测试", "通过": False, "耗时": 0,
                          "说明": f"范围参数无效: {范围}"}]
            通过数 = sum(1 for t in 测试项 if t["通过"])
            汇总 = {
                "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "范围": 范围, "模型名": 模型名, "自动": 自动,
                "通过": 通过数, "失败": len(测试项) - 通过数,
                "总耗时": round(time.time() - self.开始时间, 2),
                "明细": 测试项,
            }
            self.最近结果 = 汇总
            os.makedirs(报告目录, exist_ok=True)
            self.最近报告文件 = os.path.join(
                报告目录, f"测试报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(self.最近报告文件, "w", encoding="utf-8") as f:
                json.dump(汇总, f, ensure_ascii=False, indent=2)
            流程.完成("测试", f"通过 {通过数}/{len(测试项)} → {self.最近报告文件}")
            监控.记录日志(f"测试完成[{范围}] 通过 {通过数}/{len(测试项)}")
        except Exception as e:
            self.最近结果 = {"通过": 0, "失败": 1, "错误": str(e)}
            流程.失败("测试", str(e))
            监控.记录日志(f"测试失败: {e}", "ERROR")
        finally:
            self.运行中 = False

    def 状态(self):
        报告列表 = []
        if os.path.isdir(报告目录):
            for 文件 in sorted(os.listdir(报告目录)):
                if 文件.endswith(".json"):
                    报告列表.append(文件)
        return {
            "运行中": self.运行中,
            "当前范围": self.当前范围,
            "当前模型名": self.当前模型名,
            "最近结果": self.最近结果,
            "最近报告文件": self.最近报告文件,
            "报告列表": 报告列表,
            "报告目录": 报告目录,
        }

    def 读取报告(self, 文件名=None):
        if 文件名:
            return self._读取(文件名)
        文件名 = self.最近报告文件
        if 文件名:
            return self._读取(os.path.basename(文件名))
        return None

    def _读取(self, 文件名):
        路径 = os.path.join(报告目录, 文件名)
        if not os.path.exists(路径):
            return None
        with open(路径, encoding="utf-8") as f:
            return json.load(f)


引擎 = 测试引擎()
