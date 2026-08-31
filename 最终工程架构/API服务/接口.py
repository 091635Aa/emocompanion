# -*- coding: utf-8 -*-
"""
接口 — 自定义协议路由（/api/v1/*，中文 JSON 字段，非 OpenAI 风格）
==================================================================
统一响应格式：
    { "状态": "ok"|"error", "数据": {...}|null, "错误": null|"描述" }
覆盖：健康检查 / 模型扫描 / 注册（模型文件生成）/ 下载 / 加载 / 卸载 /
      状态 / 生成（标准流程）/ 打标（定制流程）/ 测试（激活机制）/ 流程 / 记录
"""
import sys
import os
import json
import glob

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

from 模型管理 import 管理器, 忙碌异常
from 下载器 import 下载器
from 流程编排 import 流程
from 测试引擎 import 引擎
from 打标服务 import 打标
from 监控 import 监控
from 开关 import 开关
from 记忆 import 记忆
from 复查 import 复查

数据目录 = r"f:\最终工程架构\数据"
router = APIRouter(prefix="/api/v1")


# ═══════════════════════════════════════════
# 响应辅助
# ═══════════════════════════════════════════
def 响应ok(数据):
    return {"状态": "ok", "数据": 数据, "错误": None}


def 响应错(错误, 状态码=400):
    return JSONResponse(
        {"状态": "error", "数据": None, "错误": str(错误)}, status_code=状态码)


def 捕获(函数):
    """把 函数() 的 忙碌异常/ValueError/RuntimeError 统一转成错误响应"""
    try:
        return 响应ok(函数())
    except 忙碌异常 as e:
        return 响应错(str(e), 状态码=409)
    except (ValueError, RuntimeError) as e:
        return 响应错(str(e), 状态码=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 响应错(str(e), 状态码=500)


# ═══════════════════════════════════════════
# 请求体模型（中文字段，自定义协议）
# ═══════════════════════════════════════════
class 注册请求(BaseModel):
    模型名: str
    路径: str
    类型: str = "标准"          # 标准 | 定制
    量化: str = "fp16"          # fp16 | 4bit
    动态策略: str = "B"         # A | B | C
    rag: bool = False
    lora: str = None
    长上下文: bool = False
    自动测试: bool = True


class 下载请求(BaseModel):
    目标名: str
    链接: str = None            # 直链
    HF仓库: str = None          # HuggingFace 仓库名
    镜像: str = None            # 可选镜像，如 https://hf-mirror.com


class 加载请求(BaseModel):
    模型名: str


class 生成请求(BaseModel):
    模型名: str = None
    提示词: str
    最大token: int = 128


class 打标请求(BaseModel):
    模型名: str = None
    提示词集: list = None       # 缺省读 数据\全流程_提示词.txt
    批次名: str = "批次1"
    最大token: int = 128


class 测试请求(BaseModel):
    范围: str = "全部"          # 全部 | 配置 | λ | 模型 | API | RAG | 记忆
    模型名: str = None


class 开关请求(BaseModel):
    名称: str                   # API | RAG | LoRA | 记忆 | 策略
    值: object = True           # bool 或 "A"/"B"/"C"


class 记忆开关请求(BaseModel):
    开启: bool = True


class 参数请求(BaseModel):
    名称: str                   # λ | γ | τ
    值: object = None           # 数值或 null（恢复跟随推荐）


class 复查保存请求(BaseModel):
    条目列表: list


# ═══════════════════════════════════════════
# 健康 / 服务
# ═══════════════════════════════════════════
@router.get("/health")
def 健康检查():
    return 响应ok({
        "服务": "语义回响 API 服务",
        "版本": "v1.0",
        "时间": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模型状态": 管理器.状态(),
    })


# ═══════════════════════════════════════════
# 模型：扫描 / 注册 / 下载 / 列表
# ═══════════════════════════════════════════
@router.get("/scan")
def 扫描模型():
    return 响应ok({"可用模型": 管理器.扫描可用模型()})


@router.post("/model/register")
def 注册模型(请求: 注册请求):
    return 捕获(lambda: 管理器.注册模型(
        请求.模型名, 请求.路径, 类型=请求.类型, 量化=请求.量化,
        动态策略=请求.动态策略, rag=请求.rag, lora=请求.lora,
        长上下文=请求.长上下文, 自动测试=请求.自动测试))


@router.post("/model/download")
def 下载模型(请求: 下载请求):
    def 执行():
        if not 请求.目标名:
            raise ValueError("目标名不能为空")
        if 请求.链接:
            return {"任务ID": 下载器.下载直链(请求.链接, 请求.目标名)}
        if 请求.HF仓库:
            return {"任务ID": 下载器.下载HuggingFace(
                请求.HF仓库, 请求.目标名, 镜像=请求.镜像)}
        raise ValueError("链接 与 HF仓库 至少提供一个")
    return 捕获(执行)


@router.get("/model/download/status")
def 下载状态(任务ID: str = None):
    if 任务ID:
        return 响应ok({"任务": 下载器.状态(任务ID)})
    return 响应ok({"任务列表": 下载器.全部状态()})


@router.get("/models")
def 模型列表():
    return 响应ok({"已注册": 管理器.读取模型库()})


# ═══════════════════════════════════════════
# 模型：加载 / 卸载 / 状态
# ═══════════════════════════════════════════
@router.post("/model/load")
def 加载模型(请求: 加载请求):
    return 捕获(lambda: 管理器.加载(请求.模型名))


@router.post("/model/unload")
def 卸载模型():
    return 捕获(lambda: {"已卸载": 管理器.卸载()})


@router.get("/model/status")
def 模型状态():
    return 响应ok(管理器.状态())


# ═══════════════════════════════════════════
# 生成（标准模型流程）
# ═══════════════════════════════════════════
@router.post("/generate")
def 生成(请求: 生成请求):
    return 捕获(lambda: 管理器.生成(请求.模型名, 请求.提示词, 请求.最大token))


@router.get("/generate/history")
def 生成历史(数量: int = 50):
    return 响应ok({"历史": 流程.获取()["生成历史"][-数量:]})


# ═══════════════════════════════════════════
# 打标（定制模型流程）
# ═══════════════════════════════════════════
@router.post("/label/run")
def 运行打标(请求: 打标请求):
    return 捕获(lambda: 打标.启动(
        请求.模型名, 请求.提示词集, 请求.批次名, 请求.最大token))


@router.get("/label/status")
def 打标状态():
    return 响应ok(打标.状态())


# ═══════════════════════════════════════════
# 测试（激活机制）
# ═══════════════════════════════════════════
@router.post("/test/activate")
def 激活测试(请求: 测试请求):
    return 捕获(lambda: 引擎.激活(请求.范围, 请求.模型名))


@router.get("/test/status")
def 测试状态():
    return 响应ok(引擎.状态())


@router.get("/test/reports")
def 测试报告(文件名: str = None):
    return 捕获(lambda: {"报告": 引擎.读取报告(文件名)})


# ═══════════════════════════════════════════
# 流程 / 记录
# ═══════════════════════════════════════════
@router.get("/flow/status")
def 流程状态():
    return 响应ok(流程.获取())


@router.get("/records")
def 运行记录():
    """数据\全流程_运行记录_*.json 摘要列表"""
    记录列表 = []
    for 文件 in sorted(glob.glob(os.path.join(数据目录, "全流程_运行记录_*.json"))):
        try:
            with open(文件, encoding="utf-8") as f:
                记录 = json.load(f)
            记录列表.append({
                "文件名": os.path.basename(文件),
                "路径": 文件,
                "时间戳": 记录.get("时间戳"),
                "模型": 记录.get("模型"),
                "量化": 记录.get("量化"),
                "成功率": 记录.get("成功率"),
                "汇总均值": 记录.get("汇总均值"),
            })
        except Exception:
            continue
    return 响应ok({"记录列表": 记录列表[-20:][::-1]})


# ═══════════════════════════════════════════
# 监控 / 日志 / 开关 / 记忆
# ═══════════════════════════════════════════
@router.get("/monitor")
def 监控摘要():
    """大屏数据：token 统计 / 稳定度 / 情感 / 显存 / 系统内存 / 日志"""
    return 响应ok(监控.摘要())


@router.get("/logs")
def 日志查询(数量: int = 200, 过滤: str = None):
    return 响应ok({"日志": 监控.取日志(数量=数量, 过滤=过滤)})


@router.get("/switches")
def 开关状态():
    return 响应ok(开关.值())


@router.post("/switch")
def 切换开关(请求: 开关请求):
    return 捕获(lambda: 管理器.切换开关(请求.名称, 请求.值))


@router.get("/memory/status")
def 记忆状态():
    return 响应ok({"启用": 开关.启用记忆, **记忆.状态()})


@router.get("/memory/list")
def 记忆列表(数量: int = 50):
    return 响应ok({"记忆列表": 记忆.列表(数量=数量)})


@router.post("/memory/activate")
def 记忆开关(请求: 记忆开关请求):
    return 捕获(lambda: 管理器.切换开关("记忆", 请求.开启))


@router.post("/memory/clear")
def 记忆清除():
    return 捕获(lambda: {"已清除": 记忆.清除()})


# ═══════════════════════════════════════════
# 参数调整（λ / γ / τ 直观覆盖）
# ═══════════════════════════════════════════
@router.get("/params")
def 参数信息():
    return 响应ok(管理器.参数信息())


@router.post("/params")
def 调整参数(请求: 参数请求):
    return 捕获(lambda: 管理器.调整参数(请求.名称, 请求.值))


@router.post("/params/reset")
def 重置参数():
    return 捕获(lambda: 管理器.重置参数())


# ═══════════════════════════════════════════
# 打标二次复查
# ═══════════════════════════════════════════
@router.get("/label/tasks")
def 复查任务列表():
    return 响应ok({"任务列表": 复查.任务列表()})


@router.get("/label/task/{batch}")
def 复查加载批次(batch: str):
    return 捕获(lambda: 复查.加载批次(batch))


@router.post("/label/task/{batch}")
def 复查保存批次(batch: str, 请求: 复查保存请求):
    return 捕获(lambda: 复查.保存批次(batch, 请求.条目列表))


@router.get("/label/task/{batch}/stats")
def 复查统计(batch: str):
    return 捕获(lambda: 复查.统计(batch))
