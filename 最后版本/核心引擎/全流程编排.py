# -*- coding: utf-8 -*-
"""
全流程编排模块（Task 12）
========================
把各独立模块串成"一体化全流程"：数据输入（上传/清洗）→ 话题分割
（文本输入直接分割，音视频先 ASR 转写）→ 打标 → 导出数据包 →
日记生成（可选）→ 微调 → 达标评估 → 推理演示。

- 每步可勾选跳过（步骤配置中该键为 false / None 即跳过）；
- 每步用 try/except 包裹，失败记录 步骤结果[步骤名] = {"成功": false, "错误": ...}
  并继续后续步骤；除非失败步骤是后续步骤的前提，则跳过后续并注明"依赖前置失败"；
- 进度回调(百分比, "当前步骤：XX")，总进度按启用步骤数均分；
- 返回 {"成功": true/false, "步骤结果": {...}, "汇总": {...}}。

接口：
- 全流程(步骤配置: dict, 进度回调=None) -> dict
- 注册路由(app)：POST /api/全流程/执行（BackgroundTasks 后台执行）+
  GET /api/全流程/进度（全局 全流程进度 字典，含 当前步骤/百分比/消息/每步结果）

步骤配置示例：
    {"数据输入": {"类型": "文本", "路径": "D:/a.txt"},
     "预处理": true, "打标": true, "日记": true,
     "微调": {"模型路径": "数据/模型库/qwen2.5-1.5b", "启用情感微调": true},
     "推理": {"架构": "V1简单回响"}, "达标": true}

约束：
- 所有外部依赖（transformers / fastapi / uvicorn 等）按需 try/except 容错，
  缺失时记录友好错误（附安装命令），绝不崩溃。
"""

import json
import os
import sys
import threading
import time

try:
    from 核心引擎.配置管理 import 获取配置项, 解析路径, 项目根
    from 核心引擎 import 数据预处理, 打标引擎, 日记生成, 微调引擎, 达标评估
except Exception:
    当前目录 = os.path.dirname(os.path.abspath(__file__))
    项目根 = os.path.dirname(当前目录)
    if 项目根 not in sys.path:
        sys.path.insert(0, 项目根)
    from 核心引擎.配置管理 import 获取配置项, 解析路径, 项目根
    from 核心引擎 import 数据预处理, 打标引擎, 日记生成, 微调引擎, 达标评估

# ==================================================================
# 常量与全局状态
# ==================================================================

# 全局全流程进度缓存：任务ID -> {"当前步骤", "百分比", "消息", "完成", "每步结果", "汇总"}
全流程进度 = {}
_进度锁 = threading.Lock()

# 步骤定义（顺序即执行顺序）
步骤定义 = [
    ("数据输入", "上传并清洗"),
    ("预处理", "话题分割"),
    ("打标", "自动打标与导出数据包"),
    ("日记", "日记生成"),
    ("微调", "LoRA/QLoRA 微调"),
    ("达标", "达标评估"),
    ("推理", "推理演示"),
]

# 依赖规则：步骤 -> 其前置步骤（前置失败则该步骤跳过并注明"依赖前置失败"）
依赖规则 = {
    "预处理": ("数据输入",),
    "打标": ("预处理",),
    "日记": ("打标",),
    "微调": ("打标",),
    "达标": ("微调",),
    "推理": ("微调",),
}

# 微调/达标训练与评估依赖缺失时的统一安装命令（清华镜像，与对应模块保持一致）
训练安装命令 = (
    "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
    "transformers peft datasets accelerate bitsandbytes"
)
评估安装命令 = (
    "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers scikit-learn"
)

默认人设 = {
    "姓名": "小七",
    "性别": "女",
    "出生年份": 2008,
    "出生地": "南方小城的老巷子",
    "家庭": "爸爸妈妈和奶奶",
    "性格": "活泼开朗",
    "人设描述": "一个爱笑、爱记录生活的女孩",
    "关键经历": "",
}


# ==================================================================
# 内部辅助函数
# ==================================================================


def _生成任务ID() -> str:
    """生成唯一任务ID：时间戳（14位）+ 随机hex（6位），共 20 位。"""
    return time.strftime("%Y%m%d%H%M%S") + __import__("secrets").token_hex(3)


def _读文本文件(路径: str) -> str:
    """按多种编码尝试读取文本文件，避免乱码。"""
    for 编码 in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            with open(路径, "r", encoding=编码) as 文件:
                return 文件.read()
        except (UnicodeDecodeError, OSError):
            continue
    with open(路径, "r", encoding="utf-8", errors="replace") as 文件:
        return 文件.read()


def _更新全局进度(任务ID: str, 百分比: float, 消息: str) -> None:
    """更新模块级全流程进度缓存（线程安全）。"""
    with _进度锁:
        条目 = 全流程进度.get(任务ID)
        if 条目 is None:
            return
        条目["百分比"] = round(百分比, 2)
        条目["消息"] = 消息
        步骤名 = 消息.split("（", 1)[0].replace("当前步骤：", "").strip()
        if 步骤名 and 步骤名 != "当前步骤":
            条目["当前步骤"] = 步骤名


# ==================================================================
# 各步骤执行函数（统一签名：(配置值, 上下文, 进度回调) -> dict）
# ==================================================================


def _执行数据输入(值, 上下文, 进度回调) -> dict:
    """上传并清洗输入文件（视频/音频/文本）。"""
    值 = 值 if isinstance(值, dict) else {}
    类型 = str(值.get("类型") or "文本")
    路径 = str(值.get("路径") or "")
    if not 路径:
        return {"成功": False, "错误": "未提供输入文件路径（步骤配置.数据输入.路径）"}
    if 进度回调:
        try:
            进度回调(0.5, "正在上传并清洗输入文件")
        except Exception:
            pass
    结果 = 数据预处理.上传并清洗(路径, 类型)
    上下文["输入类型"] = 类型
    if 结果.get("成功"):
        上下文["上传路径"] = 结果.get("文件路径") or 结果.get("路径") or ""
        上下文["任务ID"] = 结果.get("任务ID") or ""
    return 结果


def _执行预处理(值, 上下文, 进度回调) -> dict:
    """话题分割：文本输入直接分割；音视频先 ASR 转写再分割。"""
    类型 = 上下文.get("输入类型") or "文本"
    上传路径 = 上下文.get("上传路径") or ""
    文本 = ""
    if 类型 == "文本":
        if not 上传路径 or not os.path.isfile(上传路径):
            return {"成功": False, "错误": f"文本输入文件不存在：{上传路径}"}
        文本 = _读文本文件(上传路径)
    else:
        # 音视频：先 ASR 转写（需配置全模态 ASR 模型，缺失时返回友好错误）
        if 进度回调:
            try:
                进度回调(0.1, "正在 ASR 转写音视频")
            except Exception:
                pass
        转写结果 = 数据预处理.音频转文本(上传路径, 进度回调=进度回调)
        if not 转写结果.get("成功"):
            return 转写结果
        文本 = 转写结果.get("文本") or 转写结果.get("转写文本") or ""
        上下文["转写结果"] = 转写结果
    if not str(文本).strip():
        return {"成功": False, "错误": "输入内容为空，无法话题分割"}
    if 进度回调:
        try:
            进度回调(0.6, "正在按话题语义分割文本")
        except Exception:
            pass
    片段列表 = 数据预处理.话题分割(str(文本), [])
    if not 片段列表:
        return {"成功": False, "错误": "话题分割结果为空（文本过短或无可分割内容）"}
    上下文["片段列表"] = 片段列表
    return {"成功": True, "片段数": len(片段列表), "片段列表": 片段列表}


def _执行打标(值, 上下文, 进度回调) -> dict:
    """自动打标 + 导出微调数据包，并把打标结果落盘供日记等后续步骤读取。"""
    片段列表 = 上下文.get("片段列表")
    if not 片段列表:
        return {"成功": False, "错误": "缺少分割片段（前置步骤未产出片段列表）"}
    if 进度回调:
        try:
            进度回调(0.4, "正在自动打标")
        except Exception:
            pass
    打标后 = 打标引擎.自动打标(片段列表)
    任务ID = 上下文.get("任务ID") or _生成任务ID()
    # 打标结果落盘 数据/打标结果/<任务ID>_打标结果.json（供日记上下文素材读取）
    打标结果目录 = 解析路径("数据/打标结果")
    os.makedirs(打标结果目录, exist_ok=True)
    打标结果路径 = os.path.join(打标结果目录, f"{任务ID}_打标结果.json")
    try:
        with open(打标结果路径, "w", encoding="utf-8") as 文件:
            json.dump(
                {"任务ID": 任务ID, "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "片段数": len(打标后), "片段列表": 打标后},
                文件, ensure_ascii=False, indent=2,
            )
    except OSError as 错误:
        打标结果路径 = ""
        print(f"[全流程] 打标结果写入失败：{错误}")
    if 进度回调:
        try:
            进度回调(0.8, "正在导出微调数据包")
        except Exception:
            pass
    导出 = 打标引擎.导出数据包(打标后)
    上下文["打标结果"] = 打标后
    上下文["打标结果路径"] = 打标结果路径
    上下文["打标结果目录"] = 打标结果目录
    上下文["数据包路径"] = 导出.get("路径") if 导出.get("成功") else ""
    return {
        "成功": bool(导出.get("成功")),
        "片段数": len(打标后),
        "打标结果路径": 打标结果路径,
        "导出数据包": 导出,
    }


def _执行日记(值, 上下文, 进度回调) -> dict:
    """日记生成（可选步骤）：教师模型未配置/依赖缺失时自动降级为内置模板。"""
    值 = 值 if isinstance(值, dict) else {}
    人设 = 值.get("人设") if isinstance(值.get("人设"), dict) else dict(默认人设)
    参数 = 值.get("参数") if isinstance(值.get("参数"), dict) else None
    数据目录 = 上下文.get("打标结果目录") or 值.get("数据目录") or ""
    日记列表 = 日记生成.生成日记(人设, 数据目录, 参数=参数, 进度回调=进度回调)
    来源统计 = {}
    for 篇 in 日记列表 or []:
        来源 = 篇.get("来源") or "未知"
        来源统计[来源] = 来源统计.get(来源, 0) + 1
    上下文["日记列表"] = 日记列表
    return {
        "成功": True,
        "角色名": 人设.get("姓名"),
        "篇数": len(日记列表 or []),
        "来源统计": 来源统计,
    }


def _执行微调(值, 上下文, 进度回调) -> dict:
    """LoRA/QLoRA 微调：依赖缺失时返回友好错误（附安装命令），不崩溃。"""
    值 = 值 if isinstance(值, dict) else {}
    训练配置 = dict(值)
    模型路径 = str(训练配置.get("模型路径") or 训练配置.get("基座模型路径") or "").strip()
    if not 模型路径:
        return {"成功": False, "错误": "缺少必填参数：微调.模型路径"}
    结果 = 微调引擎.微调(训练配置, 进度回调=进度回调)
    if 结果.get("成功"):
        上下文["微调输出目录"] = 结果.get("输出目录") or 结果.get("输出路径") or ""
    return 结果


def _执行达标(值, 上下文, 进度回调) -> dict:
    """达标评估：优先评估微调产物，未微调时可用显式提供的 模型路径。"""
    值 = 值 if isinstance(值, dict) else {}
    模型路径 = 上下文.get("微调输出目录") or 值.get("模型路径") or ""
    if not 模型路径:
        return {
            "成功": False, "错误": "缺少待评估模型路径（微调未成功且未显式提供 达标.模型路径）",
            "安装命令": 评估安装命令,
        }
    参数 = 值.get("参数") if isinstance(值.get("参数"), dict) else None
    return 达标评估.评估达标(模型路径, 参数=参数, 进度回调=进度回调)


def _执行推理(值, 上下文, 进度回调) -> dict:
    """推理演示：用微调产物（或显式提供的 模型路径）跑一次生成。"""
    值 = 值 if isinstance(值, dict) else {}
    架构 = str(值.get("架构") or 获取配置项("推理.默认架构", "V通用架构"))
    模型路径 = 上下文.get("微调输出目录") or 值.get("模型路径") or ""
    if not 模型路径:
        return {"成功": False, "错误": "缺少推理模型路径（微调未成功且未显式提供 推理.模型路径）"}
    参数 = 值.get("参数") if isinstance(值.get("参数"), dict) else None
    try:
        from 核心引擎.推理架构 import 创建推理引擎
    except Exception as 错误:
        return {
            "成功": False,
            "错误": f"推理架构导入失败：{错误}",
            "安装命令": "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers",
        }
    引擎 = None
    try:
        if 进度回调:
            try:
                进度回调(0.3, f"正在初始化 {架构} 推理引擎")
            except Exception:
                pass
        引擎 = 创建推理引擎(架构, 模型路径, 参数)
        生成结果 = 引擎.生成("你好，介绍一下你自己吧", 记忆开关=False)
        if isinstance(生成结果, dict):
            回复 = str(生成结果.get("回复") or "")
            指标 = 生成结果.get("指标") or {}
        else:
            回复 = str(生成结果 or "")
            指标 = {}
        return {"成功": bool(回复), "架构": 架构, "回复": 回复[:300], "指标": 指标}
    except Exception as 错误:
        return {"成功": False, "错误": f"推理生成失败：{错误}"}
    finally:
        if 引擎 is not None:
            try:
                引擎.释放()
            except Exception:
                pass


# ==================================================================
# 主流程：全流程编排
# ==================================================================


def 全流程(步骤配置: dict = None, 进度回调=None) -> dict:
    """串联执行全流程各步骤，返回 {"成功", "步骤结果", "汇总"}。

    参数:
        步骤配置: 步骤配置字典，键为步骤名（数据输入/预处理/打标/日记/微调/达标/推理），
                  值为 false / None 表示跳过该步骤，true 表示按默认执行，
                  dict 为带参数的步骤配置（详见模块文档示例）。
        进度回调: 可选回调函数 进度回调(百分比: float, 消息: str)，百分比 0~1。

    返回:
        dict：{"成功": bool, "步骤结果": {步骤名: 结果dict}, "汇总": {...}}。
    """
    步骤配置 = 步骤配置 if isinstance(步骤配置, dict) else {}
    上下文 = {}
    步骤结果 = {}
    启用步骤 = []

    # 第一步：判定各步骤是否启用（false / None 跳过）
    for 步骤名, _ in 步骤定义:
        值 = 步骤配置.get(步骤名)
        if 值 is False or 值 is None:
            步骤结果[步骤名] = {
                "成功": False, "跳过": True, "原因": "步骤未启用（配置为 false 或未提供）",
            }
        else:
            启用步骤.append((步骤名, 值))

    # 第二步：按顺序执行启用步骤（进度按启用步骤数均分）
    总数 = len(启用步骤)
    for 序号, (步骤名, 值) in enumerate(启用步骤, 1):
        起点 = (序号 - 1) / 总数 if 总数 else 0.0
        终点 = 序号 / 总数 if 总数 else 1.0

        # 依赖检查：前置步骤失败则该步骤跳过并注明"依赖前置失败"
        前置失败 = [
            前置 for 前置 in 依赖规则.get(步骤名, ())
            if 步骤结果.get(前置, {}).get("成功") is not True
        ]
        if 前置失败:
            步骤结果[步骤名] = {
                "成功": False, "跳过": True,
                "原因": f"依赖前置失败：{'、'.join(前置失败)}",
            }
            continue

        def 回调(步内进度: float, 消息: str) -> None:
            if 进度回调 is not None:
                try:
                    全局进度 = 起点 + (终点 - 起点) * max(0.0, min(1.0, 步内进度))
                    进度回调(全局进度, f"当前步骤：{步骤名}（{消息}）")
                except Exception:
                    pass

        try:
            执行函数 = {
                "数据输入": _执行数据输入,
                "预处理": _执行预处理,
                "打标": _执行打标,
                "日记": _执行日记,
                "微调": _执行微调,
                "达标": _执行达标,
                "推理": _执行推理,
            }[步骤名]
            结果 = 执行函数(值, 上下文, 回调)
            if not isinstance(结果, dict):
                结果 = {"成功": False, "错误": f"步骤返回非法结果：{type(结果).__name__}"}
        except Exception as 错误:
            结果 = {"成功": False, "错误": f"步骤异常：{错误}"}
        步骤结果[步骤名] = 结果
        回调(1.0, "步骤完成")

    # 第三步：汇总统计
    成功数 = sum(1 for 结果 in 步骤结果.values() if 结果.get("成功") is True)
    失败数 = sum(
        1 for 结果 in 步骤结果.values()
        if 结果.get("成功") is False and not 结果.get("跳过")
    )
    跳过数 = sum(1 for 结果 in 步骤结果.values() if 结果.get("跳过"))
    整体成功 = 失败数 == 0 and 成功数 > 0
    汇总 = {
        "总步骤数": len(步骤结果),
        "成功": 成功数,
        "失败": 失败数,
        "跳过": 跳过数,
        "整体成功": 整体成功,
    }
    if 进度回调 is not None:
        try:
            进度回调(1.0, "全流程执行结束")
        except Exception:
            pass
    return {"成功": 整体成功, "步骤结果": 步骤结果, "汇总": 汇总}


# ==================================================================
# HTTP 路由注册
# ==================================================================


def _后台执行(步骤配置: dict, 任务ID: str) -> None:
    """后台任务：执行全流程并同步全局 全流程进度。"""

    def 回调(百分比: float, 消息: str) -> None:
        _更新全局进度(任务ID, 百分比, 消息)

    try:
        结果 = 全流程(步骤配置, 进度回调=回调)
        with _进度锁:
            条目 = 全流程进度.get(任务ID)
            if 条目 is not None:
                条目["完成"] = True
                条目["每步结果"] = 结果.get("步骤结果", {})
                条目["汇总"] = 结果.get("汇总", {})
                条目["消息"] = (
                    "全流程执行完成" if 结果.get("成功") else "全流程执行结束（存在失败/跳过步骤）"
                )
    except Exception as 错误:
        with _进度锁:
            条目 = 全流程进度.get(任务ID)
            if 条目 is not None:
                条目["完成"] = True
                条目["消息"] = f"全流程异常：{错误}"


def 注册路由(app) -> None:
    """注册全流程编排模块的 HTTP 路由（挂载到 FastAPI 应用）。

    接口:
        POST /api/全流程/执行  body：{"步骤配置": {...}} → BackgroundTasks 后台执行
        GET  /api/全流程/进度  ?任务ID= → 全局 全流程进度（当前步骤/百分比/消息/每步结果）

    fastapi 不可用时静默跳过，不影响服务启动。
    """
    try:
        from fastapi import BackgroundTasks, Query
        from pydantic import BaseModel
    except Exception as 错误:
        print(f"[全流程编排] 缺少 FastAPI 依赖，跳过路由注册：{错误}")
        return

    class 全流程请求(BaseModel):
        步骤配置: dict = {}

    @app.post("/api/全流程/执行")
    def 执行接口(请求: 全流程请求, 后台任务: BackgroundTasks) -> dict:
        try:
            任务ID = _生成任务ID()
            with _进度锁:
                全流程进度[任务ID] = {
                    "当前步骤": "已提交",
                    "百分比": 0.0,
                    "消息": "任务已提交，等待后台执行",
                    "完成": False,
                    "每步结果": {},
                    "汇总": {},
                }
            后台任务.add_task(_后台执行, 请求.步骤配置, 任务ID)
            return {
                "成功": True, "任务ID": 任务ID,
                "消息": "全流程任务已提交后台执行，可查询进度",
            }
        except Exception as 错误:
            return {"成功": False, "错误": f"提交全流程任务失败：{错误}"}

    @app.get("/api/全流程/进度")
    def 进度接口(任务ID: str = Query("")) -> dict:
        try:
            if not 任务ID:
                return {"成功": True, "任务列表": 全流程进度}
            进度 = 全流程进度.get(任务ID)
            if not 进度:
                return {
                    "成功": True, "任务ID": 任务ID,
                    "进度": {"百分比": 0.0, "消息": "未找到该任务的执行记录", "完成": False},
                }
            return {"成功": True, "任务ID": 任务ID, "进度": 进度}
        except Exception as 错误:
            return {"成功": False, "错误": f"查询全流程进度失败：{错误}"}
