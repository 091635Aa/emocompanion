# -*- coding: utf-8 -*-
"""
推理架构 HTTP 接口注册
======================
- 全局引擎单例字典（架构类型 + 模型路径 → 引擎实例）
- POST /api/推理/初始化 / 生成 / 释放、GET /api/推理/参数推荐
- POST /api/记忆/添加、GET /api/记忆/检索
- fastapi 缺失时 注册路由 静默跳过（打印提示后返回），不影响服务启动
"""

import os
import json
import threading

from .V1架构 import 推荐参数, V1推理引擎
from .通用架构 import 通用推理引擎, 通用注入参数
from .记忆外挂 import 记忆外挂
from . import 创建推理引擎

# 全局引擎单例：键 = (架构类型, 模型路径) → 引擎实例
引擎单例: dict = {}
_锁 = threading.Lock()

# 全局记忆外挂实例
_记忆外挂实例: 记忆外挂 = None


def _获取记忆外挂() -> 记忆外挂:
    """惰性创建全局记忆外挂实例。"""
    global _记忆外挂实例
    if _记忆外挂实例 is None:
        _记忆外挂实例 = 记忆外挂()
    return _记忆外挂实例


def 获取引擎(架构类型: str, 模型路径: str):
    """按 (架构类型, 模型路径) 获取单例引擎；不存在返回 None。"""
    with _锁:
        return 引擎单例.get((架构类型, 模型路径))


def _读取隐藏维度(模型路径: str):
    """从模型目录 config.json 读取 hidden_size（不加载模型，仅解析配置文件）。

    返回:
        int 或 None（config.json 缺失/无 hidden_size 字段）。
    """
    try:
        config路径 = os.path.join(模型路径, "config.json")
        with open(config路径, "r", encoding="utf-8") as f:
            config = json.load(f)
        for 键 in ("hidden_size", "n_embd", "d_model", "hidden_dim"):
            if 键 in config:
                return int(config[键])
        if isinstance(config.get("text_config"), dict):
            for 键 in ("hidden_size", "n_embd", "d_model"):
                if 键 in config["text_config"]:
                    return int(config["text_config"][键])
    except Exception:
        return None
    return None


def 注册路由(app) -> None:
    """注册推理架构模块的 HTTP 路由（挂载到 FastAPI 应用）。

    参数:
        app: FastAPI 应用实例。
    """
    try:
        from fastapi import FastAPI  # noqa: F401
    except ImportError:
        print("[推理架构] 未安装 fastapi，跳过路由注册")
        return
    from pydantic import BaseModel

    # ---------- 请求模型 ----------
    class 初始化请求(BaseModel):
        架构类型: str = "V通用架构"
        模型路径: str
        参数: dict = {}

    class 生成请求(BaseModel):
        提示词: str
        角色名: str = ""
        记忆开关: bool = True
        架构类型: str = "V通用架构"
        模型路径: str = ""

    class 释放请求(BaseModel):
        架构类型: str = ""
        模型路径: str = ""

    class 添加记忆请求(BaseModel):
        角色名: str
        内容: str
        标签: str = ""

    # ---------- 推理接口 ----------
    @app.post("/api/推理/初始化")
    def 推理初始化(请求: 初始化请求) -> dict:
        """创建引擎并加载模型（引擎单例：同架构+同模型复用）。"""
        键 = (请求.架构类型, 请求.模型路径)
        with _锁:
            引擎 = 引擎单例.get(键)
            if 引擎 is None:
                try:
                    引擎 = 创建推理引擎(请求.架构类型, 请求.模型路径, 请求.参数)
                except ValueError as e:
                    return {"成功": False, "错误": str(e)}
                引擎单例[键] = 引擎
        # 创建推理引擎 内部已执行 初始化；此处返回最新初始化结果（幂等）
        return 引擎.初始化(请求.模型路径, 请求.参数)

    @app.post("/api/推理/生成")
    def 推理生成(请求: 生成请求) -> dict:
        """引擎生成，可选注入记忆。"""
        引擎 = 获取引擎(请求.架构类型, 请求.模型路径) if 请求.模型路径 else None
        if 引擎 is None:
            return {"成功": False, "错误": "引擎未初始化，请先调用 /api/推理/初始化"}
        记忆外挂实例 = _获取记忆外挂() if 请求.记忆开关 else None
        return 引擎.生成(
            请求.提示词,
            角色名=请求.角色名 or None,
            记忆开关=请求.记忆开关,
            记忆外挂实例=记忆外挂实例,
        )

    @app.post("/api/推理/释放")
    def 推理释放(请求: 释放请求) -> dict:
        """释放指定引擎或全部引擎的显存。"""
        with _锁:
            if 请求.架构类型 and 请求.模型路径:
                引擎 = 引擎单例.pop((请求.架构类型, 请求.模型路径), None)
                if 引擎 is None:
                    return {"成功": False, "错误": "引擎不存在，无需释放"}
                try:
                    引擎.释放()
                except Exception as e:
                    return {"成功": False, "错误": str(e)}
                return {"成功": True, "提示": "已释放引擎并清空显存"}
            数量 = len(引擎单例)
            for 引擎 in 引擎单例.values():
                try:
                    引擎.释放()
                except Exception:
                    pass
            引擎单例.clear()
            return {"成功": True, "提示": f"已释放全部 {数量} 个引擎"}

    @app.get("/api/推理/参数推荐")
    def 推理参数推荐(模型路径: str = "") -> dict:
        """按模型 config.json 的 hidden_dim 返回推荐 λ/γ/τ。"""
        hidden_dim = _读取隐藏维度(模型路径)
        if hidden_dim is None:
            return {"成功": False, "错误": f"无法读取模型 config.json 的 hidden_size：{模型路径}"}
        参数 = 推荐参数(hidden_dim)
        return {"成功": True, "模型路径": 模型路径, "hidden_dim": hidden_dim, **参数}

    # ---------- 记忆接口 ----------
    @app.post("/api/记忆/添加")
    def 添加记忆(请求: 添加记忆请求) -> dict:
        """写入一条记忆。"""
        外挂 = _获取记忆外挂()
        文件路径 = 外挂.添加记忆(请求.角色名, 请求.内容, 请求.标签)
        return {"成功": True, "文件": 文件路径, "角色名": 请求.角色名}

    @app.get("/api/记忆/检索")
    def 检索记忆(查询: str, 角色名: str = "", 前N: int = 5) -> dict:
        """按查询检索相关记忆。"""
        外挂 = _获取记忆外挂()
        结果 = 外挂.检索相关(查询, 前N=前N, 角色名=角色名 or None)
        return {"成功": True, "条数": len(结果), "结果": 结果}
