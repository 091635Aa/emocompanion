# -*- coding: utf-8 -*-
"""
主程序 — FastAPI 应用入口
===========================
- 注册 自定义协议路由（/api/v1/*）与 OpenAI 兼容路由（/v1/*）
- 挂载静态控制面板：/ → index.html（简化版），/pro → pro.html（专业版）
- 注入 模型加载完成后的自动测试回调
- CORS 全开（本地控制面板直接访问）
"""
import sys
import os

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

import 接口
import openai兼容
from 模型管理 import 管理器
from 测试引擎 import 引擎
from 监控 import 监控

静态目录 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "静态")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="语义回响 API 服务", version="v1.0",
              description="自定义协议 + OpenAI 兼容 + 双模式控制面板")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 自定义协议路由（先注册，优先级高于静态挂载）
app.include_router(接口.router)
# OpenAI 兼容路由
app.include_router(openai兼容.router)

# 专业版页面（/pro 别名 → pro.html）
@app.get("/pro", include_in_schema=False)
def 专业版():
    return FileResponse(os.path.join(静态目录, "pro.html"))


# 运行大屏（/dashboard → dashboard.html）
@app.get("/dashboard", include_in_schema=False)
def 运行大屏():
    return FileResponse(os.path.join(静态目录, "dashboard.html"))


# 静态控制面板（/ → index.html，html=True 支持 /pro.html /css/* /js/*）
if os.path.isdir(静态目录):
    app.mount("/", StaticFiles(directory=静态目录, html=True), name="静态")


@app.on_event("startup")
def 启动时():
    os.makedirs(os.path.join(本工程目录, "数据", "模型库"), exist_ok=True)
    os.makedirs(os.path.join(本工程目录, "数据", "测试报告"), exist_ok=True)
    # 主线程预热 CUDA 上下文（避免加载线程与请求线程并发初始化导致原生崩溃）
    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            监控.记录日志(f"CUDA 预热完成: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        监控.记录日志(f"CUDA 预热失败: {e}", "ERROR")
    # 注入自动测试回调：模型加载成功后自动触发 冒烟+API 测试
    管理器.自动测试回调 = lambda 模型名: 引擎.激活(
        范围="全部", 模型名=模型名, 自动=True)
    监控.记录日志("语义回响 API 服务启动完成")
    print("[主程序] 语义回响 API 服务已启动")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("主程序:app", host="127.0.0.1", port=8000, log_level="info")
