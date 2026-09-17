# -*- coding: utf-8 -*-
"""
硬件检测与设备适配模块
======================
检测本机硬件能力（NVIDIA CUDA / GPU 型号 / 驱动 / 显存 / 系统内存），
按参数量与量化档位预估模型推理/微调所需显存，并提供 FastAPI 路由。

约束：
- 仅支持 NVIDIA CUDA，明确不支持 AMD（无 CUDA 且 GPU 品牌含 AMD/ATI/Radeon 时
  给出"不支持AMD"提示）。
- torch / psutil / pynvml 等依赖缺失或异常时全部降级返回，不崩溃。
- 检测结果在模块级缓存 `当前硬件状态` 中缓存 10 秒，避免每次 API 调用都慢。
"""

import subprocess
import time

from 核心引擎.配置管理 import 获取配置项

# 模块级缓存：检测结果 + 缓存时间戳（秒）
当前硬件状态 = {"数据": None, "时间": 0.0}
缓存时长秒 = 10.0


def _获取驱动版本() -> str:
    """获取 NVIDIA 驱动版本；优先 pynvml，其次 nvidia-smi，失败返回空串。"""
    try:
        import pynvml

        pynvml.nvmlInit()
        return str(pynvml.nvmlSystemGetDriverVersion())
    except Exception:
        pass
    try:
        输出 = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if 输出.returncode == 0 and 输出.stdout.strip():
            return 输出.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return ""


def _获取GPU品牌() -> str:
    """尽力探测 GPU 品牌，返回 "nvidia" / "amd" / "intel" / "unknown" / ""。

    优先读 torch 设备名（有 CUDA 时一定准确）；无 CUDA 时用系统查询
    （wmic / PowerShell）兜底，因此不依赖 torch 可用。
    """
    名称 = ""
    try:
        import torch

        if torch.cuda.is_available():
            try:
                名称 = (torch.cuda.get_device_name(0) or "").lower()
            except Exception:
                名称 = ""
    except Exception:
        名称 = ""
    if not 名称:
        try:
            输出 = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, timeout=5,
            )
            名称 = (输出.stdout or "").lower()
        except Exception:
            pass
    if not 名称:
        try:
            输出 = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=5,
            )
            名称 = (输出.stdout or "").lower()
        except Exception:
            pass
    if any(关键词 in 名称 for 关键词 in ("amd", "ati", "radeon")):
        return "amd"
    if "nvidia" in 名称:
        return "nvidia"
    if "intel" in 名称:
        return "intel"
    return "unknown" if 名称 else ""


def _生成支持提示(支持状态: str) -> str:
    """按支持状态生成面向用户的明确提示文案。"""
    提示表 = {
        "正常": "已检测到 NVIDIA CUDA 环境，可正常使用。",
        "不支持AMD": "检测到 AMD 显卡，本项目仅支持 NVIDIA CUDA，暂不支持 AMD。",
        "无GPU": "未检测到独立显卡，请安装 NVIDIA 显卡与驱动后重试。",
        "无CUDA": "检测到显卡但 CUDA 不可用，请安装/更新 NVIDIA 驱动与对应 CUDA 工具包。",
    }
    return 提示表.get(支持状态, "")


def 检测硬件() -> dict:
    """检测本机硬件能力（CUDA/GPU/显存/内存），结果缓存 10 秒。

    返回:
        dict：
        - cuda可用 / cuda版本 / gpu型号 / 驱动版本
        - 显存总量MB / 显存可用MB / 内存总量GB / 内存可用GB
        - 是否AMD / 支持状态（"正常" / "不支持AMD" / "无GPU" / "无CUDA"）
        - 另含接口约定兼容字段：CUDA可用 / CUDA版本 / GPU型号 / 内存总量MB /
          内存可用MB / 支持运行 / 提示
    """
    现在 = time.time()
    if 当前硬件状态["数据"] is not None and 现在 - 当前硬件状态["时间"] < 缓存时长秒:
        return dict(当前硬件状态["数据"])

    结果 = {
        "cuda可用": False,
        "cuda版本": "",
        "gpu型号": "",
        "驱动版本": "",
        "显存总量MB": 0,
        "显存可用MB": 0,
        "内存总量GB": 0.0,
        "内存可用GB": 0.0,
        "是否AMD": False,
        "支持状态": "无GPU",
    }

    # ---- torch 检测 CUDA（torch 缺失时进入降级路径，不崩溃）----
    try:
        import torch

        if torch.cuda.is_available():
            结果["cuda可用"] = True
            结果["cuda版本"] = str(torch.version.cuda or "")
            try:
                结果["gpu型号"] = str(torch.cuda.get_device_name(0) or "")
            except Exception:
                结果["gpu型号"] = ""
            try:
                总字节 = torch.cuda.get_device_properties(0).total_memory
                结果["显存总量MB"] = int(总字节 // (1024 * 1024))
            except Exception:
                结果["显存总量MB"] = 0
            try:
                _, 可用字节 = torch.cuda.mem_get_info()
                结果["显存可用MB"] = int(可用字节 // (1024 * 1024))
            except Exception:
                结果["显存可用MB"] = 结果["显存总量MB"]
            结果["驱动版本"] = _获取驱动版本()
    except Exception:
        pass  # torch 缺失/异常：CUDA 视为不可用，继续降级检测

    # ---- GPU 品牌检测（含 AMD 判定）----
    品牌 = _获取GPU品牌()
    if 结果["cuda可用"]:
        结果["是否AMD"] = False
    elif 品牌 == "amd":
        结果["是否AMD"] = True

    # ---- 支持状态 ----
    if 结果["cuda可用"]:
        结果["支持状态"] = "正常"
    elif 结果["是否AMD"]:
        结果["支持状态"] = "不支持AMD"
    elif 品牌:  # 探测到 GPU（NVIDIA/Intel/未知）但 CUDA 不可用
        结果["支持状态"] = "无CUDA"
    else:
        结果["支持状态"] = "无GPU"

    # ---- psutil 检测系统内存（psutil 缺失时降级为 0）----
    try:
        import psutil

        内存 = psutil.virtual_memory()
        结果["内存总量GB"] = round(内存.total / (1024 ** 3), 1)
        结果["内存可用GB"] = round(内存.available / (1024 ** 3), 1)
    except Exception:
        结果["内存总量GB"] = 0.0
        结果["内存可用GB"] = 0.0

    # ---- 接口约定兼容字段（接口约定.py 中的返回结构）----
    结果["CUDA可用"] = 结果["cuda可用"]
    结果["CUDA版本"] = 结果["cuda版本"]
    结果["GPU型号"] = 结果["gpu型号"]
    结果["内存总量MB"] = int(结果["内存总量GB"] * 1024)
    结果["内存可用MB"] = int(结果["内存可用GB"] * 1024)
    结果["支持运行"] = 结果["支持状态"] == "正常"
    结果["提示"] = _生成支持提示(结果["支持状态"])

    当前硬件状态["数据"] = dict(结果)
    当前硬件状态["时间"] = time.time()
    return dict(结果)


def _生成预估建议(可推理: bool, 可微调: bool, 量化: str) -> str:
    """按可运行性生成预估建议文案。"""
    if 可推理 and 可微调:
        return "显存充足，可推理并支持 LoRA 微调"
    if 可推理:
        return "可推理；显存不足以微调，建议使用 4bit 量化（QLoRA）或更换更小模型"
    return "显存不足，建议使用 4bit 量化或更换更小的模型"


def 预估显存(参数量亿: float, 量化: str) -> dict:
    """按模型参数量与量化档位预估推理/微调所需显存。

    参数:
        参数量亿: 模型参数量（单位：亿），如 3 表示 3B 模型。
        量化: 量化档位，取值 "fp16" / "4bit"。

    返回:
        dict：参数量亿 / 量化 / 推理需要MB / 微调需要MB / 可推理 / 可微调 /
        预留MB / 备注；另含接口约定兼容字段：推理显存MB / 微调显存MB /
        可用显存MB / 建议。

    显存公式（单位 MB）：
        - 权重 = 参数量亿 × 1e8 × 每参数字节数 ÷ 1024²
          （fp16 每参数 2 字节；4bit 每参数 0.5 字节）
        - 推理 = 权重 × 1.2（KV cache + 激活 + 框架开销约 +20%）
        - 微调 = 推理 × 1.2（LoRA 需额外约 +20% 基座开销）；
          4bit（QLoRA）为 ×1.1（约 +10%）
    可运行性判定：所需显存 <= 显存可用MB - 预留MB（预留取配置 硬件.显存预留MB）。
    """
    参数量亿 = float(参数量亿)
    量化 = str(量化).lower()
    if 量化 not in ("fp16", "4bit"):
        量化 = "fp16"

    硬件 = 检测硬件()
    预留MB = int(获取配置项("硬件.显存预留MB", 1024))
    可用显存MB = 硬件["显存可用MB"] or 0

    # 检测失败（无 CUDA / torch 缺失）：用乐观估计 16GB 并在备注说明
    备注 = ""
    if not 硬件["cuda可用"] or 硬件["显存可用MB"] <= 0:
        可用显存MB = 16 * 1024
        备注 = "未检测到可用 CUDA 显存，按乐观估计 16GB 计算，实际以运行环境为准"

    每参数字节 = 2.0 if 量化 == "fp16" else 0.5
    权重MB = 参数量亿 * 1e8 * 每参数字节 / (1024 * 1024)
    推理需要MB = int(权重MB * 1.2)  # 权重 + 约 20% KV cache/激活/框架开销
    微调需要MB = int(推理需要MB * (1.2 if 量化 == "fp16" else 1.1))  # LoRA +20% / QLoRA +10%

    有效显存MB = 可用显存MB - 预留MB
    可推理 = 推理需要MB <= 有效显存MB
    可微调 = 微调需要MB <= 有效显存MB

    return {
        "参数量亿": 参数量亿,
        "量化": 量化,
        "推理需要MB": 推理需要MB,
        "微调需要MB": 微调需要MB,
        "可推理": 可推理,
        "可微调": 可微调,
        "预留MB": 预留MB,
        "备注": 备注,
        # 接口约定兼容字段
        "推理显存MB": 推理需要MB,
        "微调显存MB": 微调需要MB,
        "可用显存MB": 可用显存MB,
        "建议": _生成预估建议(可推理, 可微调, 量化),
    }


def _获取GPU利用率() -> int:
    """获取 GPU 利用率（%）；优先 pynvml，其次 nvidia-smi，均不可用返回 -1。"""
    try:
        import pynvml

        pynvml.nvmlInit()
        句柄 = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetUtilizationRates(句柄).gpu)
    except Exception:
        pass
    try:
        输出 = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if 输出.returncode == 0 and 输出.stdout.strip():
            return int(输出.stdout.strip().splitlines()[0].strip())
    except Exception:
        pass
    return -1


def 显存状态() -> dict:
    """返回实时显存/内存状态（供前端轮询），不做缓存。

    返回:
        dict：显存可用MB / 显存总量MB / 内存可用GB / 内存总量GB / gpu利用率
        （GPU 利用率取不到时返回 -1）。
    """
    结果 = {
        "显存可用MB": 0,
        "显存总量MB": 0,
        "内存可用GB": 0.0,
        "内存总量GB": 0.0,
        "gpu利用率": -1,
    }
    try:
        import torch

        if torch.cuda.is_available():
            try:
                _, 可用字节 = torch.cuda.mem_get_info()
                结果["显存可用MB"] = int(可用字节 // (1024 * 1024))
            except Exception:
                pass
            try:
                总字节 = torch.cuda.get_device_properties(0).total_memory
                结果["显存总量MB"] = int(总字节 // (1024 * 1024))
            except Exception:
                pass
    except Exception:
        pass
    try:
        import psutil

        内存 = psutil.virtual_memory()
        结果["内存可用GB"] = round(内存.available / (1024 ** 3), 1)
        结果["内存总量GB"] = round(内存.total / (1024 ** 3), 1)
    except Exception:
        pass
    结果["gpu利用率"] = _获取GPU利用率()
    return 结果


def 注册路由(app) -> None:
    """注册硬件检测模块的 HTTP 路由（挂载到 FastAPI 应用）。

    接口：
        GET /api/硬件                      → 检测硬件()
        GET /api/硬件/显存预估?参数量亿=3&量化=fp16 → 预估显存(...)
        GET /api/硬件/状态                  → 显存状态()

    fastapi 不可用时静默跳过，不影响服务启动。
    """
    try:
        from fastapi import Query
    except ImportError:
        return

    @app.get("/api/硬件")
    def 查询硬件() -> dict:
        """返回本机硬件检测结果。"""
        return 检测硬件()

    @app.get("/api/硬件/显存预估")
    def 查询显存预估(
        参数量亿: float = Query(3.0, description="模型参数量（单位：亿）"),
        量化: str = Query("fp16", description="量化档位：fp16 / 4bit"),
    ) -> dict:
        """按参数量与量化档位预估推理/微调所需显存。"""
        return 预估显存(参数量亿, 量化)

    @app.get("/api/硬件/状态")
    def 查询显存状态() -> dict:
        """返回实时显存/内存状态（供前端轮询）。"""
        return 显存状态()
