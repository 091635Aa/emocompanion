# -*- coding: utf-8 -*-
"""
启动服务_MI300X — AMD Instinct MI300X（ROCm）专属启动脚本
==========================================================
该脚本为 AMD MI300X 设备适配，可直接上传云端部署运行。

适配点（相对标准 CUDA 启动）：
1. 环境检测：识别 torch ROCm 后端（torch.version.hip）与 AMD GPU 型号/显存
2. 环境变量：HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES /
   PYTORCH_HIP_ALLOC_CONF=expandable_segments:True（大分配更稳）
3. 量化适配：ROCm 下 bitsandbytes 量化不可靠 → 自动跳过 4bit/8bit，
   改用 bf16 加载（Qwen2.5 原生 bf16 训练，MI300X 上精度/速度最优）
4. 显存/监控无需改动：torch.cuda.* 在 ROCm 下等价映射 HIP
5. 非 ROCm 环境（CUDA/CPU）自动回退标准流程并提示

用法：
    F:\\打标\\.venv\\Scripts\\python.exe 启动服务_MI300X.py [--端口 8000] [--host 127.0.0.1]
    python 启动服务_MI300X.py --https   # 可选自签名 HTTPS

云端部署注意：
- 需 ROCm 版 PyTorch（pip install torch --index-url https://download.pytorch.org/whl/rocm6.2）
- MI300X 为 gfx942，无需设置 HSA_OVERRIDE_GFX_VERSION（原生支持）
- 192GB HBM 无需担心显存，建议直接 bf16 全量加载
"""
import os
import sys
import argparse

# ════════════════════════════════════════════════════
# 0. ROCm 环境变量（必须在 import torch 之前设置）
# ════════════════════════════════════════════════════
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

本服务目录 = os.path.dirname(os.path.abspath(__file__))
if 本服务目录 not in sys.path:
    sys.path.insert(0, 本服务目录)
本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)
agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if os.path.isdir(agent_echo目录) and agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def 检测后端():
    """返回 (后端标识, 是否为 ROCm)"""
    hip = getattr(torch.version, "hip", None)
    if hip:
        return f"ROCm (HIP {hip})", True
    cuda = getattr(torch.version, "cuda", None)
    if cuda:
        return f"CUDA ({cuda})", False
    return "CPU / 未知", False


def 打印设备信息():
    print("=" * 64)
    print(f"[MI300X] PyTorch {torch.__version__}")
    print(f"[MI300X] 后端    : {检测后端()[0]}")
    try:
        if torch.cuda.is_available():
            print(f"[MI300X] GPU     : {torch.cuda.get_device_name(0)}")
            print(f"[MI300X] 显存    : {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.0f} GB")
        else:
            print("[MI300X] 警告    : 未检测到可用 GPU（请确认 ROCm 驱动与 torch 已正确安装）")
    except Exception as e:
        print(f"[MI300X] 警告    : GPU 信息读取失败: {e}")
    print("=" * 64)


def 应用ROCM补丁():
    """替换模型加载逻辑：跳过 bitsandbytes，强制 bf16（仅 ROCm 生效）"""
    import echo_common
    import 模型管理 as 模型管理模块

    def _加载模型_MI300X(模型路径, 量化=None):
        """bf16 直载（Qwen2.5 原生 bf16；MI300X 上精度与吞吐最优）"""
        if 量化:
            print(f"[MI300X] 忽略量化请求 [{量化}]：ROCm 下禁用 bitsandbytes，改用 bf16")
        kwargs = {
            "trust_remote_code": True,
            "device_map": "cuda:0",
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
        }
        model = AutoModelForCausalLM.from_pretrained(模型路径, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
        return model, tokenizer

    echo_common.加载模型 = _加载模型_MI300X
    # 量化映射保留键（前端注册不受影响），但 4bit 也落到 bf16
    模型管理模块.量化映射 = {"fp16": None, "4bit": None}
    print("[MI300X] 已应用 ROCm 适配：bf16 全量加载 / 禁用 bitsandbytes 量化")


def 主流程():
    parser = argparse.ArgumentParser(description="AMD MI300X (ROCm) 专属启动")
    parser.add_argument("--端口", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--https", action="store_true", help="启用 HTTPS（自动生成自签名证书）")
    args = parser.parse_args()

    后端, 是ROCM = 检测后端()
    打印设备信息()

    if 是ROCM:
        应用ROCM补丁()
    else:
        print(f"[MI300X] 当前为 {后端} 环境，非 ROCm：按标准流程启动（模型仍按注册量化加载）")

    ssl参数 = {}
    if args.https:
        import 证书工具
        证书路径, 密钥路径 = 证书工具.生成证书()
        ssl参数 = {"ssl_keyfile": 密钥路径, "ssl_certfile": 证书路径}
        协议 = "https"
    else:
        协议 = "http"

    import uvicorn
    print(f"[MI300X] 启动 {协议}://{args.host}:{args.端口}  |  简化版: /  专业版: /pro  大屏: /dashboard  文档: /docs")
    uvicorn.run(
        "主程序:app", host=args.host, port=args.端口,
        log_level="info", **ssl参数)


if __name__ == "__main__":
    主流程()
