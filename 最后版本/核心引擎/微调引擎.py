# -*- coding: utf-8 -*-
"""
微调引擎模块（Task 8）
======================
三维度（情感 / 记忆 / 身份）LoRA / QLoRA 微调引擎，是"一体化全流程AI应用"
数据链路（打标 → 日记 → 微调 → 达标 → 推理）的承上启下模块：

- 检测模型能力：读取 config.json 识别架构族（Qwen2/Qwen2.5/Qwen3/Llama/
  Mistral/Gemma/Phi 等）、参数量与思考能力（Qwen3 think 标记），
  推荐 LoRA target_modules 与量化档位；
- 收集训练数据：按勾选维度混合读取 数据/微调数据包/ 下的 jsonl 数据包
  （情感微调_*.jsonl / 日记微调_*.jsonl / 身份微调_*.jsonl），统一字段、
  文本清洗（去控制字符 / 去 U+FFFD / 压缩空白 / 统一换行）、去重、
  丢弃空文本与超长样例（>4096 token 预估）；
- 微调：transformers + peft 执行 LoRA/QLoRA 训练，产物落盘
  数据/微调输出/<角色名或"统一模型">_<时间戳>/，并写 微调报告.json；
- 注册路由：检测 / 数据预览 / 开始（BackgroundTasks 后台） / 进度 四个接口。

依赖策略：transformers / peft / datasets / bitsandbytes 未安装时返回
友好错误（附清华镜像安装命令），绝不崩溃；torch 缺失同理。

约定：
- 训练配置以 "模型路径" 为准（兼容接口约定的 "基座模型路径"）；
- 数据文件统一 instruction/response 或 messages 字段，逐行 JSON；
- 进度回调：进度回调(进度: float, 消息: str)，进度 0.0 ~ 1.0，可传 None。
"""

import glob
import json
import os
import re
import threading
import time

try:
    from 核心引擎 import 硬件检测
except Exception:
    # 硬件检测模块缺失/导入失败：降级为 None，"可微调"判定退化为乐观默认
    硬件检测 = None

from 核心引擎 import 配置管理

# 训练依赖缺失时的统一安装命令（清华镜像）
训练安装命令 = (
    "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
    "transformers peft datasets accelerate bitsandbytes"
)

# 维度 → 数据包文件前缀（位于 数据/微调数据包/ 下）
维度文件前缀 = {
    "情感": "情感微调_",
    "记忆": "日记微调_",
    "身份": "身份微调_",
}

# 架构族 → LoRA target_modules 推荐
架构族目标模块 = {
    "Qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "Qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "Qwen2.5": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "Qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "Llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "Mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "Gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "Phi": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "ChatGLM": ["query_key_value", "dense", "dense_h_to_4h"],
    "通用": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# 架构族 → chat 模板名（接口约定兼容字段）
架构族chat模板 = {
    "Qwen": "qwen", "Qwen2": "qwen", "Qwen2.5": "qwen", "Qwen3": "qwen",
    "Llama": "llama", "Mistral": "mistral", "Gemma": "gemma",
    "Phi": "phi", "ChatGLM": "chatglm", "通用": "通用",
}

思考标记 = ["<think>", "</think>"]

# 微调进度缓存：任务ID -> {阶段, 百分比, 消息, 日志尾部, 状态, 结果}
微调进度 = {}
_进度锁 = threading.Lock()


# ==================================================================
# 内部工具
# ==================================================================

def _生成任务ID() -> str:
    """生成唯一任务ID：时间戳（14位）+ 随机hex（6位），共 20 位。"""
    return time.strftime("%Y%m%d%H%M%S") + __import__("secrets").token_hex(3)


def _解析路径(路径: str) -> str:
    """把相对项目根路径解析为绝对路径（空值/绝对路径原样返回）。"""
    if not 路径:
        return 路径
    if os.path.isabs(路径):
        return os.path.abspath(路径)
    return 配置管理.解析路径(路径)


def _识别架构族(架构: str) -> str:
    """按架构字符串识别架构族（Qwen 系精确到 Qwen2/Qwen2.5/Qwen3）。"""
    架构 = (架构 or "").lower()
    if "qwen3" in 架构:
        return "Qwen3"
    if "qwen2.5" in 架构:
        return "Qwen2.5"
    if "qwen2" in 架构:
        return "Qwen2"
    if "qwen" in 架构:
        return "Qwen"
    if "llama" in 架构:
        return "Llama"
    if "mistral" in 架构:
        return "Mistral"
    if "gemma" in 架构:
        return "Gemma"
    if "phi" in 架构:
        return "Phi"
    if "chatglm" in 架构 or "glm" in 架构:
        return "ChatGLM"
    return "通用"


def _从名称提取参数量亿(文本) -> float:
    """从名称文本提取参数量（亿）：如 "Qwen2.5-7B" → 7.0、"500M" → 0.5。"""
    if not isinstance(文本, str):
        return None
    匹配 = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", 文本)
    if 匹配:
        return float(匹配.group(1))
    匹配 = re.search(r"(\d+(?:\.\d+)?)\s*[Mm]", 文本)
    if 匹配:
        return round(float(匹配.group(1)) / 1000, 3)
    return None


def _从配置估算参数量亿(配置: dict) -> float:
    """依据 hidden_size / 层数 / 词表大小 估算参数量（亿），无法估算返回 None。"""
    try:
        层数 = 配置.get("num_hidden_layers") or 配置.get("num_layers") or 32
        隐藏维度 = 配置.get("hidden_size") or 配置.get("d_model") or 配置.get("n_embd") or 2048
        词表大小 = 配置.get("vocab_size") or 32000
        # 每层约 16*H^2（注意力 4H^2 + MLP 12H^2），另加词表嵌入 V*H
        参数量 = 层数 * 16 * 隐藏维度 * 隐藏维度 + 词表大小 * 隐藏维度
        return round(参数量 / 1e9, 3)
    except Exception:
        return None


def _解析参数量亿(模型路径: str, 配置: dict) -> tuple:
    """解析模型参数量（亿），优先级：config 名称 → 目录名 → safetensors 索引 → hidden_size 推算。

    返回:
        (参数量亿, 来源)；无法解析时返回 (None, "")。
    """
    名称 = 配置.get("_name_or_path") or os.path.basename(模型路径)
    for 候选, 来源 in ((名称, "config名称"), (os.path.basename(模型路径), "目录名")):
        亿数 = _从名称提取参数量亿(候选)
        if 亿数:
            return 亿数, 来源
    索引路径 = os.path.join(模型路径, "model.safetensors.index.json")
    if os.path.exists(索引路径):
        try:
            with open(索引路径, "r", encoding="utf-8-sig") as 文件:
                索引 = json.load(文件)
            总字节 = (索引.get("metadata") or {}).get("total_size")
            if 总字节:
                # total_size 为权重字节数，按 2 字节/参数（fp16/bf16）换算为亿
                return round(总字节 / 2 / 1e9, 3), "索引文件"
        except Exception:
            pass
    亿数 = _从配置估算参数量亿(配置)
    if 亿数:
        return 亿数, "hidden_size推算"
    return None, ""


def _检测思考能力(模型路径: str, 配置: dict) -> tuple:
    """检测模型思考能力，返回 (支持思考: bool, 思考标记: list)。

    判定：model_type/architectures 含 qwen3 → 支持；config 含 enable_thinking
    真值 → 支持；否则查 config/tokenizer_config 的 chat_template 是否含 "think"。
    """
    model_type = str(配置.get("model_type") or "").lower()
    架构 = str(配置.get("architectures") or "").lower()
    if "qwen3" in model_type or "qwen3" in 架构:
        return True, list(思考标记)
    启用思考 = 配置.get("enable_thinking")
    if 启用思考 in (True, 1, "true", "True", "1"):
        return True, list(思考标记)
    # 查 chat_template（config.json 通常没有，补充读取 tokenizer_config.json）
    chat模板 = 配置.get("chat_template")
    if not isinstance(chat模板, str):
        chat模板 = ""
        try:
            with open(os.path.join(模型路径, "tokenizer_config.json"), "r", encoding="utf-8-sig") as 文件:
                tokenizer配置 = json.load(文件)
            chat模板 = tokenizer配置.get("chat_template") or ""
            if not isinstance(chat模板, str):
                chat模板 = str(chat模板)
        except Exception:
            chat模板 = ""
    if "think" in chat模板:
        return True, list(思考标记)
    return False, []


def _取维度启用(训练配置: dict, 默认取配置: bool = True) -> dict:
    """读取维度启用开关：训练配置显式字段优先。

    参数:
        训练配置: 训练配置字典。
        默认取配置: 未显式提供的维度是否回退取配置文件 微调.启用*微调
            （True 取配置默认值，False 视为不启用）。

    返回:
        {"情感": bool, "记忆": bool, "身份": bool}。
    """
    配置 = 训练配置 if isinstance(训练配置, dict) else {}
    默认值 = {
        "情感": 配置管理.获取配置项("微调.启用情感微调", True) if 默认取配置 else False,
        "记忆": 配置管理.获取配置项("微调.启用记忆微调", True) if 默认取配置 else False,
        "身份": 配置管理.获取配置项("微调.启用身份微调", True) if 默认取配置 else False,
    }
    return {
        "情感": bool(配置.get("启用情感微调", 默认值["情感"])),
        "记忆": bool(配置.get("启用记忆微调", 默认值["记忆"])),
        "身份": bool(配置.get("启用身份微调", 默认值["身份"])),
    }


def _清洗训练文本(文本) -> str:
    """文本清洗：去控制字符、去 U+FFFD 乱码、压缩空白、统一换行。"""
    if not isinstance(文本, str):
        return ""
    保留 = []
    for 字符 in 文本:
        if 字符 in ("\n", "\t"):
            保留.append(字符)
        elif ord(字符) < 32 or ord(字符) == 127 or 字符 == "\ufffd":
            continue
        else:
            保留.append(字符)
    文本 = "".join(保留)
    文本 = 文本.replace("\r\n", "\n").replace("\r", "\n")
    文本 = re.sub(r"[ \t]+", " ", 文本)     # 压缩水平空白
    文本 = re.sub(r"\n{2,}", "\n", 文本)    # 压缩连续空行
    return 文本.strip()


def _预估token数(文本: str) -> int:
    """按字符数估算 token 数：中文约 1.3~1.5 字符/token，保守取 1.5 系数。"""
    return max(1, int(len(文本 or "") / 1.5))


def _行转样例(行: dict, 维度: str) -> dict:
    """把 jsonl 一行统一为 {"instruction", "response", "维度"}；无法提取返回 None。

    兼容 instruction/response 与 messages 字段（取首个 user 为指令、
    首个 assistant 为回答）。
    """
    if not isinstance(行, dict):
        return None
    instruction = 行.get("instruction")
    response = 行.get("response")
    if instruction is None or response is None:
        instruction = response = None
        messages = 行.get("messages")
        if isinstance(messages, list):
            for 消息 in messages:
                if not isinstance(消息, dict):
                    continue
                角色 = 消息.get("role")
                内容 = 消息.get("content")
                if 角色 == "user" and instruction is None:
                    instruction = 内容
                elif 角色 == "assistant" and response is None:
                    response = 内容
    if instruction is None or response is None:
        return None
    return {"instruction": str(instruction), "response": str(response), "维度": 维度}


def _读取维度样例(文件路径: str, 维度: str, 最大token: int = 4096) -> list:
    """逐行读取一个 jsonl 数据包：清洗、丢弃空文本/超长样例，返回样例列表。"""
    样例列表 = []
    try:
        文件句柄 = open(文件路径, "r", encoding="utf-8-sig")
    except OSError:
        return 样例列表
    with 文件句柄:
        for 行文本 in 文件句柄:
            行文本 = 行文本.strip()
            if not 行文本:
                continue
            try:
                行 = json.loads(行文本)
            except (json.JSONDecodeError, ValueError):
                continue
            样例 = _行转样例(行, 维度)
            if not 样例:
                continue
            样例["instruction"] = _清洗训练文本(样例["instruction"])
            样例["response"] = _清洗训练文本(样例["response"])
            if not 样例["instruction"] or not 样例["response"]:
                continue
            if _预估token数(样例["instruction"]) + _预估token数(样例["response"]) > 最大token:
                continue
            样例列表.append(样例)
    return 样例列表


def _收集全部样例(训练配置: dict) -> dict:
    """按勾选维度混合读取数据包并清洗去重（instruction 相同保留一条）。

    返回:
        dict：{"成功", "总条数", "维度分布", "样例"(完整列表), "文件清单", "数据包目录"}。
    """
    数据包目录 = 配置管理.获取配置项("微调.数据包目录", "数据/微调数据包")
    数据包目录 = _解析路径(数据包目录) if 数据包目录 else 配置管理.解析路径("数据/微调数据包")
    维度启用 = _取维度启用(训练配置)

    全部样例 = []
    已见指令 = set()
    维度计数 = {"情感": 0, "记忆": 0, "身份": 0}
    文件清单 = []
    for 维度, 前缀 in 维度文件前缀.items():
        if not 维度启用.get(维度):
            continue
        for 文件路径 in sorted(glob.glob(os.path.join(数据包目录, 前缀 + "*.jsonl"))):
            文件清单.append(文件路径)
            for 样例 in _读取维度样例(文件路径, 维度):
                指令 = 样例["instruction"]
                if 指令 in 已见指令:
                    continue
                已见指令.add(指令)
                全部样例.append(样例)
                维度计数[维度] += 1
    return {
        "成功": True,
        "总条数": len(全部样例),
        "维度分布": 维度计数,
        "样例": 全部样例,
        "文件清单": 文件清单,
        "数据包目录": 数据包目录,
    }


# ==================================================================
# 一、检测模型能力
# ==================================================================

def 检测模型能力(模型路径: str) -> dict:
    """检测模型架构、参数量、思考能力，推荐 LoRA target_modules 与量化档位。

    参数:
        模型路径: 本地模型绝对路径（须含 config.json）。

    返回:
        dict：完整能力报告；路径无效或读取失败返回 {"成功": False, "错误": ...}。
    """
    if not isinstance(模型路径, str) or not 模型路径.strip():
        return {"成功": False, "错误": "模型路径为空"}
    模型路径 = os.path.abspath(模型路径)
    配置路径 = os.path.join(模型路径, "config.json")
    if not os.path.isdir(模型路径) or not os.path.exists(配置路径):
        return {"成功": False, "错误": f"模型路径无效或缺少 config.json：{模型路径}"}
    try:
        with open(配置路径, "r", encoding="utf-8-sig") as 文件:
            配置 = json.load(文件)
    except Exception as 错误:
        return {"成功": False, "错误": f"config.json 读取失败：{错误}"}

    架构 = 配置.get("architectures")
    if isinstance(架构, list):
        架构 = 架构[0] if 架构 else None
    架构 = 架构 or 配置.get("model_type") or "未知"
    架构族 = _识别架构族(str(架构))
    隐藏维度 = 配置.get("hidden_size") or 配置.get("d_model") or 配置.get("n_embd") or 0
    参数量亿, 参数来源 = _解析参数量亿(模型路径, 配置)
    支持思考, 思考标记列表 = _检测思考能力(模型路径, 配置)
    目标模块 = 架构族目标模块.get(架构族, 架构族目标模块["通用"])

    # 显存预估与可微调判定（4bit 基线；硬件检测模块缺失时按乐观默认）
    预估基准 = 参数量亿 if 参数量亿 is not None else 3.0
    可微调 = True
    显存预估4bit = None
    显存预估fp16 = None
    if 硬件检测 is not None and hasattr(硬件检测, "预估显存"):
        try:
            显存预估4bit = 硬件检测.预估显存(预估基准, "4bit")
            可微调 = bool(显存预估4bit.get("可微调", True))
        except Exception:
            可微调 = True
        try:
            显存预估fp16 = 硬件检测.预估显存(预估基准, "fp16")
        except Exception:
            显存预估fp16 = None
    # 推荐量化：>=7B 用 4bit；<7B 且 fp16 可微调时用 fp16（可被训练配置覆盖）
    if 参数量亿 is not None and 参数量亿 >= 7:
        推荐量化 = "4bit"
    elif 显存预估fp16 and 显存预估fp16.get("可微调"):
        推荐量化 = "fp16"
    else:
        推荐量化 = "4bit"

    建议策略 = ("QLoRA" if 推荐量化 == "4bit" else "LoRA") + f"({推荐量化})"
    return {
        "成功": True,
        "模型路径": 模型路径,
        "架构": 架构,
        "model_type": 配置.get("model_type", ""),
        "架构族": 架构族,
        "hidden_size": 隐藏维度,
        "参数量亿": 参数量亿,
        "参数来源": 参数来源,
        "支持思考": 支持思考,
        "思考标记": 思考标记列表 if 支持思考 else [],
        "target_modules": 目标模块,
        "推荐量化": 推荐量化,
        "可微调": 可微调,
        "显存预估": 显存预估4bit or {"量化": "4bit", "可微调": 可微调},
        "建议策略": 建议策略,
        # 接口约定兼容字段
        "支持思考模式": 支持思考,
        "chat模板": 架构族chat模板.get(架构族, "通用"),
        "模型类型": 配置.get("model_type", ""),
        "提示": "" if 可微调 else "显存不足，建议使用 4bit 量化（QLoRA）或更换更小模型",
    }


# ==================================================================
# 二、收集训练数据
# ==================================================================

def 收集训练数据(训练配置: dict) -> dict:
    """按勾选维度收集并清洗训练数据（数据预览用途）。

    参数:
        训练配置: 含 启用情感微调/启用记忆微调/启用身份微调 等字段的字典。

    返回:
        dict：{"总条数", "维度分布": {"情感": n, "记忆": n, "身份": n},
               "样例"(预览前10条), "文件清单", "数据包目录"}；
        维度无数据时记为 0，不报错。
    """
    结果 = _收集全部样例(训练配置)
    结果["样例"] = 结果["样例"][:10]
    return 结果


# ==================================================================
# 三、微调（LoRA / QLoRA 训练）
# ==================================================================

def _构建训练回调(上报):
    """依赖检查通过后动态创建 TrainerCallback 子类实例（训练期间上报 loss/step 进度）。"""
    from transformers import TrainerCallback

    class _训练进度回调(TrainerCallback):
        def __init__(self, 上报, 起始=0.78, 区间=0.14):
            self.上报 = 上报
            self.起始 = 起始
            self.区间 = 区间

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.上报 is None:
                return
            总步数 = getattr(state, "max_steps", 0) or 0
            进度 = (self.起始 + self.区间 * state.global_step / 总步数) if 总步数 > 0 else self.起始
            loss = ""
            if isinstance(logs, dict) and isinstance(logs.get("loss"), (int, float)):
                loss = f" loss={logs['loss']:.4f}"
            self.上报(min(进度, self.起始 + self.区间), f"训练中 step {state.global_step}/{总步数}{loss}")

    return _训练进度回调(上报)


def 微调(训练配置: dict, 进度回调=None) -> dict:
    """执行三维度（情感/记忆/身份）LoRA/QLoRA 微调，产物落盘 数据/微调输出/。

    参数:
        训练配置: 训练配置字典，字段见任务说明；"模型路径" 必填（兼容 "基座模型路径"）。
        进度回调: 可选回调 进度回调(进度, 消息)，进度 0.0 ~ 1.0。

    返回:
        dict：成功 {"成功": True, "输出目录", "轮数", "维度分布", "训练损失", ...}；
        失败 {"成功": False, "错误": ..., "安装命令": ...}。
    """
    if not isinstance(训练配置, dict):
        return {"成功": False, "错误": "训练配置必须为字典"}

    def 上报(进度: float, 消息: str) -> None:
        if 进度回调 is not None:
            try:
                进度回调(进度, 消息)
            except Exception:
                pass

    上报(0.02, "开始校验训练配置")

    # ---- 1. 模型路径校验 ----
    模型路径 = str(训练配置.get("模型路径") or 训练配置.get("基座模型路径") or "").strip()
    if not 模型路径:
        return {"成功": False, "错误": "缺少必填参数：模型路径"}
    模型路径 = os.path.abspath(模型路径)
    if not os.path.isdir(模型路径) or not os.path.exists(os.path.join(模型路径, "config.json")):
        return {"成功": False, "错误": f"模型路径无效或缺少 config.json：{模型路径}"}
    上报(0.06, "模型路径校验通过")

    # ---- 2. 维度启用校验（显式提供任一启用字段时，其余维度视为不启用）----
    显式维度 = any(键 in 训练配置 for 键 in ("启用情感微调", "启用记忆微调", "启用身份微调"))
    维度启用 = _取维度启用(训练配置, 默认取配置=not 显式维度)
    if not any(维度启用.values()):
        return {"成功": False, "错误": "至少需要启用一个微调维度（情感/记忆/身份）"}
    上报(0.1, "微调维度校验通过")

    # ---- 3. 数据文件存在性校验 ----
    数据包目录 = 配置管理.获取配置项("微调.数据包目录", "数据/微调数据包")
    数据包目录 = _解析路径(数据包目录) if 数据包目录 else 配置管理.解析路径("数据/微调数据包")
    缺失维度 = [
        维度 for 维度, 前缀 in 维度文件前缀.items()
        if 维度启用.get(维度) and not glob.glob(os.path.join(数据包目录, 前缀 + "*.jsonl"))
    ]
    if 缺失维度:
        return {
            "成功": False,
            "错误": "未找到 "
            + "、".join(f"{维度}维度数据（{维度文件前缀[维度]}*.jsonl）" for 维度 in 缺失维度)
            + f"，请先在 {数据包目录} 下准备对应数据包",
        }
    上报(0.15, "数据文件校验通过")

    # ---- 4. 收集并清洗训练数据 ----
    数据结果 = _收集全部样例(训练配置)
    if 数据结果["总条数"] <= 0:
        return {"成功": False, "错误": "未找到有效训练样例（数据为空或清洗后全部被过滤）"}
    上报(0.2, f"训练数据就绪：共 {数据结果['总条数']} 条")

    # ---- 5. 检测模型能力（决定 target_modules 与推荐量化）----
    能力 = 检测模型能力(模型路径)
    if not 能力.get("成功"):
        return {"成功": False, "错误": f"模型能力检测失败：{能力.get('错误', '')}"}
    上报(0.28, f"模型能力检测完成（架构族：{能力.get('架构族')}）")

    # ---- 6. 训练依赖检查（缺失时返回友好错误，不崩溃）----
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
            DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
    except Exception as 错误:
        return {
            "成功": False,
            "错误": f"缺少训练依赖（transformers/peft/datasets/bitsandbytes）：{错误}",
            "安装命令": 训练安装命令,
        }
    上报(0.35, "训练依赖就绪")

    # ---- 7. 训练参数决策（训练配置 > 检测推荐 > 配置文件默认值）----
    量化 = str(训练配置.get("量化") or 配置管理.获取配置项("微调.默认量化", "auto")).lower()
    if 量化 == "auto":
        量化 = 能力.get("推荐量化", "4bit")
    if 量化 not in ("fp16", "4bit"):
        量化 = "4bit"
    轮数 = int(训练配置.get("轮数") or 配置管理.获取配置项("微调.默认轮数", 3))
    学习率 = float(训练配置.get("学习率") or 配置管理.获取配置项("微调.默认学习率", 0.0002))
    批量 = int(训练配置.get("批量") or 配置管理.获取配置项("微调.默认批量", 4))
    max_length = int(训练配置.get("max_length") or 512)
    组织方式 = str(训练配置.get("模型组织方式") or 配置管理.获取配置项("微调.模型组织方式", "一角色一模型"))
    角色名 = str(训练配置.get("角色名") or 训练配置.get("角色") or "").strip()

    # ---- 8. 输出目录：数据/微调输出/<角色名或"统一模型">_<时间戳>/ ----
    输出目录 = 训练配置.get("输出目录") or ""
    if 输出目录:
        输出目录 = _解析路径(输出目录)
    else:
        微调输出根 = 配置管理.获取配置项("微调.输出目录", "数据/微调输出")
        微调输出根 = _解析路径(微调输出根)
        模型名称 = 角色名 if 组织方式 == "一角色一模型" and 角色名 else "统一模型"
        输出目录 = os.path.join(微调输出根, f"{模型名称}_{time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(输出目录, exist_ok=True)
    上报(0.4, f"输出目录：{输出目录}")

    # ---- 9. 加载模型与 tokenizer（4bit 走 BitsAndBytesConfig / 其余 fp16）----
    模型 = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        加载参数 = {"trust_remote_code": True, "device_map": "auto"}
        if 量化 == "4bit":
            from transformers import BitsAndBytesConfig
            加载参数["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.float16,
            )
        else:
            加载参数["torch_dtype"] = torch.float16
        模型 = AutoModelForCausalLM.from_pretrained(模型路径, **加载参数)
    except Exception as 错误:
        for 对象 in (tokenizer, 模型):
            try:
                del 对象
            except Exception:
                pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return {
            "成功": False,
            "错误": f"模型加载失败（{量化}）：{错误}",
            "安装命令": 训练安装命令 if "bitsandbytes" in str(错误).lower() else "",
        }
    上报(0.55, f"模型加载完成（量化：{量化}）")

    try:
        # ---- 10. LoRA 配置（r=16, lora_alpha=32）----
        目标模块 = 能力.get("target_modules") or 架构族目标模块["通用"]
        lora配置 = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=目标模块,
        )
        模型 = get_peft_model(模型, lora配置)
        try:
            模型.print_trainable_parameters()
        except Exception:
            pass
        上报(0.6, "LoRA 适配器装配完成")

        # ---- 11. 构建训练数据集（优先 apply_chat_template，失败回退 Q/A 文本）----
        def _格式化文本(样例: dict) -> str:
            try:
                文本 = tokenizer.apply_chat_template(
                    [{"role": "user", "content": 样例["instruction"]},
                     {"role": "assistant", "content": 样例["response"]}],
                    tokenize=False, add_generation_prompt=False,
                )
                if 文本 and str(文本).strip():
                    return str(文本)
            except Exception:
                pass
            return f"Q: {样例['instruction']}\nA: {样例['response']}"

        def _分词样例(样例: dict) -> dict:
            输出 = tokenizer(
                _格式化文本(样例), truncation=True, max_length=max_length, padding=False
            )
            输出["labels"] = 输出["input_ids"].copy()
            return 输出

        数据集 = Dataset.from_list([_分词样例(样例) for 样例 in 数据结果["样例"]])
        上报(0.68, f"数据集构建完成：{len(数据集)} 条，max_length={max_length}")

        # ---- 12. 训练参数与 Trainer ----
        训练参数 = TrainingArguments(
            output_dir=输出目录,
            num_train_epochs=轮数,
            learning_rate=学习率,
            per_device_train_batch_size=批量,
            gradient_accumulation_steps=2,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            fp16=bool(torch.cuda.is_available()),
            dataloader_pin_memory=False,
        )
        回调列表 = [_构建训练回调(上报)] if 进度回调 is not None else []
        训练器 = Trainer(
            model=模型,
            args=训练参数,
            train_dataset=数据集,
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
            callbacks=回调列表,
        )
        上报(0.75, "开始训练")
        开始时间 = time.time()
        训练结果 = 训练器.train()
        用时秒 = round(time.time() - 开始时间, 1)

        # ---- 13. 收集指标并保存产物 ----
        训练损失 = None
        try:
            训练损失 = float(getattr(训练结果, "training_loss", None) or 0.0)
        except (TypeError, ValueError):
            训练损失 = None
        损失曲线 = [
            round(条目["loss"], 4)
            for 条目 in 训练器.state.log_history
            if isinstance(条目, dict)
            and isinstance(条目.get("loss"), (int, float))
        ]
        上报(0.92, f"训练完成：loss={训练损失}，用时 {用时秒} 秒")

        os.makedirs(输出目录, exist_ok=True)
        模型.save_pretrained(输出目录)
        tokenizer.save_pretrained(输出目录)

        维度分布 = 数据结果["维度分布"]
        总条数 = 数据结果["总条数"]
        各维度占比 = {
            维度: round(计数 / 总条数, 4) for 维度, 计数 in 维度分布.items()
        } if 总条数 else {"情感": 0, "记忆": 0, "身份": 0}
        报告 = {
            "模型路径": 模型路径,
            "架构": 能力.get("架构"),
            "架构族": 能力.get("架构族"),
            "参数量亿": 能力.get("参数量亿"),
            "支持思考": 能力.get("支持思考"),
            "量化": 量化,
            "target_modules": 目标模块,
            "轮数": 轮数,
            "学习率": 学习率,
            "批量": 批量,
            "max_length": max_length,
            "总条数": 总条数,
            "维度分布": 维度分布,
            "各维度数据占比": 各维度占比,
            "训练损失": 训练损失,
            "损失曲线": 损失曲线,
            "角色名": 角色名,
            "模型组织方式": 组织方式,
            "用时秒": 用时秒,
            "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        报告路径 = os.path.join(输出目录, "微调报告.json")
        with open(报告路径, "w", encoding="utf-8") as 文件:
            json.dump(报告, 文件, ensure_ascii=False, indent=2)
        上报(1.0, "微调完成，产物已保存")
        return {
            "成功": True,
            "状态": "完成",
            "输出目录": 输出目录,
            "输出路径": 输出目录,
            "报告路径": 报告路径,
            "轮数": 轮数,
            "学习率": 学习率,
            "批量": 批量,
            "量化": 量化,
            "维度分布": 维度分布,
            "各维度数据占比": 各维度占比,
            "总条数": 总条数,
            "训练损失": 训练损失,
            "损失曲线": 损失曲线,
            "用时秒": 用时秒,
            "架构族": 能力.get("架构族"),
        }
    except Exception as 错误:
        return {
            "成功": False,
            "错误": f"微调过程异常：{错误}",
            "安装命令": 训练安装命令,
        }
    finally:
        # 无论成败都释放模型与显存
        for 对象 in (模型, tokenizer):
            try:
                del 对象
            except Exception:
                pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ==================================================================
# 四、HTTP 路由
# ==================================================================

def _更新进度(任务ID: str, 进度: float, 消息: str) -> None:
    """更新全局微调进度缓存（线程安全），消息追加进日志尾部（保留最后 2000 字符）。"""
    with _进度锁:
        条目 = 微调进度.setdefault(任务ID, {
            "阶段": "准备", "百分比": 0.0, "消息": "", "日志尾部": "", "状态": "运行中", "结果": None,
        })
        条目["百分比"] = round(进度, 3)
        条目["消息"] = 消息
        新行 = f"[{time.strftime('%H:%M:%S')}] {消息}"
        尾部 = 条目["日志尾部"]
        条目["日志尾部"] = ((尾部 + "\n" + 新行) if 尾部 else 新行)[-2000:]
        if 进度 >= 1.0:
            条目["阶段"] = "完成"
        elif 进度 >= 0.75:
            条目["阶段"] = "训练中"
        elif 进度 > 0:
            条目["阶段"] = "准备中"


def _后台微调任务(训练配置: dict, 任务ID: str) -> None:
    """后台线程执行微调，把阶段/百分比/消息/日志尾部写入全局进度字典。"""
    _更新进度(任务ID, 0.0, "任务已提交，准备开始")
    print(f"[微调引擎] 任务 {任务ID} 开始后台训练")
    try:
        结果 = 微调(
            训练配置, 进度回调=lambda 进度, 消息: _更新进度(任务ID, 进度, 消息)
        )
    except Exception as 错误:
        结果 = {"成功": False, "错误": f"微调异常：{错误}", "安装命令": 训练安装命令}
    with _进度锁:
        条目 = 微调进度.get(任务ID)
        if 条目 is None:
            条目 = 微调进度.setdefault(任务ID, {
                "阶段": "完成", "百分比": 0.0, "消息": "", "日志尾部": "", "状态": "运行中", "结果": None,
            })
        条目["状态"] = "完成" if 结果.get("成功") else "失败"
        条目["阶段"] = "完成"
        条目["百分比"] = 1.0 if 结果.get("成功") else 0.0
        收尾行 = (
            f"[{time.strftime('%H:%M:%S')}] 微调完成，产物已保存"
            if 结果.get("成功") else f"[{time.strftime('%H:%M:%S')}] {结果.get('错误', '微调失败')}"
        )
        尾部 = 条目["日志尾部"]
        条目["日志尾部"] = ((尾部 + "\n" + 收尾行) if 尾部 else 收尾行)[-2000:]
        条目["消息"] = "微调完成" if 结果.get("成功") else 结果.get("错误", "微调失败")
        条目["结果"] = 结果
    print(f"[微调引擎] 任务 {任务ID} 结束：{'成功' if 结果.get('成功') else '失败'}")


def 注册路由(app) -> None:
    """注册微调引擎模块的 HTTP 路由（挂载到 FastAPI 应用）。

    接口:
        POST /api/微调/检测      body：{模型路径} → 检测模型能力
        POST /api/微调/数据预览  body：训练配置 → 收集训练数据预览
        POST /api/微调/开始      body：训练配置 → BackgroundTasks 后台训练
        GET  /api/微调/进度      ?任务ID= → 全局微调进度

    fastapi 不可用时静默跳过，不影响服务启动。
    """
    try:
        from fastapi import BackgroundTasks, Body
    except Exception as 错误:
        print(f"[微调引擎] 缺少 FastAPI 依赖，跳过路由注册：{错误}")
        return

    @app.post("/api/微调/检测")
    def 检测接口(请求: dict = Body(...)):
        try:
            return 检测模型能力(str(请求.get("模型路径") or "").strip())
        except Exception as 错误:
            return {"成功": False, "错误": f"模型能力检测失败：{错误}"}

    @app.post("/api/微调/数据预览")
    def 数据预览接口(请求: dict = Body(...)):
        try:
            return 收集训练数据(请求)
        except Exception as 错误:
            return {"成功": False, "错误": f"数据预览失败：{错误}"}

    @app.post("/api/微调/开始")
    def 开始微调接口(请求: dict = Body(...), 后台任务: BackgroundTasks = None):
        任务ID = _生成任务ID()
        with _进度锁:
            微调进度[任务ID] = {
                "阶段": "已提交", "百分比": 0.0, "消息": "任务已提交，等待执行",
                "日志尾部": "", "状态": "运行中", "结果": None,
            }
        if 后台任务 is None:
            return {"成功": False, "错误": "后台任务不可用"}
        后台任务.add_task(_后台微调任务, 请求, 任务ID)
        return {"成功": True, "任务ID": 任务ID, "消息": "微调任务已启动，可查询进度"}

    @app.get("/api/微调/进度")
    def 进度接口(任务ID: str = ""):
        try:
            if 任务ID:
                return {
                    "成功": True, "任务ID": 任务ID,
                    "进度": 微调进度.get(任务ID, {
                        "阶段": "未开始", "百分比": 0.0, "消息": "", "日志尾部": "", "状态": "未开始",
                    }),
                }
            return {"成功": True, "进度列表": 微调进度}
        except Exception as 错误:
            return {"成功": False, "错误": f"查询进度失败：{错误}"}
