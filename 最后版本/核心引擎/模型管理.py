# -*- coding: utf-8 -*-
"""
模型管理模块
============
负责本地模型库的扫描、索引登记、在线下载与性能评估（Task 3）。

- 模型库目录：配置项 模型.模型库目录（默认 数据/模型库，经 配置管理.解析路径 转绝对路径）
- 下载镜像源：配置项 模型.可用镜像源（魔搭社区 / HuggingFace镜像）
- 扫描结果登记到 数据/模型库/模型索引.json（不存在则创建）
- 硬件检测模块（Task 2）可能尚未实现：import 时 try/except 容错，
  缺失时"可微调"判定退化为乐观默认值，不影响本模块其余功能
- 进度回调约定：`进度回调(进度, 消息)`，进度取值 0.0 ~ 1.0
- HTTP 接口：注册路由(app) 供 核心引擎/主服务.py 动态挂载
"""

import json
import os
import re
import threading
import time

try:
    from 核心引擎 import 硬件检测
except Exception:
    # 硬件检测模块尚未实现或导入失败：降级为 None，不阻断本模块
    硬件检测 = None

from 核心引擎 import 配置管理

# 全局下载进度：模型ID -> {百分比, 消息, 完成}，供进度查询接口读取
下载进度 = {}

# 性能评估冒烟参数
评估提示词 = "你好，请简单介绍一下你自己"
评估生成Token数 = 32

# 镜像源缺失时的安装提示
魔搭安装提示 = "缺少 modelscope 库，请运行: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope"
镜像安装提示 = "缺少 huggingface_hub 库，请运行: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple huggingface_hub"


# ==================================================================
# 一、内部工具
# ==================================================================

def _获取模型库目录() -> str:
    """读取配置项 模型.模型库目录 并解析为绝对路径（配置缺失时回退默认值）。"""
    配置值 = 配置管理.获取配置项("模型.模型库目录", "数据/模型库")
    if not 配置值:
        配置值 = "数据/模型库"
    return 配置管理.解析路径(配置值)


def _从名称提取参数量亿(文本) -> float:
    """从名称文本中提取参数量（单位：十亿 B），如 "Qwen2.5-7B" -> 7.0、"500M" -> 0.5。

    项目统一约定：参数量亿 的数值等价于模型名称中的 "X B" 数值
    （Qwen2.5-7B 记为 7，与 接口约定.py 示例一致）。
    """
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
    """依据 hidden_size / 层数 / 词表大小 估算参数量（单位：十亿 B），标记为估算值。"""
    try:
        层数 = 配置.get("num_hidden_layers") or 配置.get("num_layers") or 32
        隐藏维度 = 配置.get("hidden_size") or 配置.get("d_model") or 配置.get("n_embd") or 2048
        词表大小 = 配置.get("vocab_size") or 32000
        # 每层约 16*H^2（注意力 4H^2 + MLP 12H^2），另加词表嵌入 V*H
        参数量 = 层数 * 16 * 隐藏维度 * 隐藏维度 + 词表大小 * 隐藏维度
        return round(参数量 / 1e9, 3)
    except Exception:
        return None


def _解析模型路径参数量(模型路径: str) -> tuple:
    """解析模型参数量，按优先级：config 名称标记 → safetensors 索引 → hidden_size 推算。

    返回:
        (参数量亿, 来源, 是否估算)，参数无法解析时返回 (None, "", True)。
    """
    配置 = {}
    配置路径 = os.path.join(模型路径, "config.json")
    try:
        with open(配置路径, "r", encoding="utf-8-sig") as 文件:
            配置 = json.load(文件)
    except Exception:
        配置 = {}
    名称 = 配置.get("_name_or_path") or os.path.basename(模型路径)
    亿数 = _从名称提取参数量亿(名称)
    if 亿数:
        return 亿数, "config名称", False
    亿数 = _从名称提取参数量亿(os.path.basename(模型路径))
    if 亿数:
        return 亿数, "目录名", False
    索引路径 = os.path.join(模型路径, "model.safetensors.index.json")
    if os.path.exists(索引路径):
        try:
            with open(索引路径, "r", encoding="utf-8-sig") as 文件:
                索引 = json.load(文件)
            总字节 = (索引.get("metadata") or {}).get("total_size")
            if 总字节:
                # total_size 为权重字节数，按 2 字节/参数（fp16/bf16）换算为十亿
                return round(总字节 / 2 / 1e9, 3), "索引文件", False
        except Exception:
            pass
    亿数 = _从配置估算参数量亿(配置)
    if 亿数:
        return 亿数, "hidden_size推算", True
    return None, "", True


def _检测量化档位(配置: dict, 目录: str) -> str:
    """识别模型量化档位：优先 config.quantization_config，其次目录名/文件名标记。"""
    量化配置 = 配置.get("quantization_config") or {}
    if isinstance(量化配置, dict):
        方法 = 量化配置.get("quant_method")
        if 方法:
            return str(方法)
    目录名 = os.path.basename(目录).lower()
    文件标记 = []
    try:
        文件标记 = [文件.lower() for 文件 in os.listdir(目录)]
    except OSError:
        pass
    全部文本 = [目录名] + 文件标记
    for 标记 in ("4bit", "4-bit", "int4", "qlora"):
        if any(标记 in 文本 for 文本 in 全部文本):
            return "4bit"
    for 标记 in ("gptq", "awq", "quant"):
        if any(标记 in 文本 for 文本 in 全部文本):
            return 标记
    return "fp16/未知"


def _读取模型目录(目录: str):
    """读取一个模型目录（含 config.json）并提取模型信息；无效目录返回 None。"""
    配置路径 = os.path.join(目录, "config.json")
    if not os.path.exists(配置路径):
        return None
    try:
        with open(配置路径, "r", encoding="utf-8-sig") as 文件:
            配置 = json.load(文件)
    except Exception:
        return None
    目录名 = os.path.basename(目录)
    名称路径 = 配置.get("_name_or_path") or 目录名
    模型ID = 名称路径 if (isinstance(名称路径, str) and "/" in 名称路径) else 目录名
    模型名 = os.path.basename(名称路径) if isinstance(名称路径, str) else 目录名
    参数量亿, 参数来源, 是估算 = _解析模型路径参数量(目录)
    架构 = 配置.get("architectures")
    if isinstance(架构, list):
        架构 = 架构[0] if 架构 else None
    架构 = 架构 or 配置.get("model_type") or "未知"
    隐藏维度 = 配置.get("hidden_size") or 配置.get("d_model") or 配置.get("n_embd") or 0
    return {
        "模型ID": 模型ID,
        "模型名": 模型名,
        "目录路径": 目录,
        "本地路径": 目录,
        "参数量亿": 参数量亿,
        "参数量": 参数量亿,
        "参数来源": 参数来源,
        "参数量为估算": 是估算,
        "hidden_size": 隐藏维度,
        "架构": 架构,
        "量化档位": _检测量化档位(配置, 目录),
        "状态": "就绪",
    }


def _生成模型目录名(模型ID: str) -> str:
    """把模型ID最后一段转换为安全的目录名（小写）。"""
    最后一段 = str(模型ID).split("/")[-1].strip() or "model"
    安全名 = re.sub(r"[^\w.\-]", "-", 最后一段, flags=re.UNICODE)
    安全名 = re.sub(r"-+", "-", 安全名).strip("-.") or "model"
    return 安全名.lower()


def _目录大小MB(目录: str) -> float:
    """递归统计目录总大小（MB）。"""
    try:
        总字节 = 0
        for 根, _, 文件列表 in os.walk(目录):
            for 文件 in 文件列表:
                try:
                    总字节 += os.path.getsize(os.path.join(根, 文件))
                except OSError:
                    pass
        return round(总字节 / 1024 / 1024, 1)
    except Exception:
        return 0.0


def _周期进度上报(上报, 停止事件: threading.Event, 模型ID: str, 镜像源: str) -> None:
    """下载期间每秒调用一次进度回调（百分比递增至 0.85，完成由主流程收尾到 1.0）。"""
    进度 = 0.02
    while not 停止事件.is_set():
        停止事件.wait(1)
        if 停止事件.is_set():
            break
        进度 = min(进度 + 0.06, 0.85)
        上报(进度, f"正在从 {镜像源} 下载 {模型ID} ...")


# ==================================================================
# 二、扫描与登记
# ==================================================================

def 扫描本地模型() -> list:
    """遍历模型库目录，读取每个含 config.json 的子目录，返回模型信息列表（按模型名排序）。

    返回:
        list，元素为 dict；模型库不存在或为空时返回空列表。
    """
    模型库目录 = _获取模型库目录()
    if not os.path.isdir(模型库目录):
        return []
    try:
        条目列表 = sorted(os.listdir(模型库目录))
    except OSError:
        return []
    结果 = []
    for 条目 in 条目列表:
        子目录 = os.path.join(模型库目录, 条目)
        if not os.path.isdir(子目录):
            continue
        模型信息 = _读取模型目录(子目录)
        if 模型信息:
            结果.append(模型信息)
    结果.sort(key=lambda 模型: str(模型.get("模型名", "")).lower())
    return 结果


def 登记模型(模型信息: dict) -> bool:
    """把模型信息写入 数据/模型库/模型索引.json（文件不存在则创建）。

    参数:
        模型信息: 扫描结果列表（list）或包含 "模型列表" 键的字典。

    返回:
        bool：写入成功返回 True，失败返回 False。
    """
    try:
        模型库目录 = _获取模型库目录()
        os.makedirs(模型库目录, exist_ok=True)
        索引路径 = os.path.join(模型库目录, "模型索引.json")
        if isinstance(模型信息, list):
            模型列表 = 模型信息
        elif isinstance(模型信息, dict) and isinstance(模型信息.get("模型列表"), list):
            模型列表 = 模型信息["模型列表"]
        elif isinstance(模型信息, dict):
            模型列表 = [模型信息]
        else:
            return False
        待写入 = {
            "更新时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "模型数量": len(模型列表),
            "模型列表": 模型列表,
        }
        with open(索引路径, "w", encoding="utf-8") as 文件:
            json.dump(待写入, 文件, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# ==================================================================
# 三、在线下载
# ==================================================================

def _下载魔搭(模型ID: str, 模型库目录: str, 目标目录: str) -> str:
    """从魔搭社区下载模型，返回实际落盘目录；缺库/失败抛异常由上层统一兜底。"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise RuntimeError(魔搭安装提示)
    os.makedirs(目标目录, exist_ok=True)
    try:
        return snapshot_download(模型ID, local_dir=目标目录)
    except TypeError:
        # 老版本 modelscope 不支持 local_dir，退回 cache_dir
        return snapshot_download(模型ID, cache_dir=模型库目录)


def _下载镜像(模型ID: str, 模型库目录: str, 目标目录: str) -> str:
    """从 HuggingFace 镜像（hf-mirror.com）下载模型，返回实际落盘目录。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(镜像安装提示)
    os.makedirs(目标目录, exist_ok=True)
    try:
        return snapshot_download(repo_id=模型ID, local_dir=目标目录)
    except TypeError:
        return snapshot_download(repo_id=模型ID, cache_dir=模型库目录)


def 下载模型(模型ID: str, 镜像源: str, 量化: str, 进度回调=None) -> dict:
    """从指定镜像源下载模型到 数据/模型库/<模型名>，完成后登记模型索引。

    参数:
        模型ID: 模型仓库 ID，如 "Qwen/Qwen2.5-3B-Instruct"。
        镜像源: 取值见配置项 模型.可用镜像源（"魔搭社区" / "HuggingFace镜像"）。
        量化: 量化档位，如 "fp16" / "4bit"（提示用，下载内容以仓库为准）。
        进度回调: 可选回调函数 `进度回调(进度, 消息)`，下载期间每秒调用。

    返回:
        dict：{"成功": True, "路径": ..., "模型ID": ...} 或 {"成功": False, "错误": ...}。
    """
    下载进度[模型ID] = {"百分比": 0.0, "消息": "准备下载", "完成": False}

    def 上报进度(百分比: float, 消息: str) -> None:
        下载进度[模型ID] = {"百分比": round(百分比, 2), "消息": 消息, "完成": bool(百分比 >= 1.0)}
        if 进度回调 is not None:
            try:
                进度回调(百分比, 消息)
            except Exception:
                pass

    if not isinstance(模型ID, str) or not 模型ID.strip():
        错误信息 = "模型ID不能为空"
        上报进度(0.0, 错误信息)
        return {"成功": False, "错误": 错误信息, "模型ID": 模型ID}
    镜像源 = 镜像源 or 配置管理.获取配置项("模型.默认镜像源", "魔搭社区")
    可用镜像源 = 配置管理.获取配置项("模型.可用镜像源", ["魔搭社区", "HuggingFace镜像"])
    if 镜像源 not in 可用镜像源:
        错误信息 = f"不支持的镜像源：{镜像源}，可选：{'、'.join(可用镜像源)}"
        上报进度(0.0, 错误信息)
        return {"成功": False, "错误": 错误信息, "模型ID": 模型ID}

    模型库目录 = _获取模型库目录()
    目标目录 = os.path.join(模型库目录, _生成模型目录名(模型ID))
    上报进度(0.0, f"准备从 {镜像源} 下载 {模型ID}（量化档位：{量化}）")

    停止事件 = threading.Event()
    进度线程 = None
    if 进度回调 is not None:
        进度线程 = threading.Thread(
            target=_周期进度上报, args=(上报进度, 停止事件, 模型ID, 镜像源), daemon=True
        )
        进度线程.start()
    try:
        if 镜像源 == "魔搭社区":
            下载路径 = _下载魔搭(模型ID, 模型库目录, 目标目录)
        else:
            下载路径 = _下载镜像(模型ID, 模型库目录, 目标目录)
        os.makedirs(下载路径, exist_ok=True)
        停止事件.set()
        上报进度(1.0, "下载完成")
        # 刷新并登记模型索引
        try:
            登记模型(扫描本地模型())
        except Exception:
            pass
        总大小MB = _目录大小MB(下载路径)
        return {
            "成功": True,
            "状态": "已完成",
            "模型ID": 模型ID,
            "路径": 下载路径,
            "本地路径": 下载路径,
            "已下载MB": 总大小MB,
            "总计MB": 总大小MB,
        }
    except Exception as 错误:
        停止事件.set()
        错误信息 = str(错误)
        上报进度(0.0, f"下载失败：{错误信息}")
        return {"成功": False, "错误": 错误信息, "模型ID": 模型ID}
    finally:
        if 进度线程 is not None:
            进度线程.join(timeout=0.1)


# ==================================================================
# 四、性能评估
# ==================================================================

def 性能评估(模型路径: str) -> dict:
    """加载模型跑一次冒烟生成（32 token），输出性能评估报告。

    加载策略：fp16 CUDA → 显存不足降级 4bit → 再失败降级 CPU，并记录 降级原因。
    无论成败，结束后释放模型与显存（del + torch.cuda.empty_cache()）。

    参数:
        模型路径: 本地模型绝对路径（须包含 config.json）。

    返回:
        dict：完整评估报告；加载失败返回 {"成功": False, "错误": ...}。
    """
    if not isinstance(模型路径, str) or not 模型路径.strip():
        return {"成功": False, "错误": "模型路径为空"}
    模型路径 = os.path.abspath(模型路径)
    if not os.path.isdir(模型路径) or not os.path.exists(os.path.join(模型路径, "config.json")):
        return {"成功": False, "错误": f"模型路径无效或缺少 config.json：{模型路径}"}
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as 错误:
        return {
            "成功": False,
            "错误": f"缺少 torch/transformers 依赖：{错误}，请运行: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch transformers",
        }

    使用CUDA = bool(torch.cuda.is_available())
    模型 = None
    tokenizer = None
    降级原因 = ""
    try:
        # 加载 tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True, use_fast=False)
            except Exception as 错误:
                return {"成功": False, "错误": f"模型 tokenizer 加载失败：{错误}"}

        # 模型加载：fp16 CUDA → 4bit → CPU 逐级降级
        if 使用CUDA:
            try:
                模型 = AutoModelForCausalLM.from_pretrained(
                    模型路径, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
                )
            except Exception as 错误:
                降级原因 = "显存不足" if "out of memory" in str(错误).lower() else f"fp16 加载失败：{错误}"
        if 模型 is None and 使用CUDA:
            try:
                模型 = AutoModelForCausalLM.from_pretrained(
                    模型路径, load_in_4bit=True, device_map="auto", trust_remote_code=True
                )
                降级原因 = f"{降级原因}，已降级为 4bit 量化加载"
            except Exception as 错误:
                降级原因 = f"{降级原因}；4bit 加载失败：{错误}"
        if 模型 is None:
            try:
                模型 = AutoModelForCausalLM.from_pretrained(模型路径, trust_remote_code=True)
                降级原因 = (降级原因 + "，") if 降级原因 else ""
                降级原因 += "已降级为 CPU 加载"
            except Exception as 错误:
                return {"成功": False, "错误": f"模型加载失败：{错误}"}

        # 冒烟生成
        编码 = tokenizer(评估提示词, return_tensors="pt")
        输入长度 = int(编码["input_ids"].shape[1])
        模型设备 = next(模型.parameters()).device
        显存前 = 0
        if 模型设备.type.startswith("cuda"):
            编码 = {键: 值.to(模型设备) for 键, 值 in 编码.items()}
            显存前 = torch.cuda.memory_allocated(模型设备)
        try:
            开始 = time.perf_counter()
            with torch.no_grad():
                输出 = 模型.generate(
                    **编码,
                    max_new_tokens=评估生成Token数,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            结束 = time.perf_counter()
        except Exception as 错误:
            return {"成功": False, "错误": f"推理生成失败：{错误}"}

        生成耗时秒 = max(结束 - 开始, 1e-6)
        新token数 = max(1, int(输出.shape[1]) - 输入长度)
        生成速度tok每秒 = 新token数 / 生成耗时秒
        显存后 = torch.cuda.memory_allocated(模型设备) if 模型设备.type.startswith("cuda") else 显存前
        显存占用MB = (显存后 - 显存前) / 1024 / 1024
        生成文本 = ""
        try:
            生成文本 = tokenizer.decode(输出[0][输入长度:], skip_special_tokens=True)
        except Exception:
            pass

        # 参数量与微调可行性
        参数量亿, 参数来源, _ = _解析模型路径参数量(模型路径)
        可推理 = True
        可微调 = True
        备注 = ""
        if 硬件检测 is not None and hasattr(硬件检测, "预估显存"):
            try:
                预估 = 硬件检测.预估显存(参数量亿 or 0, "4bit")
                可微调 = bool(预估.get("可微调", True))
                备注 = 预估.get("建议", "") if not 可微调 else ""
            except Exception as 错误:
                备注 = f"微调显存预估失败：{错误}"
        else:
            备注 = "硬件检测模块未就绪，微调能力未评估"

        return {
            "成功": True,
            "模型路径": 模型路径,
            "参数量亿": 参数量亿,
            "参数来源": 参数来源,
            "显存占用MB": round(显存占用MB, 1),
            "生成耗时秒": round(生成耗时秒, 2),
            "评估时间秒": round(生成耗时秒, 2),
            "生成速度tok/s": round(生成速度tok每秒, 2),
            "生成速度Token每秒": round(生成速度tok每秒, 2),
            "可推理": 可推理,
            "可微调": 可微调,
            "降级原因": 降级原因,
            "备注": 备注,
            "生成文本": 生成文本[:200],
        }
    except Exception as 错误:
        return {"成功": False, "错误": f"性能评估过程异常：{错误}"}
    finally:
        # 无论成败都释放模型与显存
        try:
            del 模型
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        if 使用CUDA:
            torch.cuda.empty_cache()


# ==================================================================
# 五、HTTP 路由
# ==================================================================

def _后台下载任务(模型ID: str, 镜像源: str, 量化: str) -> None:
    """后台执行下载并同步全局 下载进度（下载模型 内部也会维护，此处兜底）。"""

    def 进度回调(百分比: float, 消息: str) -> None:
        下载进度[模型ID] = {"百分比": round(百分比, 2), "消息": 消息, "完成": bool(百分比 >= 1.0)}

    try:
        结果 = 下载模型(模型ID, 镜像源, 量化, 进度回调=进度回调)
        if not 结果.get("成功"):
            下载进度[模型ID] = {"百分比": 0.0, "消息": 结果.get("错误", "下载失败"), "完成": True}
    except Exception as 错误:
        下载进度[模型ID] = {"百分比": 0.0, "消息": f"下载异常：{错误}", "完成": True}


def 注册路由(app) -> None:
    """注册模型管理模块的 HTTP 路由（挂载到 FastAPI 应用）。

    接口:
        GET  /api/模型            扫描本地模型
        POST /api/模型/扫描       重新扫描并登记索引
        POST /api/模型/下载       提交后台下载任务（模型ID/镜像源/量化）
        GET  /api/模型/下载/进度  查询最近下载进度（?模型ID=xxx）
        POST /api/模型/评估       性能评估（模型路径）
    """
    try:
        from fastapi import BackgroundTasks
        from pydantic import BaseModel
    except Exception as 错误:
        print(f"[模型管理] 缺少 FastAPI 依赖，跳过路由注册：{错误}")
        return

    class 下载请求(BaseModel):
        模型ID: str
        镜像源: str = ""
        量化: str = "fp16"

    class 评估请求(BaseModel):
        模型路径: str

    @app.get("/api/模型")
    def 获取本地模型() -> dict:
        try:
            return {"成功": True, "模型列表": 扫描本地模型()}
        except Exception as 错误:
            return {"成功": False, "错误": f"扫描模型失败：{错误}"}

    @app.post("/api/模型/扫描")
    def 重新扫描并登记() -> dict:
        try:
            模型列表 = 扫描本地模型()
            登记成功 = 登记模型(模型列表)
            return {"成功": True, "登记": 登记成功, "模型列表": 模型列表}
        except Exception as 错误:
            return {"成功": False, "错误": f"扫描登记失败：{错误}"}

    @app.post("/api/模型/下载")
    def 提交下载任务(请求: 下载请求, 后台任务: BackgroundTasks) -> dict:
        try:
            模型ID = 请求.模型ID.strip()
            if not 模型ID:
                return {"成功": False, "错误": "模型ID不能为空"}
            镜像源 = 请求.镜像源 or 配置管理.获取配置项("模型.默认镜像源", "魔搭社区")
            可用镜像源 = 配置管理.获取配置项("模型.可用镜像源", ["魔搭社区", "HuggingFace镜像"])
            if 镜像源 not in 可用镜像源:
                return {"成功": False, "错误": f"不支持的镜像源：{镜像源}，可选：{'、'.join(可用镜像源)}"}
            下载进度[模型ID] = {"百分比": 0.0, "消息": "任务已提交，等待后台执行", "完成": False}
            后台任务.add_task(_后台下载任务, 模型ID, 镜像源, 请求.量化)
            return {"成功": True, "消息": "下载任务已启动", "模型ID": 模型ID}
        except Exception as 错误:
            return {"成功": False, "错误": f"提交下载任务失败：{错误}"}

    @app.get("/api/模型/下载/进度")
    def 查询下载进度(模型ID: str = "") -> dict:
        try:
            if 模型ID:
                return {
                    "成功": True,
                    "模型ID": 模型ID,
                    "进度": 下载进度.get(模型ID, {"百分比": 0.0, "消息": "未找到该模型的下载记录", "完成": False}),
                }
            return {"成功": True, "进度列表": 下载进度}
        except Exception as 错误:
            return {"成功": False, "错误": f"查询下载进度失败：{错误}"}

    @app.post("/api/模型/评估")
    def 评估模型性能(请求: 评估请求) -> dict:
        try:
            return 性能评估(请求.模型路径)
        except Exception as 错误:
            return {"成功": False, "错误": f"性能评估失败：{错误}"}
