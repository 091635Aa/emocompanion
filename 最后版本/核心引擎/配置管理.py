# -*- coding: utf-8 -*-
"""
配置管理模块
==============
负责全局配置文件的读取、保存、重置与路径解析，是全项目唯一配置入口。

- 配置文件：<项目根>/配置/系统配置.json
- 所有路径字段在读取时自动解析为相对项目根的绝对路径
- 模块级变量 `配置` 作为全项目共享的配置缓存
- 各业务模块统一通过 `获取配置项("模块名.字段名")` 读取配置，
  不硬编码任何输入/输出/格式/时间参数
"""

import json
import os
from copy import deepcopy

# 项目根目录：本文件位于 <项目根>/核心引擎/ 下，向上两级即为项目根
项目根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
配置文件路径 = os.path.join(项目根, "配置", "系统配置.json")

# 需要解析为绝对路径的字段名（出现在各模块配置段中）
路径字段 = {
    "模型库目录",
    "输入目录",
    "分割片段目录",
    "打标结果目录",
    "日记目录",
    "输出目录",
    "数据包目录",
}

# 默认配置：与 配置/系统配置.json 保持一致，文件缺失时据此创建
默认配置 = {
    "系统": {
        "项目名称": "一体化全流程AI应用",
        "版本": "0.1.0",
        "默认端口": 8765,
        "自动打开浏览器": True,
        "语言": "zh-CN",
    },
    "硬件": {
        "强制NVIDIA检查": True,
        "显存预留MB": 1024,
        "支持AMD": False,
    },
    "模型": {
        "模型库目录": "数据/模型库",
        "默认镜像源": "魔搭社区",
        "可用镜像源": ["魔搭社区", "HuggingFace镜像"],
        "ASR模型": "",
        "ASR模型类型": "全模态(音频解码器)",
        "教师模型": "",
        "教师模型类型": "本地30B千问",
    },
    "数据预处理": {
        "输入目录": "数据/上传",
        "分割片段目录": "数据/分割片段",
        "按话题分割": True,
        "最小片段秒": 5,
        "最大片段秒": 300,
        "清洗去重": True,
    },
    "打标": {
        "打标结果目录": "数据/打标结果",
        "自动打标": True,
        "人工复核": True,
        "打标维度": ["情感维度", "内容标签", "风格标签"],
    },
    "日记": {
        "日记目录": "数据/日记",
        "起始年龄": 7,
        "当前年龄": 18,
        "每篇字数最小": 300,
        "每篇字数最大": 2000,
        "日记风格": "口语化温暖",
        "生成数量上限": 100,
    },
    "微调": {
        "输出目录": "数据/微调输出",
        "数据包目录": "数据/微调数据包",
        "默认轮数": 3,
        "默认学习率": 0.0002,
        "默认批量": 4,
        "量化档位": ["fp16", "4bit"],
        "默认量化": "auto",
        "启用情感微调": True,
        "启用记忆微调": True,
        "启用身份微调": True,
        "模型组织方式": "一角色一模型",
        "允许单模型多记忆": False,
    },
    "推理": {
        "默认架构": "V通用架构",
        "可用架构": ["V1简单回响", "V通用架构"],
        "默认λ": 0.08,
        "默认γ": 0.07,
        "默认τ": 0.09,
        "最大新Token": 256,
    },
    "达标评估": {
        "评估基准": "图灵测试简化版",
        "通过门槛": 0.5,
    },
}

# 模块级配置缓存（路径字段已解析为绝对路径）
配置 = {}


def 解析路径(相对路径: str) -> str:
    """把相对项目根的路径解析为绝对路径。

    参数:
        相对路径: 相对项目根（j:\\最后版本！）的路径，如 "数据/模型库"。

    返回:
        绝对路径字符串；若传入的是空串或已是绝对路径则原样返回。
    """
    if not 相对路径:
        return 相对路径
    if os.path.isabs(相对路径):
        return os.path.abspath(相对路径)
    return os.path.abspath(os.path.join(项目根, 相对路径))


def _递归合并(目标: dict, 来源: dict) -> None:
    """把来源字典递归合并进目标字典（来源覆盖目标，用于补齐新增配置项）。"""
    for 键, 值 in 来源.items():
        if 键 in 目标 and isinstance(目标[键], dict) and isinstance(值, dict):
            _递归合并(目标[键], 值)
        else:
            目标[键] = deepcopy(值)


def _解析配置内路径(配置片段: dict) -> dict:
    """递归遍历配置，把所有路径字段解析为绝对路径。"""
    结果 = {}
    for 键, 值 in 配置片段.items():
        if 键 in 路径字段 and isinstance(值, str):
            结果[键] = 解析路径(值)
        elif isinstance(值, dict):
            结果[键] = _解析配置内路径(值)
        else:
            结果[键] = deepcopy(值)
    return 结果


def _还原配置内路径(配置片段: dict) -> dict:
    """保存前把解析过的绝对路径还原为相对路径，避免污染配置文件。"""
    结果 = {}
    for 键, 值 in 配置片段.items():
        if 键 in 路径字段 and isinstance(值, str) and os.path.isabs(值):
            try:
                if os.path.commonpath([项目根, os.path.abspath(值)]) == 项目根:
                    结果[键] = os.path.relpath(值, 项目根).replace("\\", "/")
                    continue
            except ValueError:
                pass
            结果[键] = 值
        elif isinstance(值, dict):
            结果[键] = _还原配置内路径(值)
        else:
            结果[键] = deepcopy(值)
    return 结果


def 读取配置() -> dict:
    """读取配置文件；文件不存在时自动用默认配置创建。

    返回:
        完整配置字典（路径字段已解析为绝对路径），并刷新模块级缓存 `配置`。

    异常:
        RuntimeError: 配置文件存在但内容损坏（JSON 解析失败）时抛出。
    """
    global 配置
    if os.path.exists(配置文件路径):
        try:
            with open(配置文件路径, "r", encoding="utf-8-sig") as 文件:
                原始配置 = json.load(文件)
        except (json.JSONDecodeError, OSError) as 错误:
            raise RuntimeError(f"配置文件损坏或无法读取：{配置文件路径}（{错误}）") from 错误
    else:
        # 首次运行：自动创建配置文件
        原始配置 = deepcopy(默认配置)
        _写入文件(_还原配置内路径(原始配置))
    # 与默认配置递归合并，补齐版本升级带来的新字段
    合并后 = deepcopy(默认配置)
    _递归合并(合并后, 原始配置)
    配置 = _解析配置内路径(合并后)
    return 配置


def _写入文件(待写入: dict) -> None:
    """把字典以 UTF-8 写入配置文件（无 BOM，中文不转义）。"""
    os.makedirs(os.path.dirname(配置文件路径), exist_ok=True)
    with open(配置文件路径, "w", encoding="utf-8") as 文件:
        json.dump(待写入, 文件, ensure_ascii=False, indent=2)


def 保存配置(新配置: dict) -> bool:
    """把配置写回配置文件。

    参数:
        新配置: 需要保存的配置字典（允许只含部分字段，缺失字段按默认值补齐）。

    返回:
        bool：写入成功返回 True，失败（非法参数或 IO 错误）返回 False。
    """
    global 配置
    if not isinstance(新配置, dict):
        return False
    try:
        # 与默认配置合并后再写入，避免缺字段导致配置丢失
        合并后 = deepcopy(默认配置)
        _递归合并(合并后, 新配置)
        _写入文件(_还原配置内路径(合并后))
        配置 = _解析配置内路径(合并后)
        return True
    except OSError:
        return False


def 获取配置项(路径: str, 默认值=None):
    """按点分路径获取配置项。

    参数:
        路径: 点分路径，如 "系统.默认端口"、"模型.可用镜像源"。
        默认值: 路径不存在时返回的值。

    返回:
        配置项的值；路径不存在返回 `默认值`。
    """
    global 配置
    if not 配置:
        读取配置()
    当前 = 配置
    for 键 in 路径.split("."):
        if isinstance(当前, dict) and 键 in 当前:
            当前 = 当前[键]
        else:
            return 默认值
    return 当前


def 重置配置() -> bool:
    """恢复默认配置并写回文件。

    返回:
        bool：重置成功返回 True，失败返回 False。
    """
    global 配置
    配置 = _解析配置内路径(deepcopy(默认配置))
    return 保存配置(配置)


# 模块加载即完成首次读取，保证 `配置` 缓存立即可用
if not 配置:
    读取配置()
