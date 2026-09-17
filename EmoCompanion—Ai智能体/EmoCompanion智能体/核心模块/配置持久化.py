# -*- coding: utf-8 -*-
"""配置持久化：模型/音色别名（重命名）、风格预设、最近使用、发音纠正，存于 配置文件.json。

配置文件键全部为中文；API 层（核心模块/后端服务.py）负责把中文配置键映射为英文接口字段。
"""
import copy
import json
import threading

from 环境配置 import 数据目录

配置文件路径 = 数据目录() / "配置文件.json"

默认配置 = {
    "模型别名": {},          # 模型ID -> 显示别名（重命名结果）
    "音色别名": {},          # 音色ID -> 显示别名
    "风格预设": [            # 全局风格指令预设
        {"名称": "默认·温柔自然",
         "指令": "用温柔、清澈、带一点俏皮的少女语气，自然地说话。"},
        {"名称": "睡前故事",
         "指令": "语速缓慢轻柔，像哄孩子入睡一样，声音低柔温暖，尾音放轻。"},
        {"名称": "情绪递进·独白",
         "指令": "女声独白，情绪层层递进：开头委屈不安、声音微微发颤；"
                 "讲到害怕时呼吸加重；然后如释重负、带着笑意；"
                 "最后转为俏皮撒娇又带一点温柔。语速自然，感情真挚。"},
    ],
    "最近使用": {
        "模型": "qwen-audio-3.0-tts-plus",
        "音色": "qwen-audio-3.0-tts-plus-emocompanion-c6cf949d19734ab5a5552a9c5ce2da9f",
        "格式": "wav",
        "采样率": 48000,
    },
    "发音纠正": [            # 发音纠正表：词 -> 带声调拼音（合成时自动注音）
        {"词": "EmoCompanion", "拼音": "yuan3 yuan4"},
    ],
    "隐藏音色": [],           # 被删除/隐藏的音色ID列表（列表页不再展示）
}

_锁 = threading.Lock()
_数据 = None


def _深合并(基础, 覆盖):
    """深合并：字典递归合并，其余类型直接覆盖。"""
    输出 = copy.deepcopy(基础)
    for 键, 值 in (覆盖 or {}).items():
        if isinstance(值, dict) and isinstance(输出.get(键), dict):
            输出[键] = _深合并(输出[键], 值)
        else:
            输出[键] = 值
    return 输出


def 读取配置():
    """读取（并缓存）配置；文件不存在或损坏时回退到默认配置。"""
    global _数据
    with _锁:
        if _数据 is None:
            if 配置文件路径.exists():
                try:
                    原始 = json.loads(配置文件路径.read_text(encoding="utf-8"))
                except Exception:
                    原始 = {}
                _数据 = _深合并(默认配置, 原始)
            else:
                _数据 = copy.deepcopy(默认配置)
        return _数据


def 保存配置():
    """把当前配置写回 配置文件.json（utf-8、ensure_ascii=False）。"""
    with _锁:
        配置文件路径.write_text(
            json.dumps(_数据, ensure_ascii=False, indent=2), encoding="utf-8")


def 模型别名(模型ID):
    return 读取配置()["模型别名"].get(模型ID, 模型ID)


def 音色别名(音色ID):
    return 读取配置()["音色别名"].get(音色ID, 音色ID)


def 重命名模型(模型ID, 别名):
    配置 = 读取配置()
    别名 = (别名 or "").strip()
    if 别名:
        配置["模型别名"][模型ID] = 别名
    else:
        配置["模型别名"].pop(模型ID, None)
    保存配置()


def 重命名音色(音色ID, 别名):
    配置 = 读取配置()
    别名 = (别名 or "").strip()
    if 别名:
        配置["音色别名"][音色ID] = 别名
    else:
        配置["音色别名"].pop(音色ID, None)
    保存配置()


def 风格预设():
    return 读取配置()["风格预设"]


def 新增风格预设(名称, 指令):
    配置 = 读取配置()
    名称 = (名称 or "").strip()
    if not 名称:
        raise ValueError("预设名称不能为空")
    for 预设 in 配置["风格预设"]:
        if 预设["名称"] == 名称:
            预设["指令"] = 指令
            保存配置()
            return 预设
    预设 = {"名称": 名称, "指令": 指令}
    配置["风格预设"].append(预设)
    保存配置()
    return 预设


def 删除风格预设(名称):
    配置 = 读取配置()
    配置["风格预设"] = [预设 for 预设 in 配置["风格预设"] if 预设["名称"] != 名称]
    保存配置()


def 最近使用():
    return 读取配置()["最近使用"]


def 记录最近使用(**参数):
    配置 = 读取配置()
    配置["最近使用"].update({键: 值 for 键, 值 in 参数.items() if 值 is not None})
    保存配置()


def 发音纠正表():
    return 读取配置()["发音纠正"]


def 新增发音纠正(词, 拼音):
    """新增或更新一条发音纠正。返回 (条目, 是否新增)。"""
    配置 = 读取配置()
    词 = (词 or "").strip()
    拼音 = (拼音 or "").strip()
    if not 词 or not 拼音:
        raise ValueError("词和拼音都不能为空")
    for 条目 in 配置["发音纠正"]:
        if 条目["词"] == 词:
            条目["拼音"] = 拼音
            保存配置()
            return 条目, False
    条目 = {"词": 词, "拼音": 拼音}
    配置["发音纠正"].append(条目)
    保存配置()
    return 条目, True


def 删除发音纠正(词):
    配置 = 读取配置()
    配置["发音纠正"] = [条目 for 条目 in 配置["发音纠正"] if 条目["词"] != 词]
    保存配置()


def 隐藏音色列表():
    """已隐藏（删除）的音色ID列表。"""
    return 读取配置()["隐藏音色"]


def 隐藏音色(音色ID):
    """把音色ID加入隐藏列表（去重）并保存，返回更新后的列表。"""
    配置 = 读取配置()
    音色ID = (音色ID or "").strip()
    if 音色ID and 音色ID not in 配置["隐藏音色"]:
        配置["隐藏音色"].append(音色ID)
        保存配置()
    return 配置["隐藏音色"]


def 取消隐藏音色(音色ID):
    """从隐藏列表移除音色ID并保存，返回更新后的列表。"""
    配置 = 读取配置()
    配置["隐藏音色"] = [ID for ID in 配置["隐藏音色"] if ID != 音色ID]
    保存配置()
    return 配置["隐藏音色"]
