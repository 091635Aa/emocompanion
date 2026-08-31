# -*- coding: utf-8 -*-
"""
结果解析模块

负责将模型输出的原始文本解析为结构化的 RPG 素材打标结果字典：
- 清理模型输出（去除 markdown 代码块围栏、裁剪首个 { 到末个 } 之间的内容）
- json 解析失败抛 ValueError（错误信息含原文片段）
- 校验必需字段：名称 / 类型 / 内容描述 / 视觉特征(list) / 标签(list) / 置信度
- 类型兜底：模型输出的类型为空时用 素材信息.类型 填充
- 合并 素材信息 提供的 素材ID、来源文件、切割坐标、图片路径、缩略图路径、打标时间 等字段
"""

import sys
import os
import json
import re

# 将项目根目录加入模块搜索路径，保证中文模块名可导入
项目根目录 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if 项目根目录 not in sys.path:
    sys.path.append(项目根目录)

# 统一 stdout 编码，避免中文打印乱码
sys.stdout.reconfigure(encoding="utf-8")

# 打标结果必需字段（缺失或类型非法一律抛 ValueError）
必填字段 = ["名称", "类型", "内容描述", "视觉特征", "标签", "置信度"]

# 数组类型字段（非数组一律判为非法）
数组字段 = ["视觉特征", "标签"]

# 从 素材信息 合并进结果的元数据字段（模型不负责输出，由分割/流水线提供）
元数据字段 = [
    "素材ID", "来源文件", "切割坐标", "图片路径", "缩略图路径", "打标时间",
    "序号", "行", "列",
]


def 截取片段(文本, 最大长度=200):
    """截取文本片段（单行化 + 限长）用于错误提示，避免异常信息过长。"""
    if not isinstance(文本, str):
        return repr(文本)[:最大长度]
    文本 = 文本.replace("\r", " ").replace("\n", " ").strip()
    if len(文本) > 最大长度:
        return 文本[:最大长度] + "……"
    return 文本


def 提取JSON对象(原始文本):
    """
    从模型输出文本中清理并解析 JSON 对象；全部失败时返回 None。

    清理顺序：
    1. 直接 json.loads(去头尾空白)；
    2. 提取 markdown 代码块（```json ... ``` 或 ``` ... ```）内容后解析；
    3. 裁剪首个 '{' 到末个 '}' 之间的子串后解析。
    """
    if not isinstance(原始文本, str):
        return None
    文本 = 原始文本.strip()
    if not 文本:
        return None

    # 1. 直接解析
    try:
        return json.loads(文本)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. 去除 ```json ... ``` / ``` ... ``` 围栏
    代码块匹配 = re.search(r"```(?:json)?\s*([\s\S]*?)```", 文本, re.IGNORECASE)
    if 代码块匹配:
        try:
            return json.loads(代码块匹配.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. 裁剪首个 '{' 到末个 '}' 之间的内容
    起始位置 = 文本.find("{")
    结束位置 = 文本.rfind("}")
    if 起始位置 != -1 and 结束位置 > 起始位置:
        try:
            return json.loads(文本[起始位置:结束位置 + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def 清洗字符串列表(值):
    """清洗为字符串列表：过滤非字符串与空白项，并去重（保留首次出现顺序）。"""
    if not isinstance(值, list):
        return []
    结果 = []
    已见 = set()
    for 项 in 值:
        if isinstance(项, str) and 项.strip():
            标签 = 项.strip()
            if 标签 not in 已见:
                已见.add(标签)
                结果.append(标签)
    return 结果


def 解析打标结果(原始文本, 素材信息=None):
    """
    将模型输出的原始文本解析为完整打标结果字典。

    入参：
    - 原始文本：模型返回的原始文本（可能含 markdown 围栏、前后缀说明）
    - 素材信息：可选 dict，由 素材分割/打标流水线 提供（含 类型、素材ID、来源文件、
      切割坐标、图片路径、缩略图路径、打标时间 等）

    返回：完整结果 dict（模型字段 + 素材信息元数据字段）。
    解析失败 / 校验失败抛 ValueError，错误信息包含原文片段。
    """
    素材信息 = 素材信息 if isinstance(素材信息, dict) else {}
    原文片段 = 截取片段(原始文本)

    # 1-2. 清理并解析 JSON；失败抛 ValueError（含原文片段）
    原始对象 = 提取JSON对象(原始文本)
    if not isinstance(原始对象, dict):
        raise ValueError("模型输出无法解析为 JSON 对象（原文片段：{}）".format(原文片段))

    # 3. 校验必需字段：名称/类型/内容描述/视觉特征(list)/标签(list)/置信度
    for 字段 in 必填字段:
        if 字段 not in 原始对象 or 原始对象[字段] is None:
            raise ValueError("打标结果缺少必需字段：{}（原文片段：{}）".format(字段, 原文片段))

    # 字符串字段校验
    名称 = 原始对象.get("名称")
    if not isinstance(名称, str) or not 名称.strip():
        raise ValueError("打标结果 名称 不是非空字符串：{}（原文片段：{}）".format(repr(名称), 原文片段))
    内容描述 = 原始对象.get("内容描述")
    if not isinstance(内容描述, str):
        raise ValueError("打标结果 内容描述 不是字符串：{}（原文片段：{}）".format(repr(内容描述), 原文片段))

    # 数组字段校验
    清洗字段 = {}
    for 字段 in 数组字段:
        值 = 原始对象.get(字段)
        if not isinstance(值, list):
            raise ValueError("打标结果 {} 不是数组：{}（原文片段：{}）".format(字段, repr(值), 原文片段))
        清洗字段[字段] = 清洗字符串列表(值)

    # 置信度：必须为数字（兼容数字字符串），收敛到 0-1 范围
    置信度 = 原始对象.get("置信度")
    if isinstance(置信度, bool) or not isinstance(置信度, (int, float)):
        try:
            置信度 = float(置信度)
        except (TypeError, ValueError):
            raise ValueError("打标结果 置信度 不是数字：{}（原文片段：{}）".format(repr(置信度), 原文片段))
    置信度 = max(0.0, min(1.0, 置信度))

    # 4. 类型兜底：模型输出的类型为空时用 素材信息.类型 填充
    类型 = 原始对象.get("类型")
    if not isinstance(类型, str) or not 类型.strip():
        类型 = 素材信息.get("类型", "")
    if not isinstance(类型, str) or not 类型.strip():
        raise ValueError("打标结果 类型 为空且素材信息未提供类型（原文片段：{}）".format(原文片段))

    # 适用场景：非必填，模型给出则保留（清洗为字符串）
    适用场景 = 原始对象.get("适用场景", "")
    if not isinstance(适用场景, str):
        适用场景 = ""

    # 5. 合并 素材信息 提供的元数据字段（素材ID/来源文件/切割坐标/图片路径/缩略图路径/打标时间 等）
    结果 = {}
    for 字段 in 元数据字段:
        if 字段 in 素材信息 and 素材信息[字段] is not None:
            结果[字段] = 素材信息[字段]

    # 6. 覆盖模型输出字段（类型已兜底）
    结果.update({
        "名称": 名称.strip(),
        "类型": 类型.strip(),
        "内容描述": 内容描述.strip(),
        "视觉特征": 清洗字段["视觉特征"],
        "标签": 清洗字段["标签"],
        "适用场景": 适用场景.strip(),
        "置信度": 置信度,
    })

    # 保留模型输出的其余字段（如未来扩充的补充字段），保证结果完整
    已处理字段 = {"名称", "类型", "内容描述", "视觉特征", "标签", "适用场景", "置信度"}
    for 键, 值 in 原始对象.items():
        if 键 not in 已处理字段:
            结果[键] = 值

    return 结果


if __name__ == "__main__":
    print("=" * 60)
    print("结果解析模块自测")
    print("=" * 60)

    # 示例素材信息（模拟 素材分割/打标流水线 传入）
    示例素材信息 = {
        "素材ID": "T0003",
        "类型": "角色精灵图",
        "来源文件": "Actor1.png",
        "切割坐标": {"行": 0, "列": 2, "x": 96, "y": 0, "宽": 48, "高": 48},
        "序号": 3,
        "图片路径": "数据层\\分割素材\\Actor1_角色1_帧03.png",
        "缩略图路径": "数据层\\缩略图\\T0003.png",
        "打标时间": "2026-08-07 10:00:00",
    }

    print("\n【测试一】合法样例：markdown 围栏 + 前后缀文字 + 类型缺失（应兜底为素材信息.类型）")
    合法样例 = """好的，以下是该素材的分析结果：
```json
{
    "名称": "红衣骑士",
    "类型": "",
    "内容描述": "红色披风的像素骑士，手持长剑站立姿态。",
    "视觉特征": ["像素风", "红色披风", "银白盔甲", "手持长剑", "站立姿态"],
    "标签": ["骑士", "战士", "红衣", "长剑", "角色精灵图", "像素风", "战斗"],
    "适用场景": "可用于 RPG 游戏中的战士职业角色。",
    "置信度": 0.92
}
```
以上分析完毕。"""
    结果一 = 解析打标结果(合法样例, 示例素材信息)
    print(json.dumps(结果一, ensure_ascii=False, indent=2))
    assert 结果一["类型"] == "角色精灵图", "类型兜底失败"
    assert 结果一["素材ID"] == "T0003", "元数据合并失败"
    assert 结果一["来源文件"] == "Actor1.png", "元数据合并失败"
    assert 结果一["置信度"] == 0.92, "置信度解析失败"
    assert isinstance(结果一["视觉特征"], list) and isinstance(结果一["标签"], list), "数组字段失败"
    print("\n测试一通过：类型兜底、元数据合并、字段清洗均正常")

    print("\n【测试二】非法样例：完全不是 JSON 的文本（应抛 ValueError 且含原文片段）")
    try:
        解析打标结果("模型出现幻觉，输出了这段没有任何 JSON 结构的废话文本。")
        print("测试二失败：未抛异常")
    except ValueError as 异常:
        print("符合预期抛 ValueError：", 异常)
        assert "原文片段" in str(异常), "错误信息缺少原文片段"
        print("测试二通过")

    print("\n【测试三】非法样例：缺少必需字段 标签（应抛 ValueError）")
    try:
        解析打标结果(json.dumps({
            "名称": "红衣骑士", "类型": "角色精灵图",
            "内容描述": "红色披风的像素骑士。",
            "视觉特征": ["像素风", "红色披风"], "置信度": 0.9,
        }, ensure_ascii=False))
        print("测试三失败：未抛异常")
    except ValueError as 异常:
        print("符合预期抛 ValueError：", 异常)
        print("测试三通过")

    print("\n【测试四】非法样例：视觉特征 不是数组（应抛 ValueError）")
    try:
        解析打标结果(json.dumps({
            "名称": "红衣骑士", "类型": "角色精灵图",
            "内容描述": "红色披风的像素骑士。",
            "视觉特征": "像素风，红色披风", "标签": ["骑士"], "置信度": 0.9,
        }, ensure_ascii=False))
        print("测试四失败：未抛异常")
    except ValueError as 异常:
        print("符合预期抛 ValueError：", 异常)
        print("测试四通过")

    print("\n【测试五】非法样例：置信度 不是数字（应抛 ValueError）")
    try:
        解析打标结果(json.dumps({
            "名称": "红衣骑士", "类型": "角色精灵图",
            "内容描述": "红色披风的像素骑士。",
            "视觉特征": ["像素风"], "标签": ["骑士"], "置信度": "很高",
        }, ensure_ascii=False))
        print("测试五失败：未抛异常")
    except ValueError as 异常:
        print("符合预期抛 ValueError：", 异常)
        print("测试五通过")

    print("\n全部自测通过 ✓")
