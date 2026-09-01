# -*- coding: utf-8 -*-
"""R12 离线回归：RPG 素材打标 LLM 输出解析器（纯逻辑，无 GPU/LLM）
覆盖：提取JSON对象（围栏/前后缀裁剪）、必填/数组/置信度校验、类型兜底、元数据合并。
运行：python3 /workspace/_regression/test_x_rpg_resultparse.py
"""
import sys, os, json

DIR = "/workspace/打标_RPG"
sys.path.insert(0, DIR)

from 业务逻辑层.结果解析 import (提取JSON对象, 清洗字符串列表,  # noqa: E402
                             解析打标结果, 必填字段)

PASS = 0
FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")

素材 = {"素材ID": "T0003", "类型": "角色精灵图", "来源文件": "Actor1.png",
        "切割坐标": {"行": 0, "列": 2}}

print("== 提取JSON对象 三种兜底 ==")
check("直接 JSON", 提取JSON对象('{"a":1}') == {"a": 1})
围栏 = '前文```json\n{"b":2}\n```后文'
check("markdown 围栏", 提取JSON对象(围栏) == {"b": 2})
裁剪 = '说明{"c":3}更多'
check("{}裁剪", 提取JSON对象(裁剪) == {"c": 3})
check("非字符串 None", 提取JSON对象(None) is None)
check("空文本 None", 提取JSON对象("  ") is None)

print("== 清洗字符串列表 ==")
check("清洗去重保序", 清洗字符串列表(["像素","像素","","风格"]) == ["像素","风格"],
      f"got={清洗字符串列表(['像素','像素','','风格'])}")
check("非列表→[]", 清洗字符串列表("像素") == [])

print("== 解析打标结果 合法样本 ==")
合法 = '```json\n{"名称":"红衣骑士","类型":"","内容描述":"红披风骑士。","视觉特征":["像素","红披风"],"标签":["骑士"],"适用场景":"战斗","置信度":0.92}\n```'
r = 解析打标结果(合法, 素材)
check("类型兜底用素材信息", r["类型"] == "角色精灵图", f"got={r['类型']!r}")
check("元数据合并", r["素材ID"] == "T0003" and r["来源文件"] == "Actor1.png")
check("置信度保留", r["置信度"] == 0.92)

print("== 解析失败/校验失败严格抛 ValueError ==")
def 期望抛(name, 文本):
    try:
        解析打标结果(文本)
        check(name, False, "未抛异常")
    except ValueError as e:
        check(name, True)

期望抛("非JSON抛错", "这完全不是 JSON 结构")
期望抛("缺必填字段抛错", json.dumps({"名称":"x","类型":"t"}))
期望抛("视觉特征非数组抛错", json.dumps({"名称":"x","类型":"t","内容描述":"c","视觉特征":"字符串","标签":["t"],"置信度":0.5}))
期望抛("置信度非数字抛错", json.dumps({"名称":"x","类型":"t","内容描述":"c","视觉特征":["v"],"标签":["t"],"置信度":"很高"}))

print("== 置信度越界收敛 ==")
r2 = 解析打标结果(json.dumps({"名称":"x","类型":"t","内容描述":"c","视觉特征":["v"],"标签":["t"],"置信度":7.5}), {"类型":"t"})
check("置信度>1 收敛到 1", r2["置信度"] == 1.0, f"got={r2['置信度']}")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)