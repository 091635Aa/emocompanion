# -*- coding: utf-8 -*-
"""
冒烟测试脚本（Task 12.2）
========================
最小链路：文本输入 → 微调数据包 → 微调 → 推理。

步骤：
  a. 用 数据预处理.上传并清洗 创建示例文本文件（"今天很开心，和朋友去公园玩"）
  b. 用 打标引擎.自动打标 打标 → 导出数据包 生成 情感微调_*.jsonl
  c. 用 微调引擎.微调 跑（transformers/peft/datasets 缺失时记录"依赖缺失跳过"）
  d. 用 推理架构.V1推理引擎 生成（transformers 缺失时记录"依赖缺失跳过"）

输出：
  控制台逐步骤 PASS/FAIL/SKIP + 汇总；测试报告写入 测试\冒烟测试报告.txt。
退出码：0（全部通过或跳过）、1（有失败）。
"""

import os
import sys

# 确保项目根在模块搜索路径中（从任意目录运行均可用）
脚本目录 = os.path.dirname(os.path.abspath(__file__))
项目根 = os.path.dirname(脚本目录)
if 项目根 not in sys.path:
    sys.path.insert(0, 项目根)

# Windows 控制台统一输出 UTF-8，避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 测试报告路径
测试目录 = 脚本目录
报告路径 = os.path.join(测试目录, "冒烟测试报告.txt")

# 依赖缺失时的统一安装命令（清华镜像）
训练安装命令 = (
    "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
    "transformers peft datasets accelerate bitsandbytes"
)
推理安装命令 = (
    "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers"
)

步骤记录 = []   # {"步骤": str, "状态": "PASS"/"FAIL"/"SKIP", "原因": str}
待安装命令 = set()


def 记录(步骤: str, 状态: str, 原因: str = "") -> None:
    """记录一步测试结果并打印控制台。"""
    步骤记录.append({"步骤": 步骤, "状态": 状态, "原因": 原因})
    行 = f"[{状态}] {步骤}"
    if 原因:
        行 += f"：{原因}"
    print(行)


def 检查依赖(模块包列表) -> list:
    """检查模块是否可导入，返回缺失列表 [(模块名, pip包名), ...]。"""
    缺失 = []
    for 模块名, 包名 in 模块包列表:
        try:
            __import__(模块名)
        except ImportError:
            缺失.append((模块名, 包名))
    return 缺失


def 读文本文件(路径: str) -> str:
    """按多种编码尝试读取文本文件。"""
    for 编码 in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with open(路径, "r", encoding=编码) as 文件:
                return 文件.read()
        except (UnicodeDecodeError, OSError):
            continue
    with open(路径, "r", encoding="utf-8", errors="replace") as 文件:
        return 文件.read()


# ==================================================================
# 步骤 a：文本输入 → 上传并清洗
# ==================================================================

def 步骤_文本输入() -> str:
    """创建示例文本文件并上传清洗，返回清洗后文本（空串表示失败）。"""
    from 核心引擎 import 数据预处理

    临时路径 = os.path.join(测试目录, "冒烟_示例文本.txt")
    with open(临时路径, "w", encoding="utf-8") as 文件:
        文件.write("今天很开心，和朋友去公园玩。")
    结果 = 数据预处理.上传并清洗(临时路径, "文本")
    if not 结果.get("成功"):
        记录("a.文本输入→上传并清洗", "FAIL", str(结果.get("错误", "上传失败")))
        return ""
    上传路径 = 结果.get("文件路径") or 结果.get("路径") or 临时路径
    文本 = 读文本文件(上传路径).strip()
    if not 文本:
        记录("a.文本输入→上传并清洗", "FAIL", "清洗后文本为空")
        return ""
    记录("a.文本输入→上传并清洗", "PASS", f"任务ID={结果.get('任务ID')}，文本长度={len(文本)}")
    return 文本


# ==================================================================
# 步骤 b：自动打标 → 导出数据包
# ==================================================================

def 步骤_打标导出(文本: str) -> bool:
    """打标文本并导出微调数据包，返回是否成功。"""
    from 核心引擎 import 数据预处理, 打标引擎

    片段列表 = 数据预处理.话题分割(文本, [])
    if not 片段列表:
        # 单句可能只分一段，构造兜底片段保证链路可测
        片段列表 = [{
            "片段ID": "seg_0001", "话题ID": "seg_0001",
            "话题摘要": "示例文本", "起始秒": 0.0, "结束秒": 1.0, "文本": 文本,
        }]
    打标后 = 打标引擎.自动打标(片段列表)
    if not 打标后:
        记录("b.自动打标", "FAIL", "打标结果为空")
        return False
    情感维度 = 打标后[0].get("情感维度", "")
    记录("b.自动打标", "PASS", f"片段数={len(打标后)}，情感维度={情感维度}")

    导出 = 打标引擎.导出数据包(打标后, "jsonl", "情感")
    if not 导出.get("成功"):
        记录("b.导出数据包", "FAIL", str(导出.get("错误", "导出失败")))
        return False
    if not os.path.exists(导出.get("路径", "")):
        记录("b.导出数据包", "FAIL", f"数据包文件不存在：{导出.get('路径')}")
        return False
    记录("b.导出数据包", "PASS", f"条数={导出.get('条数')}，路径={导出.get('路径')}")
    return True


# ==================================================================
# 步骤 c：微调（依赖缺失 → SKIP）
# ==================================================================

def 步骤_微调() -> None:
    """跑微调引擎；transformers/peft/datasets 缺失时记录依赖缺失跳过。"""
    缺失 = 检查依赖([("transformers", "transformers"), ("peft", "peft"), ("datasets", "datasets")])
    if 缺失:
        待安装命令.add(训练安装命令)
        缺失名 = "、".join(f"{模块}（{包}）" for 模块, 包 in 缺失)
        记录("c.微调", "SKIP", f"依赖缺失跳过：{缺失名}")
        return
    from 核心引擎 import 微调引擎

    结果 = 微调引擎.微调({"模型路径": "本地测试模型", "启用情感微调": True})
    if 结果.get("成功"):
        记录("c.微调", "PASS", f"输出目录={结果.get('输出目录')}")
    else:
        记录("c.微调", "FAIL", str(结果.get("错误", "微调失败")))


# ==================================================================
# 步骤 d：V1 推理（依赖缺失 → SKIP）
# ==================================================================

def 步骤_推理() -> None:
    """用 V1推理引擎 生成；transformers 缺失时记录依赖缺失跳过。"""
    缺失 = 检查依赖([("transformers", "transformers")])
    if 缺失:
        待安装命令.add(推理安装命令)
        记录("d.推理（V1推理引擎）", "SKIP", "依赖缺失跳过：transformers")
        return
    from 核心引擎.推理架构 import V1推理引擎

    引擎 = V1推理引擎()
    try:
        初始化结果 = 引擎.初始化("本地测试模型")
        if not isinstance(初始化结果, dict) or not 初始化结果.get("成功"):
            错误 = 初始化结果.get("错误") if isinstance(初始化结果, dict) else str(初始化结果)
            记录("d.推理（V1推理引擎）", "FAIL", f"初始化失败：{错误}")
            return
        生成结果 = 引擎.生成("你好，介绍一下你自己")
        回复 = 生成结果.get("回复") if isinstance(生成结果, dict) else str(生成结果 or "")
        if not 回复:
            记录("d.推理（V1推理引擎）", "FAIL", "生成了空回复")
            return
        记录("d.推理（V1推理引擎）", "PASS", f"回复长度={len(回复)}，回复={回复[:30]}…")
    except Exception as 错误:
        记录("d.推理（V1推理引擎）", "FAIL", f"推理异常：{错误}")
    finally:
        try:
            引擎.释放()
        except Exception:
            pass


# ==================================================================
# 主流程
# ==================================================================

def 写报告() -> None:
    """把测试结果写入 测试\冒烟测试报告.txt。"""
    PASS数 = sum(1 for 条 in 步骤记录 if 条["状态"] == "PASS")
    FAIL数 = sum(1 for 条 in 步骤记录 if 条["状态"] == "FAIL")
    SKIP数 = sum(1 for 条 in 步骤记录 if 条["状态"] == "SKIP")
    有失败 = FAIL数 > 0

    行 = []
    行.append("=" * 60)
    行.append("一体化全流程AI应用 · 冒烟测试报告")
    行.append(f"时间：{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")
    行.append("目标：文本输入 → 微调数据包 → 微调 → 推理（最小链路）")
    行.append("=" * 60)
    行.append("")
    for 条 in 步骤记录:
        行.append(f"[{条['状态']}] {条['步骤']}" + (f"：{条['原因']}" if 条["原因"] else ""))
    行.append("")
    行.append("-" * 60)
    行.append(f"汇总：PASS {PASS数} / FAIL {FAIL数} / SKIP {SKIP数}")
    行.append(f"总体：{'通过' if not 有失败 else '未通过（存在失败步骤）'}")
    if 待安装命令:
        行.append("")
        行.append("待安装命令（依赖缺失，安装后可重新运行）：")
        for 命令 in sorted(待安装命令):
            行.append(f"  {命令}")
    行.append("")
    行.append(f"退出码：{1 if 有失败 else 0}")

    内容 = "\n".join(行)
    with open(报告路径, "w", encoding="utf-8") as 文件:
        文件.write(内容 + "\n")
    print("")
    print(内容)
    print(f"[冒烟测试] 报告已写入：{报告路径}")


def 主入口() -> int:
    """主流程：逐步骤执行并输出 PASS/FAIL/SKIP，返回退出码。"""
    print("=" * 60)
    print("一体化全流程AI应用 · 冒烟测试")
    print("=" * 60)

    文本 = 步骤_文本输入()
    if 文本:
        步骤_打标导出(文本)
    else:
        # 输入失败时，后续链路步骤记录为依赖前置失败（FAIL）
        记录("b.自动打标与导出数据包", "FAIL", "依赖前置失败：文本输入失败")
    步骤_微调()
    步骤_推理()

    写报告()
    PASS数 = sum(1 for 条 in 步骤记录 if 条["状态"] == "PASS")
    FAIL数 = sum(1 for 条 in 步骤记录 if 条["状态"] == "FAIL")
    print(f"[冒烟测试] 汇总：PASS {PASS数} / FAIL {FAIL数} / SKIP "
          f"{sum(1 for 条 in 步骤记录 if 条['状态'] == 'SKIP')}")
    return 1 if FAIL数 > 0 else 0


if __name__ == "__main__":
    sys.exit(主入口())
