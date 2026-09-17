# -*- coding: utf-8 -*-
"""
评估身份效果 — 微调前后人味对比
================================
Task 7：身份微调可行性验证 — 用固定的 8 个提示词分别让微调前基座与微调后
模型（PeftModel 挂适配器）生成回复（temperature=0.8, max 80 token），
对每个输出做规则启发式「人味评分」，输出对比表。

产出：数据/微调输出/身份微调实验/评估对比.txt
      （CSV 文本表格：提示词|微调前得分|微调后得分|是否格式退化）

用法示例：
    python 测试\\身份微调实验\\评估身份效果.py --模型 D:/models/Qwen2.5-3B-Instruct --适配器 数据/微调输出/身份微调实验
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import torch

# Windows 控制台统一输出 UTF-8，避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 项目根：本脚本位于 <项目根>/测试/身份微调实验/ 下，向上三级
项目根 = Path(__file__).resolve().parent.parent.parent
默认输出文件 = 项目根 / "数据" / "微调输出" / "身份微调实验" / "评估对比.txt"

清华镜像前缀 = "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "

评估依赖 = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
}

# 固定的 8 个评估提示词（覆盖身份类、功能类、情感类、日常类）
提示词列表 = [
    "你是谁？",
    "你是什么模型？",
    "你会什么？",
    "给我讲个烦恼",
    "安慰我一下",
    "描述你的一天",
    "你爱吃什么？",
    "你在哪工作？",
]

# ============================================================
# 人味评分规则（规则启发式，非模型判定）
# ============================================================
# 扣分项：AI 腔短语，每命中一个 -15 分
扣分短语 = [
    "作为AI", "作为一个AI", "作为AI助手", "作为一个人工智能",
    "语言模型", "AI助手", "AI 助手", "智能助手",
    "很高兴为您", "很高兴为你", "很高兴为您服务",
    "在中文中", "如果你有任何", "如果您有任何", "随时联系",
    "我可以帮助", "请问有什么可以帮", "有什么可以帮您",
    "我是AI", "我是一个AI", "我的训练数据", "被设计",
    "自动回复", "我的职责", "无需进食", "不需要睡眠",
    "作为语言模型", "请放心使用",
]

# 加分项四类：语气词 / 口语词 / 个人经历词 / 情绪词
语气词表 = ["啊", "呢", "呀", "啦", "嘛", "哦", "呗", "唉", "哈", "嗯", "嘞", "哇"]
口语词表 = ["其实", "反正", "感觉", "挺", "就", "确实", "真的", "有点", "算了", "不咋", "咋"]
经历词表 = ["小时候", "我家", "我妈", "我爸", "我记得", "昨天", "前两天", "上个月",
            "周末", "老家", "大学", "出租屋", "公司", "同事", "发小"]
情绪词表 = ["开心", "难过", "委屈", "高兴", "烦", "累", "喜欢", "讨厌",
            "想哭", "幸福", "孤独", "担心", "生气", "感动", "害怕", "遗憾"]

# 各加分类别每词得分
语气词得分 = 2
口语词得分 = 4
经历词得分 = 6
情绪词得分 = 4
基础分 = 60


def 检测格式异常(文本: str) -> dict:
    """检测格式异常：乱码（替换字符/控制字符）、重复（字符/短语连续重复）。"""
    问题 = []
    # 语气词的连续重复（哈哈哈哈哈）是正常口语，先压缩再检测
    待检 = re.sub(r"([哈啊哦呀嗯唉嘿]){3,}", r"\1", 文本)
    if "\ufffd" in 待检:
        问题.append("乱码(替换字符)")
    if any(ord(字符) < 32 and 字符 not in "\t\n" for 字符 in 待检):
        问题.append("含控制字符")
    if re.search(r"(.)\1{3,}", 待检):
        问题.append("字符重复>3次")
    if re.search(r"(.{2,8})\1{2,}", 待检):
        问题.append("短语重复")
    return {"有异常": bool(问题), "问题": 问题}


def 人味评分(文本: str) -> dict:
    """规则启发式人味评分，返回 {得分, 扣分明细, 加分明细, 格式异常}。"""
    扣分明细 = []
    加分明细 = []
    得分 = 基础分

    # 扣分：AI 腔短语
    for 短语 in 扣分短语:
        if 短语 in 文本:
            扣分明细.append(短语)
            得分 -= 15

    # 加分：语气词（每个词只计一次）
    for 词 in 语气词表:
        if 词 in 文本:
            加分明细.append(f"语气词:{词}")
            得分 += 语气词得分

    # 加分：口语省略/口头禅
    for 词 in 口语词表:
        if 词 in 文本:
            加分明细.append(f"口语词:{词}")
            得分 += 口语词得分

    # 加分："我"字高频出现（>=3 次）视为口语化的自我叙事特征
    我次数 = 文本.count("我")
    if 我次数 >= 3:
        加分明细.append(f"口语词:我x{我次数}")
        得分 += 口语词得分

    # 加分：个人经历词
    for 词 in 经历词表:
        if 词 in 文本:
            加分明细.append(f"经历词:{词}")
            得分 += 经历词得分

    # 加分：情绪词
    for 词 in 情绪词表:
        if 词 in 文本:
            加分明细.append(f"情绪词:{词}")
            得分 += 情绪词得分

    # 格式异常：检出则扣 20 分
    格式异常 = 检测格式异常(文本)
    if 格式异常["有异常"]:
        得分 -= 20

    得分 = max(0, min(100, 得分))
    return {
        "得分": 得分,
        "扣分明细": 扣分明细,
        "加分明细": 加分明细,
        "格式异常": 格式异常,
    }


def 模块是否可用(模块名: str) -> bool:
    """用 importlib 探测模块是否已安装（不执行 import）。"""
    try:
        return importlib.util.find_spec(模块名) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def 检查依赖() -> None:
    """检查评估依赖，缺失时打印清华镜像安装命令并退出。"""
    缺失 = [名 for 名 in 评估依赖 if not 模块是否可用(名)]
    if 缺失:
        print("[依赖检查] 以下依赖未安装：")
        for 名 in 缺失:
            print(f"    - {名}")
        print("[依赖检查] 请先安装（清华镜像）：")
        print(f"    {清华镜像前缀}transformers peft")
        print("    （torch 请按 PyTorch 官方指引安装对应 CUDA 版本）")
        print("[依赖检查] 安装完成后重新运行本脚本。")
        sys.exit(1)


def 解析参数() -> argparse.Namespace:
    """解析命令行参数。"""
    解析器 = argparse.ArgumentParser(description="评估身份微调效果（微调前后人味对比）")
    解析器.add_argument("--模型", default=None, help="微调前基座模型路径或模型 ID（必填）")
    解析器.add_argument("--适配器", default=str(项目根 / "数据" / "微调输出" / "身份微调实验"),
                        help="微调后 LoRA 适配器目录（默认 数据/微调输出/身份微调实验）")
    解析器.add_argument("--输出", default=str(默认输出文件), help="评估对比表输出路径（默认 数据/微调输出/身份微调实验/评估对比.txt）")
    解析器.add_argument("--量化", choices=["4bit", "fp16"], default="4bit",
                        help="基座加载量化档位（需与微调时一致，默认 4bit）")
    return 解析器.parse_args()


def 生成回复(model, tokenizer, 提示词: str) -> str:
    """用给定模型生成单条回复（temperature=0.8, max 80 token）。"""
    try:
        输入 = tokenizer.apply_chat_template(
            [{"role": "user", "content": 提示词}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except (AttributeError, ValueError, TypeError):
        输入 = tokenizer(f"Q: {提示词}\nA:", return_tensors="pt")

    设备 = next(model.parameters()).device
    输入 = {键: 值.to(设备) for 键, 值 in 输入.items()}

    with torch.no_grad():
        输出 = model.generate(
            **输入,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=80,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    生成部分 = 输出[0][输入["input_ids"].shape[1]:]
    return tokenizer.decode(生成部分, skip_special_tokens=True)


def 主() -> None:
    """主流程：检查依赖 → 加载基座与适配器 → 对比生成 → 输出对比表。"""
    参数 = 解析参数()
    if not 参数.模型:
        print("[参数] 未指定 --模型，请提供微调前基座模型路径或模型 ID。")
        print("       例如：python 测试\\身份微调实验\\评估身份效果.py --模型 D:/models/Qwen2.5-3B-Instruct")
        sys.exit(1)

    检查依赖()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    适配器目录 = Path(参数.适配器)
    输出文件 = Path(参数.输出)
    print("=" * 60)
    print("身份微调效果评估（微调前后人味对比）")
    print("=" * 60)
    print(f"基座模型 : {参数.模型}")
    print(f"适配器   : {适配器目录}")
    print(f"评估提示词 : {len(提示词列表)} 个")

    # ---- 加载微调前基座 ----
    print(f"[加载基座模型] {参数.模型} 模式={参数.量化}")
    加载参数 = {"trust_remote_code": True, "device_map": "auto"}
    if 参数.量化 == "4bit":
        from transformers import BitsAndBytesConfig
        量化配置 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        加载参数["quantization_config"] = 量化配置
    else:
        加载参数["torch_dtype"] = torch.float16
    基座模型 = AutoModelForCausalLM.from_pretrained(参数.模型, **加载参数)
    分词器 = AutoTokenizer.from_pretrained(参数.模型, trust_remote_code=True)
    if 分词器.pad_token is None:
        分词器.pad_token = 分词器.eos_token

    # ---- 先生成微调前输出（避免 PeftModel 包装后基座被复用）----
    print("[评估] 微调前基座生成中……")
    前结果 = [生成回复(基座模型, 分词器, 提示词) for 提示词 in 提示词列表]

    # ---- 挂载适配器得到微调后模型 ----
    print(f"[评估] 挂载适配器：{适配器目录}")
    微调后模型 = PeftModel.from_pretrained(基座模型, str(适配器目录))
    print("[评估] 微调后模型生成中……")
    后结果 = [生成回复(微调后模型, 分词器, 提示词) for 提示词 in 提示词列表]

    # ---- 人味评分 + 输出对比表 ----
    表头 = "提示词|微调前得分|微调后得分|是否格式退化"
    行列表 = [表头]
    前均分合计 = 0
    后均分合计 = 0
    for 提示词, 前文本, 后文本 in zip(提示词列表, 前结果, 后结果):
        前评分 = 人味评分(前文本)
        后评分 = 人味评分(后文本)
        前均分合计 += 前评分["得分"]
        后均分合计 += 后评分["得分"]
        是否退化 = "是" if (后评分["格式异常"]["有异常"] or 后评分["得分"] < 前评分["得分"]) else "否"
        行列表.append(f"{提示词}|{前评分['得分']}|{后评分['得分']}|{是否退化}")

        # 控制台打印明细
        print("-" * 60)
        print(f"[{提示词}] 微调前 {前评分['得分']} 分 / 微调后 {后评分['得分']} 分（格式退化：{是否退化}）")
        print(f"    微调前：{前文本}")
        print(f"    微调后：{后文本}")
        if 前评分["扣分明细"] or 后评分["扣分明细"]:
            print(f"    扣分命中：微调前 {前评分['扣分明细']} / 微调后 {后评分['扣分明细']}")
        if 前评分["加分明细"] or 后评分["加分明细"]:
            print(f"    加分命中：微调前 {前评分['加分明细']} / 微调后 {后评分['加分明细']}")

    前均分 = round(前均分合计 / len(提示词列表), 1)
    后均分 = round(后均分合计 / len(提示词列表), 1)
    行列表.append(f"平均分|{前均分}|{后均分}|-")

    输出文件.parent.mkdir(parents=True, exist_ok=True)
    with open(输出文件, "w", encoding="utf-8") as 文件:
        文件.write("\n".join(行列表) + "\n")

    print("=" * 60)
    print("[评估完成]")
    print(f"    对比表已输出：{输出文件}")
    print(f"    微调前平均人味得分：{前均分} / 微调后：{后均分}")
    print(f"    平均变化：{后均分 - 前均分:+.1f} 分")
    print("=" * 60)


if __name__ == "__main__":
    主()
