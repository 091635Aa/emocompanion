# -*- coding: utf-8 -*-
"""
身份微调实验 — LoRA 微调脚本
============================
Task 7：身份微调可行性验证 — 在基座模型上以 LoRA/QLoRA 微调"真人身份"小样本数据，
验证"告诉模型它是真人"能否去除 AI 味。

流程：依赖检查 → 加载基座（默认 4bit QLoRA）→ 自动检测 target_modules →
      apply_chat_template（无模板回退 "Q: {instruction}\\nA: {response}"）→ 训练 → 保存适配器。

用法示例：
    python 测试\\身份微调实验\\身份微调实验.py --模型 D:/models/Qwen2.5-3B-Instruct
    python 测试\\身份微调实验\\身份微调实验.py --模型 D:/models/Qwen2.5-3B-Instruct --量化 fp16 --轮数 3

参数：
    --模型  基座模型路径或模型 ID（必填，提示使用本地小模型如 Qwen2.5-3B）
    --数据  训练数据 JSONL（默认 数据/微调数据包/身份微调_小样本.jsonl）
    --输出  产物输出目录（默认 数据/微调输出/身份微调实验/）
    --轮数  训练轮数（默认 2）
    --量化  量化档位：4bit（默认，QLoRA 省显存）/ fp16（LoRA）
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# Windows 控制台统一输出 UTF-8，避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 项目根：本脚本位于 <项目根>/测试/身份微调实验/ 下，向上三级
项目根 = Path(__file__).resolve().parent.parent.parent
默认数据文件 = 项目根 / "数据" / "微调数据包" / "身份微调_小样本.jsonl"
默认输出目录 = 项目根 / "数据" / "微调输出" / "身份微调实验"

清华镜像前缀 = "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "

# 训练所需第三方依赖：模块名 -> 安装名
训练依赖 = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "datasets": "datasets",
    "accelerate": "accelerate",
}


def 模块是否可用(模块名: str) -> bool:
    """用 importlib 探测模块是否已安装（不执行 import，避免副作用）。"""
    try:
        return importlib.util.find_spec(模块名) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def 检查依赖(额外依赖: dict = None) -> None:
    """检查训练依赖，缺失时打印清华镜像安装命令并退出（不静默安装）。"""
    所需 = dict(训练依赖)
    if 额外依赖:
        所需.update(额外依赖)
    缺失 = [名 for 名 in 所需 if not 模块是否可用(名)]
    if 缺失:
        print("[依赖检查] 以下依赖未安装：")
        for 名 in 缺失:
            print(f"    - {名}")
        print("[依赖检查] 请先安装（清华镜像）：")
        print(f"    {清华镜像前缀}transformers peft datasets")
        if "torch" in 缺失:
            print("    （torch 请按 PyTorch 官方指引安装对应 CUDA 版本）")
        if "bitsandbytes" in 缺失:
            print("    （bitsandbytes 为 4bit 量化所需，缺失时可改用 --量化 fp16）")
        print("[依赖检查] 安装完成后重新运行本脚本。")
        sys.exit(1)


def 解析参数() -> argparse.Namespace:
    """解析命令行参数。"""
    解析器 = argparse.ArgumentParser(description="身份微调实验（LoRA/QLoRA）")
    解析器.add_argument("--模型", default=None, help="基座模型路径或模型 ID（建议本地小模型如 Qwen2.5-3B-Instruct）")
    解析器.add_argument("--数据", default=str(默认数据文件), help="训练数据 JSONL 路径（默认 数据/微调数据包/身份微调_小样本.jsonl）")
    解析器.add_argument("--输出", default=str(默认输出目录), help="产物输出目录（默认 数据/微调输出/身份微调实验/）")
    解析器.add_argument("--轮数", type=int, default=2, help="训练轮数（默认 2）")
    解析器.add_argument("--量化", choices=["4bit", "fp16"], default="4bit",
                        help="量化档位：4bit=QLoRA（默认，省显存）；fp16=LoRA")
    return 解析器.parse_args()


def 读取数据(数据路径: Path) -> list:
    """读取 JSONL 训练数据（每行 {"instruction","response"}），返回原始字典列表。"""
    原始 = []
    with open(数据路径, encoding="utf-8") as 文件:
        for 行号, 行 in enumerate(文件, start=1):
            行 = 行.strip()
            if not 行:
                continue
            try:
                条目 = json.loads(行)
            except json.JSONDecodeError as 错误:
                raise ValueError(f"数据文件第{行号}行不是合法 JSON：{错误}") from 错误
            if not isinstance(条目, dict) or "instruction" not in 条目 or "response" not in 条目:
                raise ValueError(f"数据文件第{行号}行缺少 instruction/response 字段")
            原始.append(条目)
    if not 原始:
        raise ValueError(f"数据文件为空：{数据路径}")
    return 原始


def 构建数据集(tokenizer, 原始数据: list):
    """把 instruction/response 转为模型文本（优先 chat 模板，无模板回退 Q/A 拼接）。"""
    from datasets import Dataset

    def 格式化(条目: dict) -> dict:
        try:
            文本 = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": 条目["instruction"]},
                    {"role": "assistant", "content": 条目["response"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        except (AttributeError, ValueError, TypeError):
            # 无 chat 模板时使用 Q/A 拼接格式
            文本 = f"Q: {条目['instruction']}\nA: {条目['response']}"
        return {"text": 文本}

    数据集 = Dataset.from_list([格式化(条目) for 条目 in 原始数据])

    def 分词(样本: dict) -> dict:
        输出 = tokenizer(样本["text"], truncation=True, max_length=256, padding=False)
        输出["labels"] = 输出["input_ids"].copy()
        return 输出

    return 数据集.map(分词, remove_columns=["text"])


def 自动检测目标模块(model, 配置) -> list:
    """按架构自动选择 LoRA target_modules。

    优先尝试常见名称列表（Llama/Qwen 系、ChatGLM 系），
    找不到时回退为模型中名称含 proj/dense 的模块，再兜底常用四件套。
    """
    架构名 = "".join(getattr(配置, "architectures", None) or [])
    模块名集合 = {名字 for 名字, _ in model.named_modules()}

    候选组 = [
        ["q_proj", "k_proj", "v_proj", "o_proj"],  # Llama/Qwen 基础四件套
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 全量 MLP
        ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # ChatGLM 系
    ]
    for 候选 in 候选组:
        命中 = [名字 for 名字 in 候选 if 名字 in 模块名集合]
        if len(命中) >= 3:
            print(f"[架构识别] {架构名 or '未知'} → target_modules: {命中}")
            return 命中

    # 回退：收集名称含 proj/dense 且非 embedding/lm_head/norm 的模块
    回退 = [
        名字 for 名字 in 模块名集合
        if ("proj" in 名字 or "dense" in 名字)
        and not any(排除 in 名字 for 排除 in ("embed", "lm_head", "norm", "rotary", "head"))
    ][:8]
    if not 回退:
        回退 = ["q_proj", "k_proj", "v_proj", "o_proj"]
    print(f"[架构识别] 未匹配已知架构，回退 target_modules: {回退}")
    return 回退


def 主() -> None:
    """主流程：检查依赖 → 加载基座 → 训练 → 保存适配器。"""
    参数 = 解析参数()

    if not 参数.模型:
        print("[参数] 未指定 --模型。")
        print("       请提供本地小模型路径或模型 ID，例如：")
        print("       python 测试\\身份微调实验\\身份微调实验.py --模型 D:/models/Qwen2.5-3B-Instruct")
        print("       （建议先在 RTX 3080 Laptop 16GB 上用 Qwen2.5-3B 级别的小模型验证可行性）")
        sys.exit(1)

    数据路径 = Path(参数.数据)
    输出目录 = Path(参数.输出)
    print("=" * 60)
    print("身份微调实验（LoRA）")
    print("=" * 60)
    print(f"基座模型 : {参数.模型}")
    print(f"数据文件 : {数据路径}")
    print(f"输出目录 : {输出目录}")
    print(f"训练轮数 : {参数.轮数}")
    print(f"量化档位 : {参数.量化}")

    # 4bit 需要 bitsandbytes，缺失时在依赖检查阶段给出针对性提示
    额外依赖 = {"bitsandbytes": "bitsandbytes"} if 参数.量化 == "4bit" else None
    检查依赖(额外依赖)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    # ---- 加载基座模型 ----
    print(f"[加载基座模型] {参数.模型} 模式={参数.量化}")
    加载参数 = {"trust_remote_code": True, "device_map": "auto"}
    if 参数.量化 == "4bit":
        from transformers import BitsAndBytesConfig
        量化配置 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.float16,
        )
        加载参数["quantization_config"] = 量化配置
    else:
        加载参数["torch_dtype"] = torch.float16
    模型 = AutoModelForCausalLM.from_pretrained(参数.模型, **加载参数)
    分词器 = AutoTokenizer.from_pretrained(参数.模型, trust_remote_code=True)
    if 分词器.pad_token is None:
        # 统一用 eos 作为 pad，避免 DataCollator 填充报错
        分词器.pad_token = 分词器.eos_token

    # ---- LoRA 配置（r=16, lora_alpha=32，target_modules 自动检测）----
    目标模块 = 自动检测目标模块(模型, 模型.config)
    LoRA配置 = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=目标模块,
    )
    模型 = get_peft_model(模型, LoRA配置)
    模型.print_trainable_parameters()

    # ---- 数据 ----
    原始数据 = 读取数据(数据路径)
    数据集 = 构建数据集(分词器, 原始数据)
    print(f"[训练数据] 样本数：{len(数据集)}（{数据路径.name}）")

    # ---- 训练 ----
    训练参数 = TrainingArguments(
        output_dir=str(输出目录 / "checkpoints"),
        num_train_epochs=参数.轮数,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        fp16=True,
        dataloader_pin_memory=False,
    )
    训练器 = Trainer(
        model=模型,
        args=训练参数,
        train_dataset=数据集,
        data_collator=DataCollatorForSeq2Seq(分词器, padding=True),
    )
    print("[开始训练]")
    训练结果 = 训练器.train()

    # ---- 保存适配器 ----
    输出目录.mkdir(parents=True, exist_ok=True)
    模型.save_pretrained(str(输出目录))
    分词器.save_pretrained(str(输出目录))
    模型文件 = 输出目录 / "adapter_config.json"

    训练损失 = 训练结果.metrics.get("train_loss", "N/A")
    print("-" * 60)
    print("[训练完成]")
    print(f"    训练损失 : {训练损失}")
    print(f"    样本数   : {len(数据集)}")
    print(f"    产物路径 : {输出目录}")
    print(f"    适配器   : {模型文件}（存在：{模型文件.exists()}）")
    print("=" * 60)
    print("下一步：运行 评估身份效果.py 对比微调前后的人味得分。")
    print(f"    python 测试\\身份微调实验\\评估身份效果.py --模型 {参数.模型} --适配器 {输出目录}")


if __name__ == "__main__":
    主()
