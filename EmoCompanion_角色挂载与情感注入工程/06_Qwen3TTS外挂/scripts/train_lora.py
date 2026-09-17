# -*- coding: utf-8 -*-
"""
Qwen3-TTS 角色外挂 —— LoRA 训练脚本（Task 3 实装）
=====================================================
- Base 权重只读（bnb 4bit + 冻结），仅训练 LoRA 旁路 → 独立导出外挂包
- 情感条件控制：输入拼 [emotion]<标签>[/emotion] 前缀，ref 音色作声纹条件
- 收敛监控：train/val loss，val loss ≥N 轮不降 → 早停；记录最佳 checkpoint
- 可选 Langfuse 埋点（训练 loss 上报），无环境变量时静默跳过

用法：
  python train_lora.py \
      --base_model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
      --train_jsonl data/split/train.jsonl \
      --val_jsonl   data/split/val.jsonl \
      --output_dir  output/tyy_luoyuan
"""
import argparse
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader

# PEFT / bnb（依赖：peft, bitsandbytes, transformers, accelerate）
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


# ---------------- 情感词表：标签 -> 条件前缀 ----------------
DEFAULT_EMOTION_VOCAB = {
    "开心": "[emotion]开心[/emotion]",
    "俏皮": "[emotion]俏皮[/emotion]",
    "悲伤": "[emotion]悲伤[/emotion]",
    "撒娇": "[emotion]撒娇[/emotion]",
    "轻松": "[emotion]轻松[/emotion]",
    "激动": "[emotion]激动[/emotion]",
    "温柔": "[emotion]温柔[/emotion]",
    "中性": "",
}


def lazy_import_langfuse():
    """可选 import langfuse；缺失时返回 None（不阻塞）。"""
    try:
        from langfuse import Langfuse
        return Langfuse()
    except Exception:
        return None


class TTSTextDataset(Dataset):
    """将 JSONL 行映射为 (text_with_emotion, ref_audio_path, emotion)。"""

    def __init__(self, jsonl_path, emotion_vocab):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get("text", "")
                emotion = rec.get("emotion_primary", "中性")
                prefix = emotion_vocab.get(emotion, "")
                self.items.append({
                    "input_text": f"{prefix}{text}".strip(),
                    "audio_filepath": rec.get("audio_filepath", ""),
                    "emotion": emotion,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def build_lora_model(base_model, lora_rank=16, lora_alpha=32, target_modules=None):
    """加载 Base 为 bnb4bit(只读) + 附加 LoRA。"""
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = base_model  # 外部已用 from_pretrained(quant_config=...) 加载
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def collate_fn(batch, tokenizer, max_length=512):
    texts = [b["input_text"] for b in batch]
    encoded = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    encoded["labels"] = encoded["input_ids"].clone()
    return encoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Base 模型路径/HF id")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", default="", help="留空则只用训练集(不评估)")
    ap.add_argument("--output_dir", default="output/lora")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--early_stop_patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token

    # 2026-08 实测：目标 loss ∈ [4.1, 4.2] 兼具音色与指令；<4.1 防过拟合、>4.5 欠拟合
    target_loss = (4.1, 4.2)

    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quant_config, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = build_lora_model(base, args.lora_rank, args.lora_alpha)
    model.print_trainable_parameters()

    train_ds = TTSTextDataset(args.train_jsonl, DEFAULT_EMOTION_VOCAB)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length))
    val_dl = None
    if args.val_jsonl and os.path.exists(args.val_jsonl):
        val_ds = TTSTextDataset(args.val_jsonl, DEFAULT_EMOTION_VOCAB)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length))
    print(f"train {len(train_ds)} samples" + (f" / val {len(val_ds)}" if val_dl else ""))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 可选 Langfuse 埋点
    lf = lazy_import_langfuse()

    best_val, no_improve = float("inf"), 0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        optimizer.zero_grad()
        for batch in train_dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            total_loss += loss.item() * args.grad_accum
            n += 1
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_train = total_loss / max(n, 1)
        print(f"epoch {epoch}  train_loss={avg_train:.4f}")

        # 验证 + 早停
        if val_dl:
            model.eval()
            vloss, vn = 0.0, 0
            with torch.no_grad():
                for batch in val_dl:
                    batch = {k: v.cuda() for k, v in batch.items()}
                    vloss += model(**batch).loss.item()
                    vn += 1
            avg_val = vloss / max(vn, 1)
            print(f"epoch {epoch}  val_loss={avg_val:.4f}  "
                  f"(target {target_loss[0]}-{target_loss[1]})")
            if avg_val < best_val:
                best_val, no_improve = avg_val, 0
            else:
                no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[early-stop] val_loss 连续 {args.early_stop_patience} 轮未改进")
                break

        # Langfuse 训练埋点（每 epoch 一条）
        if lf is not None:
            try:
                lf.trace(name="qwen3tts.lora.train").update(
                    output={"epoch": epoch, "train_loss": avg_train,
                            "val_loss": avg_val if val_dl else None})
            except Exception:
                pass

    # 导出外挂包（仅 adapter，Base 不动）
    lora_dir = os.path.join(args.output_dir, "lora")
    model.save_pretrained(lora_dir)
    with open(os.path.join(args.output_dir, "emotion_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(DEFAULT_EMOTION_VOCAB, f, ensure_ascii=False, indent=2)
    print(f"LoRA adapter 已导出: {lora_dir}")


if __name__ == "__main__":
    main()