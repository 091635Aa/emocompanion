# coding=utf-8
"""
缘圆 双路 LoRA 微调（音色路 / 情感路）
====================================================================
基于官方 Qwen3-TTS sft_12hz.py 改造，将「全参 SFT」改为「LoRA PEFT」，
满足「外挂优先、不污染 Base」铁律：
  - Base（含 talker/sub_talker/code_predictor）以 bnb4bit 只读加载
  - 仅训练 talker 的 LoRA 旁路（q/k/v/o + gate/up/down）
  - 导出仅 adapter_model.safetensors + adapter_config.json（外挂包）
  - 训练期间 speaker embedding 由 ref_mels 编码，作为说话人条件

两条路径共用本脚本，靠参数区分：
  1) voice:   --mode voice    （audio + 原文 + 固定中性 ref）
  2) emotion: --mode emotion （audio + [emotion]<标签>[/emotion]前缀 + 情感ref）
每条路独立导出 adapter → 推理时双 adapter 叠加。

注意：依赖官方 finetuning 的 dataset.py（本目录已放置）。
用法：
  python sft_lora.py --mode voice --init_model_path <Base> \
      --train_jsonl data/voice_train_codes.jsonl --output_model_path out/voice_lora ...
"""
import argparse
import json
import os
import shutil

# 强制离线：避免 from_pretrained 在 CPU 端访问 HuggingFace hub 造成阻塞
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 缓解 CUDA 显存碎片化（长序列 + 无 flash-attn 时激活量大）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 顶层只导入经过验证的最小集合（与 load_verify.py 一致，8s 成功加载到 GPU）。
# accelerate / peft / dataset 推迟到模型加载完成后惰性导入，避免与 qwen_tts 的 import 相互干扰。
import torch
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from transformers import AutoConfig

target_speaker_embedding = None


def train():
    global target_speaker_embedding
    print("[train] enter", flush=True)
    from accelerate import Accelerator
    from dataset import TTSDataset
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from peft import LoraConfig, get_peft_model
    print("[train] imports done", flush=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["voice", "emotion"], required=True)
    ap.add_argument("--init_model_path", required=True)
    ap.add_argument("--output_model_path", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_codes", type=int, default=200,
                    help="截断 audio_codes 帧数；直播长文本数千帧会撑爆 16GB 显存，voice LoRA 固定音色用前缀即可")
    ap.add_argument("--max_text", type=int, default=60,
                    help="截断文本字数（与 codec 帧对齐，控制序列长度）")
    ap.add_argument("--speaker_name", type=str, default="tyy_luoyuan")
    args = ap.parse_args()

    run_dir = os.path.join(args.output_model_path, "logs", args.mode)
    os.makedirs(run_dir, exist_ok=True)

    # 先加载模型（此时尚未 import/实例化 accelerate、peft，与 load_verify 已验证的成功路径一致）
    print("[train] loading model...", flush=True)
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
        trust_remote_code=True,
    )
    print("[train] model loaded", flush=True)
    config = AutoConfig.from_pretrained(args.init_model_path,
                                         local_files_only=True,
                                         trust_remote_code=True)

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision="bf16", log_with="tensorboard",
                              project_dir=run_dir)
    accelerator.init_trackers(args.mode)

    # 将 talker 主干包成 LoRA 可训练版本（基座只读，Bias 冻结，仅训练旁路）
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(qwen3tts.model, lora_config)
    model.print_trainable_parameters()

    train_data = []
    for l in open(args.train_jsonl, encoding="utf-8-sig").readlines():
        d = json.loads(l)
        codes = d.get("audio_codes")
        if isinstance(codes, list) and args.max_codes > 0 and len(codes) > args.max_codes:
            d = dict(d)
            d["audio_codes"] = codes[:args.max_codes]
        t = d.get("text", "")
        if isinstance(t, str) and args.max_text > 0 and len(t) > args.max_text:
            d = dict(d)
            d["text"] = t[:args.max_text]
        train_data.append(d)
    print(f"[train] samples={len(train_data)} max_codes={args.max_codes} max_text={args.max_text}", flush=True)
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    collate_fn=dataset.collate_fn)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model.train()

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dl):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]
                input_text_embedding = model.talker.model.text_embedding(
                    input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(
                    input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding
                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    emb = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]) * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + emb

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[0][-1]
                talker_hidden = hidden[codec_mask[:, :-1]]
                talker_codes = codec_ids[codec_mask]
                _, sub_loss = model.talker.forward_sub_talker_finetune(
                    talker_codes, talker_hidden)
                loss = outputs.loss + 0.3 * sub_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    # 导出 LoRA adapter（外挂包）：不复制 Base，只存 adapter 权重+配置
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        out = os.path.join(args.output_model_path,
                           f"{args.mode}_checkpoint-epoch-{args.num_epochs-1}")
        os.makedirs(out, exist_ok=True)
        unwrapped.save_pretrained(out)   # 仅 adapter_model.safetensors + adapter_config.json
        # 备份首样本 speaker embedding 作为角色音色参考（启发式）
        if target_speaker_embedding is not None:
            torch.save(target_speaker_embedding.cpu(),
                       os.path.join(out, "target_speaker_embedding.pt"))
        print(f"[ok] LoRA adapter 导出(外挂包): {out}")
        print("注意：Base 权重未被修改（bnb4bit 只读），本目录仅含 adapter。")


if __name__ == "__main__":
    train()