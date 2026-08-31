# coding=utf-8
"""
sft_style_lora —— 说话风格（StylePlug）权重级 LoRA 训练（第三外挂 adapter）
=====================================================================
与 voice/emotion 双路 LoRA 同一套训练框架（sft_lora.py），区别在数据与容量：
  - 数据：合并 voice+emotion 全量 raw 自然语料（去情感标签），学习「缘圆整体说话
    风格」（直播语气、节奏、口头禅色彩），而不是单一情感。
  - 容量：lora_r/alpha 更高(r=24/alpha=48)，保证风格印记更充分（~95% 可复现、
    不随采样噪声漂移——这是采样层 StylePlug 做不到的稳定性）。
  - 数据里仍保留固定中性 speaker ref(503)，音色由 speaker embedding + LoRA 共同锚定。
训练完 adapter 用 scripts/hf_lora_to_tts_gguf.py 转 GGUF，放到
D:\\AI情感\\pykits\\models\\style_lora_qwen3tts.gguf
tts_gguf.GGUFTTS 会自动把它作为第 3 个 adapter 叠加（--lora voice,emotion,style）。

用法（先确认 GPU 空闲、停止占用显存的服务）：
  python sft_style_lora.py --init_model_path <Qwen3-TTS-Base> \
      --train_jsonl ../data/style_train_codes.jsonl --output_model_path ../out/style_lora
"""
import argparse
import json
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402


def build_style_train_jsonl(out_path, max_rows=None):
    """合并 voice+emotion raw/sub_codes 自然语料 -> style_train_codes.jsonl（去情感标签）。

    情感 raw 的 text 带 [emotion] 前缀，此处剥离前缀，只保留自然正文，让风格 LoRA
    学到的是"缘圆整体怎么开口"，而不是某种情感。
    """
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    cands = ["voice_train_sub_codes.jsonl", "emotion_train_sub_codes.jsonl"]
    rows, seen = [], set()
    for fn in cands:
        p = os.path.join(base, fn)
        if not os.path.isfile(p):
            print(f"[skip] 缺 {fn}", flush=True)
            continue
        for line in open(p, encoding="utf-8-sig"):
            d = json.loads(line)
            audio = d.get("audio", "")
            if audio in seen:
                continue
            seen.add(audio)
            txt = re.sub(r"\[emotion\].*?\[/emotion\]\s*", "", d.get("text", "")).strip()
            if not txt:
                continue
            rows.append({"audio": audio, "text": txt,
                         "ref_audio": d.get("ref_audio", ""),
                         "audio_codes": d.get("audio_codes", [])})
            if max_rows and len(rows) >= max_rows:
                break
        if max_rows and len(rows) >= max_rows:
            break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] style 语料 {len(rows)} 条 -> {out_path}", flush=True)
    return out_path


def train_style():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_model_path", required=True)
    ap.add_argument("--output_model_path", required=True)
    ap.add_argument("--train_jsonl", default="")
    ap.add_argument("--lora_r", type=int, default=24)
    ap.add_argument("--lora_alpha", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num_epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_codes", type=int, default=150)
    ap.add_argument("--max_text", type=int, default=60)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    train_jsonl = args.train_jsonl or build_style_train_jsonl(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                     "style_train_codes.jsonl"),
        max_rows=args.max_rows or None)

    from dataset import TTSDataset
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.init_model_path, local_files_only=True,
                                        trust_remote_code=True)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(qwen3tts.model, lora_config)
    model.print_trainable_parameters()

    train_data = []
    for line in open(train_jsonl, encoding="utf-8-sig"):
        d = json.loads(line)
        codes = d.get("audio_codes")
        if isinstance(codes, list) and len(codes) > args.max_codes:
            d = dict(d); d["audio_codes"] = codes[:args.max_codes]
        t = d.get("text", "")
        if isinstance(t, str) and len(t) > args.max_text:
            d = dict(d); d["text"] = t[:args.max_text]
        train_data.append(d)
    print(f"[train] style samples={len(train_data)}", flush=True)

    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    collate_fn=dataset.collate_fn)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    target_emb = None
    for epoch in range(args.num_epochs):
        model.train()
        tot, n = 0.0, 0
        for batch in dl:
            input_ids = batch['input_ids']; codec_ids = batch['codec_ids']
            ref_mels = batch['ref_mels']
            tmask = batch['text_embedding_mask']; cmask0 = batch['codec_embedding_mask']
            amask = batch['attention_mask']
            lab = batch['codec_0_labels']; cmask = batch['codec_mask']
            emb = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
            if target_emb is None:
                target_emb = emb
            it_ids = input_ids[:, :, 0]; ic_ids = input_ids[:, :, 1]
            te = model.talker.model.text_embedding(it_ids) * tmask
            ce = model.talker.model.codec_embedding(ic_ids) * cmask0
            ce[:, 6, :] = emb
            inp = te + ce
            for i in range(1, 16):
                inp = inp + model.talker.code_predictor.get_input_embeddings()[i - 1](
                    codec_ids[:, :, i]) * cmask.unsqueeze(-1)
            out = model.talker(inputs_embeds=inp[:, :-1, :], attention_mask=amask[:, :-1],
                               labels=lab[:, 1:], output_hidden_states=True)
            hidden = out.hidden_states[0][-1]
            th = hidden[cmask[:, :-1]]; tc = codec_ids[cmask]
            _, sloss = model.talker.forward_sub_talker_finetune(tc, th)
            loss = out.loss + 0.3 * sloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            tot += loss.item(); n += 1
        print(f"Epoch {epoch} | loss {tot/max(n,1):.4f} | "
              f"target 4.1-4.2 兼音色与风格", flush=True)

    out = os.path.join(args.output_model_path, "style_checkpoint-epoch-%d" % (args.num_epochs - 1))
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    if target_emb is not None:
        torch.save(target_emb.cpu(), os.path.join(out, "target_speaker_embedding.pt"))
    print(f"[ok] style LoRA 外挂导出: {out}")
    print("下一步: python scripts/hf_lora_to_tts_gguf.py --lora-dir <out> "
          "--out D:\\AI情感\\pykits\\models\\style_lora_qwen3tts.gguf --f16")


if __name__ == "__main__":
    train_style()