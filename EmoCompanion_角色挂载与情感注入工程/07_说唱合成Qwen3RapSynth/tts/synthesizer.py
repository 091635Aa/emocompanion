# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · TTS 基座封装（间接控制面核心）

结论来自 Phase 0 控制面探查：
  公开 Qwen3-TTS Base 的底层 generate **不暴露** pitch/duration/energy。
  可取的控制面是 `voice_clone_prompt`（ref_spk_embedding / ref_code / icl_mode）与采样参数。

因此这里提供两条可控路径：
  1) x-vector 路径（默认）：用训练期导出的 `target_speaker_embedding.pt` 作为
     ref_spk_embedding，`x_vector_only_mode=True` 得到稳定音色，无需 ref_audio；
  2) ICL 路径（可选）：提供一句节奏参考人声 `--ref`，`icl_mode=True` 让模型在
     LoRA 之外再受一句真实韵律约束（作为间接节奏/腔调提示）。

本模块懒加载，未调用合成不占用显存。调用方异常（缺件/缺卡）抛结构化异常，供上层转 JSON。
"""
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.abspath(os.path.join(_HERE, "..", "..", "06_Qwen3TTS外挂", "out"))

BASE_MODEL = os.environ.get(
    "QQ_BASE_MODEL",
    r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\master",
)
ADAPTERS = {
    "voice": os.path.join(_OUT, "voice_lora", "voice_checkpoint-epoch-2"),
    "emotion": os.path.join(_OUT, "emotion_lora", "emotion_checkpoint-epoch-2"),
}
SPEAKER_EMB = os.path.join(ADAPTERS["voice"], "target_speaker_embedding.pt")
SR = 24000

EMOTION_VOCAB = {
    "开心": "[emotion]开心[/emotion]", "俏皮": "[emotion]俏皮[/emotion]",
    "悲伤": "[emotion]悲伤[/emotion]", "平静": "[emotion]平静[/emotion]",
    "兴奋": "[emotion]兴奋[/emotion]", "硬核": "[emotion]硬核[/emotion]",
}
EMOTION_FALLBACK = "硬核"


class TTSUnavailable(RuntimeError):
    pass


@dataclass
class RaSynthCore:
    base_model: str = BASE_MODEL
    adapter_dirs: dict = field(default_factory=lambda: dict(ADAPTERS))
    speaker_emb: str = SPEAKER_EMB
    _model: object = None
    _prompt: object = None
    _loaded: bool = False
    _err: Optional[str] = None

    # ---------- 装载 ----------
    def _check(self):
        missing = []
        if not os.path.isdir(self.base_model):
            missing.append(f"Base: {self.base_model}")
        for k, p in self.adapter_dirs.items():
            if not os.path.isdir(p):
                missing.append(f"外挂[{k}]: {p}")
        if not os.path.isfile(self.speaker_emb):
            missing.append(f"音色 embedding: {self.speaker_emb}")
        if missing:
            raise TTSUnavailable("缺少 TTS 组件：\n- " + "\n- ".join(missing))

    def load(self, ref_audio: Optional[str] = None, ref_text: Optional[str] = None):
        """懒加载。返回 self。ref_audio 提供时走 ICL 路径；否则 x-vector 路径。"""
        if self._loaded:
            return self
        if self._err:
            raise TTSUnavailable(self._err)
        try:
            self._check()
            import torch
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
            from peft import PeftModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            # 1) Base 只读
            model = Qwen3TTSModel.from_pretrained(
                self.base_model, torch_dtype=dtype, device_map=device,
                local_files_only=True, trust_remote_code=True,
            )
            # 2) 音色 LoRA 作为主 adapter（不污染 Base 权重）
            model.model = PeftModel.from_pretrained(model.model, self.adapter_dirs["voice"])
            # 3) 追加情感 LoRA（可选叠加）
            if os.path.isdir(self.adapter_dirs["emotion"]):
                try:
                    model.model.load_adapter(self.adapter_dirs["emotion"], adapter_name="emotion")
                except Exception:
                    model.model.set_adapter("default")
            # 4) 说话人 embedding
            emb = torch.load(self.speaker_emb, map_location="cpu")
            emb = emb.to(device).to(dtype)

            if ref_audio:
                items = model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=ref_text or "",
                    x_vector_only_mode=False,
                )
                # 用训练音色 embedding 覆盖参考音色，保留其节奏/腔调（icl_mode=True）
                items[0].ref_spk_embedding = emb
                items[0].ref_code = items[0].ref_code.to(device)
            else:
                # x-vector 路径：仅音色嵌入，无参考韵律
                from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
                items = [VoiceClonePromptItem(
                    ref_code=None, ref_spk_embedding=emb,
                    x_vector_only_mode=True, icl_mode=False, ref_text=None,
                )]
            self._prompt = items
            self._model = model
            self._loaded = True
            return self
        except TTSUnavailable:
            raise
        except Exception as e:
            self._err = f"{type(e).__name__}: {e}"
            raise TTSUnavailable(self._err)

    # ---------- 合成 ----------
    def synthesize(self, text: str, emotion: str = EMOTION_FALLBACK,
                   temperature: float = 0.9) -> Tuple[object, int, float]:
        """合成单句，返回 (float32 wav ndarray, sr, 时长秒)。"""
        if not self._loaded:
            self.load()
        pref = EMOTION_VOCAB.get(emotion, EMOTION_VOCAB[EMOTION_FALLBACK])
        input_text = f"{pref}{text}".strip()
        wavs, sr = self._model.generate_voice_clone(
            text=input_text, language="Chinese",
            voice_clone_prompt=self._prompt,
            temperature=temperature,
        )
        wav = wavs[0] if isinstance(wavs, list) and wavs else wavs
        import numpy as np
        arr = np.asarray(wav.detach().cpu().numpy() if hasattr(wav, "detach") else wav)
        return arr.astype("float32"), int(sr), arr.shape[0] / float(sr)


def get_core() -> RaSynthCore:
    if not hasattr(get_core, "_inst") or get_core._inst is None:
        get_core._inst = RaSynthCore()
    return get_core._inst