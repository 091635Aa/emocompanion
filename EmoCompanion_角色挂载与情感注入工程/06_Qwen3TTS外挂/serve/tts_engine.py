# -*- coding: utf-8 -*-
"""EmoCompanion TTS 引擎封装（惰性加载 · 双路外挂）

- Base: Qwen3-TTS-12Hz-1.7B-Base（modelscope 缓存，bf16 / CUDA 惰性加载）
- 外挂: voice(音色) LoRA + emotion(情感) LoRA + 角色 target_speaker_embedding.pt
- 对外: synthesize(text, emotion) -> (float32 wav numpy, sr=24000)
- 缺失组件时抛 TTSUnavailable（结构化），不崩溃，便于上层转 JSON 错误。

用法(服务内)：
  from tts_engine import get_engine
  eng = get_engine()
  wav, sr, meta = eng.synthesize("哥哥你回来啦", "开心")
"""
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

# -------------------- 路径常量 --------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(_HERE, "..", "out"))

# 基础模型（训练时由 modelscope 下载到该缓存，官方布局）
BASE_MODEL = os.environ.get(
    "YY_BASE_MODEL",
    r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\master",
)

# 双路外挂 adapter 目录 + 角色音色 embedding
ADAPTERS = {
    "voice": os.path.join(OUT_DIR, "voice_lora", "voice_checkpoint-epoch-2"),
    "emotion": os.path.join(OUT_DIR, "emotion_lora", "emotion_checkpoint-epoch-2"),
}
SPEAKER_EMB = os.path.join(ADAPTERS["voice"], "target_speaker_embedding.pt")

SR = 24000  # Qwen3-TTS 采样率

# 情感词表（与训练的情感标签对齐；缺失时用简单前缀回退）
EMOTION_VOCAB = {
    "开心": "[emotion]开心[/emotion]",
    "俏皮": "[emotion]俏皮[/emotion]",
    "悲伤": "[emotion]悲伤[/emotion]",
    "平静": "[emotion]平静[/emotion]",
    "兴奋": "[emotion]兴奋[/emotion]",
    "撒娇": "[emotion]撒娇[/emotion]",
}
EMOTION_FALLBACK = "平静"


class TTSUnavailable(RuntimeError):
    """TTS 组件缺失/失败的结构化异常，message 面向用户。"""


@dataclass
class TTSEngine:
    """惰性单例（进程内一实例）。"""

    base_model: str = BASE_MODEL
    adapters: dict = field(default_factory=lambda: dict(ADAPTERS))
    speaker_emb_path: str = SPEAKER_EMB

    _model: object = None      # Qwen3TTSModel
    _speaker_emb: Optional[object] = None
    _loaded: bool = False
    _load_err: Optional[str] = None

    def _check_paths(self):
        """校验关键组件存在，缺失则给出含路径的结构化错误。"""
        missing = []
        if not os.path.isdir(self.base_model):
            missing.append(f"Base 模型: {self.base_model}")
        for name, p in self.adapters.items():
            if not os.path.isdir(p):
                missing.append(f"外挂包[{name}]: {p}")
        if not os.path.isfile(self.speaker_emb_path):
            missing.append(f"角色音色 embedding: {self.speaker_emb_path}")
        if missing:
            raise TTSUnavailable(
                "缺少 TTS 关键组件，无法合成：\n- " + "\n- ".join(missing)
                + "\n提示：请确认外挂包已训练导出，或用 YY_BASE_MODEL 环境变量指定 Base 路径。"
            )

    def load(self, policy: str = None):
        """加载 TTS 模型（惰性）。幂等。policy 缺省读环境变量 YY_TTS_POLICY。

        policy:
          - bfloat16: 默认，质量优先（实测 8GB 卡基准 ~4.4GB reserved）
          - float16:  同体积
          - int8:     走 load_in_8bit 尝试降显存；**实测当前 wrapper 未真正量化(仍≈4.4GB)**，勿过度期待
        后台跑 RVC 时建议：保持 bf16(≈4.4GB)，且**不要让 04 文本 LLM 与 RVC 同卡抢显存**。
        """
        if policy is None:
            policy = os.environ.get("YY_TTS_POLICY", "bfloat16")
        if self._loaded:
            return self
        if self._load_err:
            raise TTSUnavailable(self._load_err)
        import torch
        self._policy = policy
        cuda = torch.cuda.is_available()
        if not cuda:
            self._dtype = torch.float32
        elif policy == "float16":
            self._dtype = torch.float16
        else:
            self._dtype = torch.bfloat16
        try:
            self._check_paths()
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

            device = "cuda" if cuda else "cpu"
            # 1) Base 只读加载（不污染原始权重）；int8 走 bitsandbytes 量化减显存
            kw = dict(torch_dtype=self._dtype, device_map=device,
                      local_files_only=True, trust_remote_code=True)
            if cuda:
                # attention 实现：默认 flash_attention_2（装了 flash_attn 但实测与
                # Qwen3-TTS 部分层不兼容会挂起）；可用 YY_ATTN=sdpa|eager 覆盖回退
                # 以换取稳定合成（sdpa/eager 能出结果但 RTF 较高）。
                attn_impl = os.environ.get("YY_ATTN", "flash_attention_2")
                kw["attn_implementation"] = attn_impl
            if cuda and policy == "int8":
                kw["torch_dtype"] = torch.float16
                kw["load_in_8bit"] = True
                kw.pop("attn_implementation", None)   # 8bit 权重不走 flash-attn
            try:
                model = Qwen3TTSModel.from_pretrained(self.base_model, **kw)
            except Exception as e:
                if policy == "int8":
                    print(f"[tts] int8 加载失败，回落 bf16: {e}", flush=True)
                    self._dtype = torch.bfloat16 if cuda else torch.float32
                    fb_kw = dict(torch_dtype=self._dtype, device_map=device,
                                 local_files_only=True, trust_remote_code=True)
                    if cuda:
                        fb_kw["attn_implementation"] = "flash_attention_2"
                    model = Qwen3TTSModel.from_pretrained(
                        self.base_model, **fb_kw)
                else:
                    raise
            # 2) 挂载音色 LoRA 到 talker 主干（训练目标即 talker 的 q/k/v/o/gate/up/down）
            #    包裹子模块后，model.generate 内部调用 self.talker(...) 即走 LoRA 旁路。
            self._adapter_attached = False
            self._talker = None
            self._adapters_loaded = set()
            try:
                from peft import PeftModel
                talker = getattr(model.model, "talker", None)
                if talker is not None:
                    talker = PeftModel.from_pretrained(
                        talker, self.adapters["voice"], adapter_name="voice")
                    self._adapters_loaded.add("voice")
                    if os.path.isdir(self.adapters["emotion"]):
                        try:
                            talker.load_adapter(self.adapters["emotion"], adapter_name="emotion")
                            self._adapters_loaded.add("emotion")
                        except Exception:
                            pass
                    try:
                        talker.set_adapter("emotion")   # 情感 LoRA 作为默认(律动/语气更像目标)
                    except Exception:
                        pass
                    model.model.talker = talker
                    self._talker = talker
                    self._adapter_attached = True
            except Exception as e:
                print(f"[tts] LoRA 挂载跳过(Base 直用): {e}", flush=True)
            # 3) 角色音色 embedding（训练期样本编码；x_vector_only 说话人条件）
            emb = torch.load(self.speaker_emb_path, map_location="cpu")
            if emb.dim() == 2 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            self._speaker_emb = emb.to(device=device, dtype=self._dtype)
            self._model = model
            self._loaded = True
            # 显存占用上报（用于规划 RVC 同步训练）
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                self.gpu_used_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)
                self.gpu_reserved_mb = round(torch.cuda.memory_reserved() / 1e6, 1)
            except Exception:
                self.gpu_used_mb = self.gpu_reserved_mb = 0.0
            return self
        except TTSUnavailable:
            raise
        except Exception as e:  # 留痕以供诊断，不静默
            self._load_err = f"TTS 加载失败: {type(e).__name__}: {e}"
            raise TTSUnavailable(self._load_err)

    # ---------------- 生成策略 ----------------
    def _set_active_adapter(self, name: str):
        """切换激活的 LoRA adapter：voice(音色)/emotion(情感律动)。"""
        t = getattr(self, "_talker", None)
        if t is not None and name in getattr(self, "_adapters_loaded", set()):
            try:
                t.set_adapter(name)
            except Exception as e:
                print(f"[tts] 切换 adapter={name} 失败: {e}", flush=True)

    def _compose_text(self, text: str, emotion: str) -> str:
        # 关键：本模型 tokenizer 无情感 token，`[emotion]…[/emotion]` 会被当口播文本念出来。
        # 因此不再把标签塞进"要念的文本行"；情感一律走原生参考音频机制。
        return (text or "").strip()

    def _resolve_refs(self, emotions) -> list:
        """按情感列表批量解析参考音频资产 serve/refs/<emotion>.wav(+同名.txt 转写)。

        返回 list[dict{audio, text, emotion}]：
          - text 统一带 `[情感]` 前缀，让模型学到「标签 ↔ 音频韵律」映射（该前缀只进
            ICL prompt 做条件，不会被朗读出来）。
          - 返回的 refs 可一次传入 generate_voice_clone 实现多段 ICL。
        不存在对应 wav 的情感会被跳过。
        """
        refs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
        if isinstance(emotions, str):
            emotions = [emotions]
        out = []
        for emo in emotions:
            wav = os.path.join(refs_dir, f"{emo}.wav")
            if not os.path.isfile(wav):
                continue
            txt = os.path.join(refs_dir, f"{emo}.txt")
            ref_text = None
            if os.path.isfile(txt):
                raw = open(txt, "r", encoding="utf-8").read().strip()
                if raw:
                    ref_text = f"[{emo}]{raw}"
            out.append({"audio": wav, "text": ref_text, "emotion": emo})
        return out

    def _resolve_ref(self, emotion: str):
        """单情感便捷解析（兼容旧调用），返回 dict 或 None。"""
        refs = self._resolve_refs([emotion])
        return refs[0] if refs else None

    def _merge_refs_to_single(self, refs: list, gap_s: float = 0.1) -> dict:
        """把多段参考音频拼成"一个多情感参考片段"（单段 audio + 多标签文本）。

        Qwen3-TTS 的 generate_voice_clone 要求 voice_clone prompt 数 == text 数，
        多段 ref 只支持 batch 模式（N 段 ref ↔ N 条文本）。因此"一个片段含多个
        情感标签和原文"只能合并为单段传入：音频拼接成一段、文本拼成
        `[情感1]原文1 [情感2]原文2 ...`，让模型在 ICL prompt 里学到多情感韵律。

        返回 {"audio": (np.ndarray, sr), "text": str, "emotions": [...]}。
        """
        import numpy as np
        import librosa
        segs, texts, emos, sr = [], [], [], None
        for r in refs:
            wav, s = librosa.load(r["audio"], sr=None, mono=True)
            if sr is None:
                sr = s
            if s != SR:
                wav = librosa.resample(wav.astype("float32"), orig_sr=s, target_sr=SR)
            segs.append(np.asarray(wav, dtype="float32"))
            texts.append(r.get("text") or "")
            emos.append(r.get("emotion"))
        sr = SR
        gap = np.zeros(int(sr * gap_s), dtype="float32")
        pieces = []
        for i, w in enumerate(segs):
            pieces.append(w)
            if i < len(segs) - 1:
                pieces.append(gap)
        joined = np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")
        text = " ".join(t for t in texts if t).strip()
        return {"audio": (joined.astype("float32"), sr), "text": text, "emotions": emos}

    def synthesize(self, text: str, emotion: str = EMOTION_FALLBACK,
                   adapter: str = "emotion", tone_variation: float = 0.35,
                   seed: Optional[int] = None,
                   temperature: Optional[float] = None,
                   top_k: Optional[int] = None, top_p: Optional[float] = None,
                   repetition_penalty: Optional[float] = None,
                   refs: Optional[list] = None,
                   ref: Optional[dict] = None,
                   max_new_tokens: Optional[int] = None,
                   subtalker_dosample: Optional[bool] = None) -> Tuple[object, int, dict]:
        """合成语音，返回 (float32 wav numpy, sr, meta)。meta 含耗时/RTF/随机参数。

        - 情感控制：优先用 serve/refs/<emotion>.wav 走原生 ref_audio(ICL)（多段一次传入）；
          无 ref 时回落"干净文本 + 情感 adapter + 随机语气"，绝不再往文本里塞标签。
        - refs: 可选，list[dict{audio,text,emotion}]，一次传入实现多段 ICL
          （例如 [角色音色锚点, 目标情感段]）。缺省时按 emotion 动态选一段。
        - ref:  可选，单段 dict 快捷入参（旧接口兼容）。
        - max_new_tokens: 默认 300（≈25s 音频上限）。Qwen3-TTS 默认 4096，模型不触
          EOS 时会生成几百秒音频，表现为"假挂起"；限制后即使不 EOS 也能按时返回。
        - subtalker_dosample: 可选关掉 sub-talker 采样(微提速)。
        - adapter: voice(音色) / emotion(情感律动,默认)
        - tone_variation: 随机语气强度 0~1。
        """
        if self._model is None:
            self.load()
        model = self._model
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        self._set_active_adapter(adapter)
        if seed is not None:
            random.seed(seed)
            try:
                import torch as _t
                _t.manual_seed(seed)
                if _t.cuda.is_available():
                    _t.cuda.manual_seed_all(seed)
            except Exception:
                pass
        rng = random.Random(random.random())

        # ---- 随机语气变量：在 lively 区间采样，避免固定语气 ----
        t = temperature if temperature is not None else \
            (0.85 + rng.uniform(-0.20, 0.30) * max(0.0, min(tone_variation, 1.0)))
        k = top_k if top_k is not None else rng.choice([40, 50, 60])
        p = top_p if top_p is not None else \
            min(1.0, 0.86 + rng.uniform(-0.04, 0.12) * max(0.0, min(tone_variation, 1.0)))
        rp = repetition_penalty if repetition_penalty is not None else \
            (1.0 + rng.uniform(0.0, 0.10))
        t = max(t, 0.7)

        input_text = self._compose_text(text, emotion)   # 干净文本(不含任何标签)
        if ref is not None:
            refs = [ref]
        if refs is None:
            refs = self._resolve_refs([emotion])
        # 生成上限：默认 300（≈25s），防模型不触 EOS 时 4096 帧"假挂起"
        mnt = max_new_tokens if max_new_tokens is not None else 300
        gen_kw = dict(temperature=t, top_k=k, top_p=p,
                      repetition_penalty=rp, max_new_tokens=mnt)
        if subtalker_dosample is not None:
            gen_kw["subtalker_dosample"] = subtalker_dosample
        t0 = time.perf_counter()
        try:
            if refs:
                # 多段情感参考 → 合并为单段"多情感参考片段"（Qwen3-TTS 要求
                # prompt 数==text 数，多段只能走 batch 模式；合并后以单段 ICL 承载
                # 多情感韵律，ref 文本带 [情感] 标签，不会被朗读）。
                if len(refs) == 1:
                    ref_audio = refs[0]["audio"]
                    ref_text = [refs[0].get("text")]
                    used_ref = refs[0]["audio"]
                    n_refs = 1
                else:
                    merged = self._merge_refs_to_single(refs)
                    ref_audio = merged["audio"]      # (np.ndarray, sr)
                    ref_text = [merged["text"]]
                    used_ref = merged["emotions"]
                    n_refs = len(refs)
                wavs, sr = model.generate_voice_clone(
                    text=input_text, language="Chinese",
                    ref_audio=ref_audio, ref_text=ref_text,
                    x_vector_only_mode=False, **gen_kw,
                )
            else:
                prompt = [VoiceClonePromptItem(
                    ref_code=None, ref_spk_embedding=self._speaker_emb,
                    x_vector_only_mode=True, icl_mode=False, ref_text=None,
                )]
                wavs, sr = model.generate_voice_clone(
                    text=input_text, language="Chinese",
                    voice_clone_prompt=prompt, x_vector_only_mode=True,
                    **gen_kw,
                )
                used_ref = None
                n_refs = 0
        except Exception as e:
            raise TTSUnavailable(f"voice_clone 生成失败: {type(e).__name__}: {e}")
        dt = time.perf_counter() - t0
        if not wavs or wavs[0] is None:
            raise TTSUnavailable("生成返回空音频")
        import numpy as np
        arr = wavs[0]
        arr = np.asarray(arr) if not hasattr(arr, "cpu") else arr.detach().cpu().numpy()
        dur = arr.shape[0] / float(sr) if arr.ndim > 0 else 0.0
        rtf = round(dt / dur, 3) if dur > 0 else None
        meta = {
            "strategy": "ref_icl" if refs else "voice_clone_emb",
            "emotion": emotion, "sr": int(sr),
            "ref_audio": used_ref, "n_refs": n_refs,
            "seconds": round(dt, 3), "rtf": rtf, "audio_seconds": round(dur, 3),
            "adapter_attached": getattr(self, "_adapter_attached", False),
            "adapter": adapter, "seed": seed,
            "gen": {"temperature": round(t, 3), "top_k": k, "top_p": round(p, 3),
                    "repetition_penalty": round(rp, 3), "max_new_tokens": mnt},
        }
        return arr.astype("float32"), int(sr), meta

    # ---------------- 分段流式：每一句/每一小句可控情感 ----------------
    _SENT_END = re.compile(r"[。！？…!?；;，,、～~]")

    def make_schedule(self, text: str, emotions=None) -> list:
        """按句切分文案并分配情感标签，产出 [(句文本, 情感), ...]。

        emotions:
          - None  -> 每句自动识别(with未接文本引擎则走词典)
          - list  -> 长度应与句数一致，逐句给定
          - dict  -> {句索引0起: 情感} 局部覆盖，其余自动
        """
        segs = [p.strip() for p in re.split(self._SENT_END, text) if p and p.strip()]
        if not segs:
            segs = [text.strip()] or [" "]

        # 先给每个小句留占位
        tgt: list = [(s, None) for s in segs]
        if isinstance(emotions, list):
            for i, (s, _) in enumerate(tgt):
                if i < len(emotions) and emotions[i]:
                    tgt[i] = (s, str(emotions[i]))
        elif isinstance(emotions, dict):
            for i, (s, _) in enumerate(tgt):
                if i in emotions:
                    tgt[i] = (s, str(emotions[i]))
        # 其余未指定 → 自动逐句识别（懒加载 emo_detect，避免顶层循环 import）
        from emo_detect import detect_emotion
        out = [(s, e if e else detect_emotion(s, None).label) for s, e in tgt]
        return out

    def synthesize_flow(self, text: str, emotions=None, adapter: str = "emotion",
                        tone_variation: float = 0.35, seed: Optional[int] = None,
                        gap_s: float = 0.05,
                        tone_jitter_per_seg: bool = True,
                        ) -> Tuple[object, int, dict]:
        """分句合成并拼接为整段音频。每一句独立情感标签 → 情感浮动实时可见。

        - tone_jitter_per_seg: 每句再随机一次语气(温度等)，进一步拉开句子间浮动；
          =False 则全篇共享同一套随机采样参数(语气统一、音色更一致)。
        """
        if seed is not None:
            random.seed(seed)
        schedule = self.make_schedule(text, emotions)
        wavs, srs, seg_metas, cum = [], [], [], 0.0
        for s, e in schedule:
            # 每句随机语气：jitter 开 → 用新熵(句间浮动大)；关 → 共享同一种子(语气统一)
            seg_seed = None if tone_jitter_per_seg else seed
            w, sr, m = self.synthesize(s, e, adapter=adapter,
                                       tone_variation=tone_variation, seed=seg_seed)
            wavs.append(w)
            srs.append(sr)
            m = dict(m); m["segment"] = s; m["emotion"] = e
            seg_metas.append(m)
            cum += m["seconds"]
        sr = srs[0]
        import numpy as np
        gap = np.zeros(int(sr * gap_s), dtype="float32")
        pieces = []
        for i, w in enumerate(wavs):
            pieces.append(np.asarray(w, dtype="float32"))
            if i < len(wavs) - 1:
                pieces.append(gap)
        joined = np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")
        dur = joined.shape[0] / float(sr)
        meta = {
            "strategy": "flow_segments", "adapter": adapter, "sr": int(sr),
            "n_segments": len(schedule),
            "segments": seg_metas,
            "emotions_used": [m["emotion"] for m in seg_metas],
            "audio_seconds": round(dur, 3),
            "seconds": round(cum, 3),
            "rtf": round(cum / dur, 3) if dur > 0 else None,
        }
        return joined.astype("float32"), int(sr), meta


def get_engine() -> TTSEngine:
    """进程级惰性单例。"""
    if not hasattr(get_engine, "_inst") or get_engine._inst is None:
        get_engine._inst = TTSEngine()
    return get_engine._inst


def list_adapter_names():
    return {"samplerate": SR,
            "adapters": {k: v for k, v in ADAPTERS.items()},
            "emotions": list(EMOTION_VOCAB.keys()),
            "base": BASE_MODEL}


if __name__ == "__main__":
    import sys
    print(list_adapter_names())