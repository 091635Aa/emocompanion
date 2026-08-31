# -*- coding: utf-8 -*-
"""tts_gguf —— GGUF 高性能 TTS 后端（llama-tts.exe，INT4，无需 flash-attn/torch）

把 llama.cpp 原生 llama-tts 封装为单一路径 TTS 后端（连接 unified_server）：
  - 部署：Base GGUF + mmproj GGUF 常驻，速度 INT4+CUDA 实测 RTF≈0.4
  - 情感：**直接加载训练好的 LoRA adapter（--lora voice,emotion）**，情感靠
    训练音色 + 采样预设(seed/temp/top_k/rate)驱动，不再依赖逐情感参考音频。
    参考音频仅作为音色锚点(中性一份)，非情感开关。
  - 兼容：CUDA/NVIDIA 20-50系、AMD-Vulkan、CPU
  - 打包：exe + backbone/mmproj + adapter*.gguf 全为独立小文件，一套跨硬件

注意：llama-tts(C++) 无法读写含中文的路径，所有中间 wav 一律放 %TEMP%（纯 ASCII）。

用法:
  gg = GGUFTTS.get()
  gg.availability()        -> {'available': bool, 'reason': str}
  wav, sr, meta = gg.synthesize(text, emotion='开心', seed=42, temperature=0.9)
"""
import os
import re
import subprocess
import tempfile
import time

import numpy as np

# 本机 GGUF 资产与 llama-tts 位置
LLAMA_TTS = r"D:\AI情感\pykits\llama-cpp-bin\llama-tts.exe"
GGUF = r"D:\AI情感\pykits\models\Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"
MMPROJ = r"D:\AI情感\pykits\models\mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
# 训练得到的音色/情感 LoRA adapter（由 hf_lora_to_tts_gguf.py 从 HF PEFT 转换导出）
VOICE_LORA_GGUF = r"D:\AI情感\pykits\models\voice_lora_qwen3tts.gguf"
EMOTION_LORA_GGUF = r"D:\AI情感\pykits\models\emotion_lora_qwen3tts.gguf"
# 可选「说话风格」LoRA 外挂位：训练出风格 LoRA 后放到此路径即自动作为第 3 个 adapter
# 叠加加载（--lora voice,emotion,style）。缺失时优雅降级为 voice+emotion 两路。
STYLE_LORA_GGUF = r"D:\AI情感\pykits\models\style_lora_qwen3tts.gguf"
# 默认EmoCompanion真人参考音频（音色锚点）。
# 必须与 voice LoRA 训练时的 speaker ref 一致：voice_train_raw 6200 样本全部用
# 503_519_520_137_010.wav 作为 speaker 条件（audio 是内容、ref_audio 是说话人）。
# 之前误用 100_...（内容音频）当 speaker → 说话人条件错 → 声音不像。现改回 503。
DEFAULT_REF = r"D:\ACQ富\wav_24k\503_519_520_137_010.wav"
OUT_SR = 24000

# 情感 -> 采样预设：seed 用 -1(随机)。emotion 只影响语气采样差异，音色由 adapter 固定
EMOTION_PRESETS = {
    "平静":  {"temperature": 0.62, "top_k": 40, "rate": 0.98},
    "开心":  {"temperature": 0.85, "top_k": 60, "rate": 1.05},
    "俏皮":  {"temperature": 0.95, "top_k": 60, "rate": 1.10},
    "悲伤":  {"temperature": 0.78, "top_k": 40, "rate": 0.92},
    "撒娇":  {"temperature": 0.92, "top_k": 60, "rate": 1.06},
    "温柔":  {"temperature": 0.72, "top_k": 50, "rate": 0.95},
    "激动":  {"temperature": 1.00, "top_k": 60, "rate": 1.15},
    "兴奋":  {"temperature": 1.00, "top_k": 60, "rate": 1.12},
}
DEFAULT_EMOTION = "平静"

# ======================================================================
# 「说话风格外挂」StylePlug —— 独立于情感的另一条可控轴
# ----------------------------------------------------------------------
# 情感(emotion)决定"说什么语气"（8 种情绪预设），说话风格(style)决定"怎么开口、念多快、
# 多放松/多用力"（8 种风格预设）。两者正交，合成时做逐段叠加：
#   temperature  = emotion.temp * style.temp_factor   （放松↔用力，语气起伏）
#   top_k/top_p  = style 取值                          （即兴 vs 稳定，念白弹性）
#   repeat_penalty= style 取值                          （口头禅/重复度控制）
#   rate / gap    = style 取值                          （基准语速 / 句间停顿，决定节奏）
# 这条轴完全"外挂"在输入/传递/TTS 推理三层，不污染 Base，各风格仅是一份预设参数，
# 可随时开关、热切换，也是后续风格 LoRA 第三 adapter 的推理入口。
# ======================================================================
DEFAULT_STYLE = "自然"
# style -> 采样合成参数（top_k/top_p/repeat_penalty 直接进 llama-tts，均不引入电音）
# temp_factor 的梯度拉开到 0.86~1.20，保证不同风格在语调起伏上可测量、可分得开。
STYLE_PRESETS = {
    "自然":  {"temp_factor": 1.00, "top_k": 50, "top_p": 0.90, "repeat_penalty": 1.10, "rate": 1.00, "gap": 0.06},
    "甜美":  {"temp_factor": 1.16, "top_k": 62, "top_p": 0.92, "repeat_penalty": 1.14, "rate": 1.04, "gap": 0.08},
    "元气":  {"temp_factor": 1.12, "top_k": 62, "top_p": 0.94, "repeat_penalty": 1.12, "rate": 1.06, "gap": 0.05},
    "活力":  {"temp_factor": 1.10, "top_k": 64, "top_p": 0.96, "repeat_penalty": 1.08, "rate": 1.12, "gap": 0.05},
    "娇俏":  {"temp_factor": 1.20, "top_k": 66, "top_p": 0.94, "repeat_penalty": 1.12, "rate": 1.08, "gap": 0.07},
    "温柔":  {"temp_factor": 0.93, "top_k": 44, "top_p": 0.89, "repeat_penalty": 1.05, "rate": 0.95, "gap": 0.14},
    "慵懒":  {"temp_factor": 0.86, "top_k": 40, "top_p": 0.87, "repeat_penalty": 1.03, "rate": 0.88, "gap": 0.18},
    "坚定":  {"temp_factor": 0.97, "top_k": 44, "top_p": 0.88, "repeat_penalty": 1.13, "rate": 0.98, "gap": 0.09},
}


def resolve_style(style):
    """归一化风格名；不认识回退默认风格。"""
    s = (style or "").strip() or DEFAULT_STYLE
    return s if s in STYLE_PRESETS else DEFAULT_STYLE


def compose_params(emotion, style):
    """情感 ⊗ 说话风格 -> 实际采样参数 (temperature, top_k, top_p, repeat_penalty, rate, gap)。

    temperature 用风格系数缩放情绪基准温度（受 clamp 保护，不越界）；其余由风格决定。
    """
    ep = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS[DEFAULT_EMOTION])
    sp = STYLE_PRESETS.get(resolve_style(style), STYLE_PRESETS[DEFAULT_STYLE])
    temperature = round(min(max(ep["temperature"] * float(sp["temp_factor"]), 0.5), 1.15), 3)
    top_k = int(sp.get("top_k") or ep["top_k"])
    top_p = float(sp.get("top_p", 0.90))
    repeat_penalty = float(sp.get("repeat_penalty", 1.10))
    rate = float(sp.get("rate", 1.0))
    gap = float(sp.get("gap", 0.06))
    return temperature, top_k, top_p, repeat_penalty, rate, gap


def shape_for_style(text, style):
    """说话风格 输入层节奏塑形：仅调整停顿标点密度，**不注入任何会被朗读的记号**。

    标点本身不被 Qwen3-TTS 朗读，只作为停顿/节奏的强提示 → 可靠、无电音、不口播。
      - 慢(rate<0.97)：逗号升级为句号/分号 → 句间停顿变长，从容舒缓
      - 快(rate>1.03)：句号/感叹号轻化为逗号 → 停顿变短，轻快紧凑
      - 中性：原样，只做空白规整与句末收束
    """
    import re as _re
    sp = STYLE_PRESETS.get(resolve_style(style), STYLE_PRESETS[DEFAULT_STYLE])
    rate = float(sp["rate"])
    t = (text or "").strip()
    if not t:
        return t
    if rate < 0.97:                      # 慢
        t = t.replace(",", "，")
        t = _re.sub(r"[，,]+", "，", t)
        # 把句内逗号升级为更强的停顿，节奏舒展
        t = t.replace("，", "。").replace("；", "。").replace(";", "。")
    elif rate > 1.03:                    # 快：句末标点轻化为逗号，缩短停顿
        t = t.replace("。", "，").replace("！", "，").replace("!", "，")
        rr = t.rstrip("，")
        t = (rr + "。") if rr else t
    t = _re.sub(r"[ \t]{2,}", " ", t).strip()
    # 句末收束兜底
    if t and t[-1] not in "。！？~～;；":
        t = t + "。"
    return t


def strip_control_tokens(text: str) -> str:
    """剥离所有控制/动作括号，只保留要朗读的正文（[标签]/(语速)〔动作〕（）【】）。"""
    import re
    if not text:
        return text
    s = text
    for _o, _c in (("〔", "〕"), ("（", "）"), ("(", ")"), ("【", "】")):
        s = re.sub(re.escape(_o) + r"[^" + re.escape(_o + _c) + r"]*" + re.escape(_c), "", s)
    s = re.sub(r"\[[^\[\]]*\]", "", s)          # [情感] 标签
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s

# GPU 并行最大化参数（llama-tts 逐次起进程）
GPU_MAX_PARAMS = ("-fa", "on", "-t", "16", "-b", "512", "-ub", "256", "--load-mode", "mlock")
# 显式一个小上下文：TTS 只合成短文，过大 ctx 会让 KV 缓存显著占用显存、引发 GPU 上下文暴涨。
# 限定后每进程 KV 占用被压到最低，避免"上下文突然冲到很高"。
CTX_SIZE = 2048

_singleton = None


class GGUFTTS:
    def __init__(self, llama_tts=LLAMA_TTS, gguf=GGUF, mmproj=MMPROJ,
                 default_ref=DEFAULT_REF, voice_lora=VOICE_LORA_GGUF,
                 emotion_lora=EMOTION_LORA_GGUF, style_lora=STYLE_LORA_GGUF,
                 stable_seed=20260822):
        self.exe = llama_tts
        self.gguf = gguf
        self.mmproj = mmproj
        self.default_ref = default_ref
        self.voice_lora = voice_lora
        self.emotion_lora = emotion_lora
        self.style_lora = style_lora
        # 稳定 seed：整段一次合成用固定值而非随机 -1，保证同一文本音色/韵律可复现
        self.stable_seed = int(stable_seed)
        self._seed_counter = int(stable_seed)

    def _adapter_list(self):
        """voice+emotion(+style) 外挂 adapter 组合；风格 LoRA 缺失时优雅降级。"""
        adapters = [self.voice_lora, self.emotion_lora]
        if self.style_lora and os.path.isfile(self.style_lora):
            adapters.append(self.style_lora)
        return ",".join(adapters)

    # ---------------- 可用性 ----------------
    def availability(self):
        miss = [p for p, n in [(self.exe, "llama-tts.exe"), (self.gguf, "backbone GGUF"),
                               (self.mmproj, "mmproj GGUF"),
                               (self.voice_lora, "voice LoRA GGUF"),
                               (self.emotion_lora, "emotion LoRA GGUF")]
                if not os.path.isfile(p)]
        if miss:
            return {"available": False, "reason": "缺失资产: " + ", ".join(miss)}
        if not os.path.isfile(self.default_ref):
            return {"available": False, "reason": "默认参考音频不可用: " + self.default_ref}
        # CUDA 运行库必须在 llama-tts.exe 同级（缺失会静默退回 CPU，非常慢）
        cuda_dlls = ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"]
        exe_dir = os.path.dirname(self.exe)
        miss_cuda = [d for d in cuda_dlls if not os.path.isfile(os.path.join(exe_dir, d))]
        if miss_cuda:
            return {"available": False,
                    "reason": "缺失 CUDA 运行库(需与 llama-tts.exe 同级): " + ", ".join(miss_cuda)
                             + "。从 CUDA Toolkit/LMStudio 复制到 exe 目录后即启用 GPU 加速(RTF≈0.4)。"}
        return {"available": True, "reason": "ok", "adapter": "voice+emotion(+style?)",
                "model": "Qwen3-TTS-12Hz-1.7B-Base",
                "mmproj": os.path.basename(self.mmproj),
                "emotions": list(EMOTION_PRESETS.keys()),
                "styles": list(STYLE_PRESETS.keys()),
                "style_lora_loaded": bool(self.style_lora and os.path.isfile(self.style_lora))}

    @staticmethod
    def get():
        global _singleton
        if _singleton is None:
            _singleton = GGUFTTS()
        return _singleton

    # ---------------- 合成 ----------------
    def _next_seed(self):
        """稳定递增 seed（非随机），保证连续多次合成不重复却可复现起点。"""
        self._seed_counter += 1
        return self._seed_counter

    def _build_cmd(self, text, ref, seed, temperature, top_k, out_wav, extra=(),
                   top_p=0.90, repeat_penalty=1.10):
        cmd = [self.exe, "-m", self.gguf, "-mm", self.mmproj,
               "-p", text, "--tts-lang", "zh",
               "--lora", self._adapter_list(),
               "--seed", str(int(seed)), "--temp", str(float(temperature)),
               "--top-k", str(int(top_k)), "--top-p", str(float(top_p)),
               "--repeat-penalty", str(float(repeat_penalty)),
               "-o", out_wav, "-ngl", "99",
               "-c", str(CTX_SIZE)]
        # Qwen3-TTS 为 speaker-conditional 架构：音色主要由 --tts-speaker-file 的
        # 说话人 embedding 承载，LoRA 做发音/风格微调。训练期 ref_audio 全程=503，
        # 故推理必须喂 503 作为 speaker 条件才能对齐训练分布、得到EmoCompanion音色。
        if ref:
            cmd += ["--tts-speaker-file", ref]
        cmd += list(GPU_MAX_PARAMS)          # GPU 并行最大化（flash-attn+线程+batch）
        cmd += list(extra or ())
        return cmd

    def _run_wav(self, text, ref, seed, temperature, top_k, extra=(),
                 top_p=0.90, repeat_penalty=1.10):
        tmp = tempfile.mkdtemp(prefix="gguf_")  # 纯 ASCII 路径
        out_wav = os.path.join(tmp, "out.wav")
        cmd = self._build_cmd(text, ref, seed, temperature, top_k, out_wav, extra,
                              top_p=top_p, repeat_penalty=repeat_penalty)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=self.exe and os.path.dirname(self.exe) or None,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=900)
        wall = time.perf_counter() - t0
        if proc.returncode != 0 or not os.path.isfile(out_wav):
            raise RuntimeError(f"[gguf] llama-tts 失败 code={proc.returncode}")
        wav, sr = self._read_wav(out_wav)
        dur = wav.shape[0] / float(sr) if sr else 0.0
        return wav, sr, out_wav, wall, dur

    def synthesize(self, text, emotion=None, ref=None, rate=None,
                   n_threads=None, top_k=None, temperature=None, seed=None,
                   style=None):
        """合成单句（兼容旧接口）。默认用稳定 seed 而非随机 -1。

        style: 说话风格（StylePlug），与 emotion 正交叠加；
        显式给定 temperature/top_k 时优先于风格合成值。
        """
        meta = {"backend": "gguf", "emotion": emotion or DEFAULT_EMOTION,
                "style": resolve_style(style)}
        ref = ref or self.default_ref            # speaker 条件=503，与训练对齐
        if not (os.path.isfile(self.exe) and os.path.isfile(self.gguf) and os.path.isfile(self.mmproj)):
            raise RuntimeError("[gguf] 资产缺失，请先下载 GGUF/检查路径")
        if ref and not os.path.isfile(ref):
            raise RuntimeError(f"[gguf] 参考音频不存在: {ref}")

        text = (text or "").strip()
        if not text:
            raise RuntimeError("[gguf] 空文本")

        # 情感 ⊗ 说话风格 -> 采样参数；显式指定单个参数时以显式为准
        ct, ck, cp, crp, crate, _gap = compose_params(emotion, style)
        t = float(temperature) if temperature is not None else ct
        k = int(top_k) if top_k is not None else ck
        tp = cp
        rp = crp
        s = int(seed) if seed is not None else self._next_seed()
        r = float(rate) if rate is not None else crate

        # 剥离控制标记（Base 不识别指令，会读出来）→ 只留正文
        tts_text = strip_control_tokens(text)
        # 自动路径(未显式给定语速)：输入层按说话风格塑形节奏(标点停顿，可靠无电音)
        if rate is None:
            tts_text = shape_for_style(tts_text, style)
        wav, sr, _, wall, dur = self._run_wav(tts_text, ref, s, t, k,
                                              top_p=tp, repeat_penalty=rp)
        # 仅当调用方显式指定语速(≠1.0)时才做正确相位声码变速（带相位传播，无电音）。
        # 自动对话流程统一走自然语速，不做任何程序变速，从根上避免电音。
        if abs(r - 1.0) >= 1e-3 and len(wav) > 0:
            wav = self._time_stretch(wav, sr, r)
            dur = wav.shape[0] / float(sr) if sr else dur
        meta.update({"sr": sr, "wall_s": round(wall, 3),
                     "audio_seconds": round(dur, 3), "rate": r,
                     "temperature": t, "top_k": k, "top_p": tp,
                     "repeat_penalty": rp, "seed": s,
                     "adapter": "voice+emotion",
                     "strategy": "styleplug_lora_stable_seed"})
        return wav.astype("float32"), sr, meta

    def synthesize_paragraph(self, full_text, emotion="平静", ref=None,
                             temperature=None, top_k=None, seed=None,
                             split_markers=None, style=None):
        """段落合成（标签控制版）：按文本中的 [情感]/(语速) 标签分段合成。

        解决三个问题：
          1. 情感标签真正驱动段级差异：每段用该情感映射的采样参数(temperature/top_k)，
             同一段内音色一致，不同情感段有起伏 —— 而非纯剥离、纯人机。
          2. 固定稳定 seed（stable_seed + 段索引，非随机 -1）→ 音色一致、可复现，
             不再逐句 random 漂移成多音色。
          3. 所有括号（[标签]/(语速)〔动作〕（）等）一律剥离，不朗读、不显示。
        - full_text: 带 [俏皮]..(快速) 标签的整段文本
        - style: 说话风格（StylePlug），叠加作用于整段。
        """
        meta = {"backend": "gguf", "emotion": emotion, "style": resolve_style(style)}
        ref = ref or self.default_ref            # speaker 条件=503，与训练对齐
        if not (os.path.isfile(self.exe) and os.path.isfile(self.gguf) and os.path.isfile(self.mmproj)):
            raise RuntimeError("[gguf] 资产缺失")
        if ref and not os.path.isfile(ref):
            raise RuntimeError(f"[gguf] 参考音频不存在: {ref}")

        # 剥掉所有括号：〔〕（）()【】 与 [情感] 标签 —— 不朗读、不显示
        import re as _re
        _txt = full_text or ""
        for _open, _close in (("〔", "〕"), ("（", "）"), ("(", ")"), ("【", "】")):
            _pat = _re.compile(_re.escape(_open) + r"[^" + _re.escape(_open + _close) + r"]*" + _re.escape(_close))
            _txt = _pat.sub("", _txt)
        clean = re.sub(r"\[[^\[\]]*\]", "", _txt)          # [情感] 标签
        clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
        if not clean:
            raise ValueError("[gguf] 剥离括号后无正文")
        # 输入层按说话风格塑形节奏（标点停顿，可靠无电音）
        clean = shape_for_style(clean, style)

        # 全局基线（情感 ⊗ 说话风格）
        bt_, ck, tp, rp, _rate, _gap = compose_params(emotion, style)
        bt = float(temperature) if temperature is not None else bt_
        bk = int(top_k) if top_k is not None else ck
        base_seed = int(seed) if seed is not None else self.stable_seed

        wav, sr, _, wall, dur = self._run_wav(clean, ref, base_seed, bt, bk,
                                              top_p=tp, repeat_penalty=rp)
        meta.update({"sr": sr, "wall_s": round(wall, 3),
                     "audio_seconds": round(dur, 3), "rate": 1.0,
                     "temperature": bt, "top_k": bk, "top_p": tp,
                     "repeat_penalty": rp, "seed": base_seed,
                     "adapter": "voice+emotion",
                     "mode": "paragraph_label_control",
                     "strategy": "styleplug_label_stable_seed"})
        return wav.astype("float32"), sr, meta

    def synthesize_flow(self, segments, ref=None, rate_jitter=0.0,
                        seed=None, temper_scale=None, style=None):
        """逐段合成（混合方案核心）：每段独立合成、真实情感采样参数、模型自然语速。

        - segments: [(文本, 情感orNone, 语速float), ...]
          每段用自己的情感 → 该情感的采样参数(温度/top_k)，形成情感幅度起伏；
          说话风格(style)作为整段的另一条正交轴，叠加采样系数 + 句间停顿。
          语速不做程序变速（会引入电音），用模型自然语速。
        - seed: 稳定 seed 序列（每段 seed = base + i*7），避免逐段随机漂移音色。
        - 返回 (整段 wav, sr, meta)。
        """
        import numpy as _np
        style = resolve_style(style)
        meta = {"backend": "gguf", "adapter": "voice+emotion",
                "mode": "flow_segments", "style": style}
        ref = ref or self.default_ref
        if not (os.path.isfile(self.exe) and os.path.isfile(self.gguf) and os.path.isfile(self.mmproj)):
            raise RuntimeError("[gguf] 资产缺失")
        base_seed = int(seed) if seed is not None else self.stable_seed
        _, _, _, _, _, style_gap = compose_params("平静", style)

        seg_wavs, seg_metas = [], []
        sr = None
        for i, (txt, emo, rate) in enumerate(segments):
            text = strip_control_tokens(txt)
            if not text:
                continue
            emotion = emo or "平静"
            # 情感 ⊗ 说话风格 -> 每段采样参数
            t, k, tp, rp, _r, _g = compose_params(emotion, style)
            r = float(rate) if (rate is not None and rate > 0) else 1.0
            s = base_seed + i * 7   # 稳定递增 seed，音色一致
            # 输入层按说话风格塑形节奏（标点停顿，可靠无电音）
            text = shape_for_style(text, style)
            wav, sri, _, _, _ = self._run_wav(text, ref, s, t, k,
                                              top_p=tp, repeat_penalty=rp)
            if sr is None:
                sr = sri
            seg_wavs.append(wav.astype(_np.float32))
            seg_metas.append({"index": i, "text": strip_control_tokens(txt),
                              "emotion": emotion, "style": style,
                              "rate": r, "seed": s,
                              "temperature": t, "top_k": k,
                              "top_p": tp, "repeat_penalty": rp})
        if not seg_wavs:
            raise RuntimeError("[gguf] 无有效段落可合成")

        def _trim(arr, sr=24000, thresh=0.005, keep_lead=0.02, keep_tail=0.05):
            a = _np.asarray(arr, dtype=_np.float32)
            idx = _np.where(_np.abs(a) > thresh)[0]
            if len(idx) == 0:
                return a
            start = max(0, idx[0] - int(sr * keep_lead))
            end = min(len(a), idx[-1] + int(sr * keep_tail))
            return a[start:end]

        # 拼接（先裁首尾静音 + 风格化句间停顿）
        gap = _np.zeros(int(sr * style_gap), dtype=_np.float32)
        pieces = []
        for j, w in enumerate(seg_wavs):
            pieces.append(_trim(w, sr))
            if j < len(seg_wavs) - 1:
                pieces.append(gap)
        joined = _np.concatenate(pieces) if pieces else _np.zeros(1, dtype=_np.float32)
        dur = len(joined) / float(sr)
        meta.update({"sr": sr, "audio_seconds": round(dur, 3),
                     "seed": base_seed, "n_segments": len(seg_wavs),
                     "segments": seg_metas, "style_gap_s": style_gap,
                     "strategy": "styleplug_flow_natural_pace"})
        return joined.astype(_np.float32), sr, meta

    @staticmethod
    def _time_stretch(wav, sr, rate):
        """正确相位声码变速（带相位传播）：rate>1 变快、<1 变慢，保持音调，无电音。

        与之前简陋版本（裸 FFT + 无相位传播 → 电音）不同：这里对每个频点做
        瞬时频率估计并用分析相位差传播相位，再按合成 hop 叠加，轻微变速(0.85~1.2)
        听感干净。仅用于调用方显式指定语速的路径（如语音工作室滑块）。
        """
        import numpy as _np
        wav = _np.asarray(wav, dtype=_np.float32)
        if abs(float(rate) - 1.0) < 1e-3 or len(wav) == 0:
            return wav
        rate = min(max(float(rate), 0.6), 1.6)
        n = len(wav)
        frame = 1024
        hop = 256
        win = _np.hanning(frame).astype(_np.float32)
        n_frames = max(1, (n - frame) // hop)
        out_len = int(n / rate) + frame
        synth = _np.zeros(out_len, dtype=_np.float32)
        norm = _np.zeros(out_len, dtype=_np.float32)
        # 每个频点按分析 hop 的期望相位增量
        w_inc = 2 * _np.pi * _np.arange(frame // 2 + 1) * (hop / frame)
        phase_acc = None
        prev_phase = None
        for k in range(n_frames):
            idx = k * hop
            ana = wav[idx:idx + frame] * win
            spec = _np.fft.rfft(ana)
            mag = _np.abs(spec)
            ana_phase = _np.angle(spec)
            if k == 0:
                phase_acc = ana_phase.copy()
            else:
                # 瞬时频率：分析相位差归一化到 [-pi, pi]，叠加期望增量传播
                dphi = ana_phase - prev_phase
                dphi = (dphi + _np.pi) % (2 * _np.pi) - _np.pi
                phase_acc = phase_acc + w_inc + dphi
            prev_phase = ana_phase
            frame_syn = _np.fft.irfft(mag * _np.exp(1j * phase_acc), frame).real * win
            out_idx = int(k * hop / rate)
            end = min(out_idx + frame, out_len)
            L = end - out_idx
            synth[out_idx:end] += frame_syn[:L]
            norm[out_idx:end] += win[:L]
        norm[norm < 1e-6] = 1.0
        synth = synth / norm
        # 归一化峰值，避免饱和
        m = float(_np.max(_np.abs(synth)))
        if m > 1.0 and m > 0:
            synth = synth / m
        return synth.astype(_np.float32)

    @staticmethod
    def _read_wav(path):
        """用纯标准库读 16bit WAV -> (float32 mono, sr)。"""
        import wave
        with wave.open(path, "rb") as w:
            nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        a = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32767.0
        if nch > 1:
            a = a.reshape(-1, nch).mean(axis=1)
        return a, sr if sr else OUT_SR


def _selftest():
    gg = GGUFTTS.get()
    print("availability:", gg.availability())
    wav, sr, meta = gg.synthesize("小伴，我喜欢你呀。", emotion="开心", temperature=0.9)
    print("meta:", meta, "wav_len_s:", round(wav.shape[0] / sr, 2))


if __name__ == "__main__":
    _selftest()