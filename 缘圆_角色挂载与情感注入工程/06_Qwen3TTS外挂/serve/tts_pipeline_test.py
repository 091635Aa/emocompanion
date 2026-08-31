# -*- coding: utf-8 -*-
"""缘圆 端到端 TTS 测试：日常直播对话片段 → 自动情感 → 合成 → 延迟测量

场景：刘不说话，缘圆面向直播间说一段日常开场白。
流程：文本(提问后回复) ─► 情感自动识别 ─► voice_clone(角色embedding) ─► wav

运行(需 torch/qwen_tts/peft/soundfile，系统 python)：
  python tts_pipeline_test.py [--text "自定文本"] [--emotion auto] [--warm 0|1]
输出：缘圆日常直播片段.wav + 分阶段延迟打印
"""
import argparse
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OSD = os.path.join(HERE, "out")
os.makedirs(OSD, exist_ok=True)

# 场景文本（自己编写）：更长的"对话流"文案，测试长效一致性与稳定性
DEFAULT_CLIP = ("家人们晚上好呀，可算把大家盼来啦！今天下播前想跟大家多聊两句——你们老问我为什么总夸人，"
                "其实我也是从观众一步步做起来的，懂被夸一句有多暖。这不，昨天有个妹妹私信我说心情不好，"
                "我就陪她聊了半宿，今儿整个人都是暖的。所以啊，你们开心我就开心，直播间有你们在，比啥都值！")


def write_wav(path, arr, sr):
    import wave
    import numpy as np
    arr = np.asarray(arr, dtype="float32")
    pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype("int16")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_CLIP)
    ap.add_argument("--emotion", default="auto", help="auto=自动识别，或显式标签")
    ap.add_argument("--warm", type=int, default=1, help="首先生成一条用于预热")
    ap.add_argument("--compile", type=int, default=0, help="torch.compile 优化 talker(减延迟)")
    ap.add_argument("--adapter", default="emotion", help="voice(音色)/emotion(情感律动,默认)")
    ap.add_argument("--vary", type=float, default=0.6, help="随机语气强度 0~1")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="缘圆日常直播片段.wav")
    ap.add_argument("--flow", type=int, default=0, help="分句流式合成(每句独立情感)")
    ap.add_argument("--emolist", default="开心,俏皮,平静,开心", help="--flow 用：逐句显式情感列表")
    a = ap.parse_args()

    from emo_detect import detect_emotion, EMOTIONS
    from tts_engine import get_engine

    ts = {}

    # 0) 情感自动识别（对话→TTS 前高精度插入）
    t0 = time.perf_counter()
    emo = detect_emotion(a.text, text_chat_fn=None) if a.emotion == "auto" \
        else type("E", (), {"label": a.emotion, "confidence": 1.0, "source": "manual"})()
    ts["emotion_detect_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    # 1) 加载（冷启动；一次性成本）
    eng = get_engine()
    t0 = time.perf_counter()
    eng.load()
    ts["model_load_s"] = round(time.perf_counter() - t0, 3)

    # 1.5) torch.compile（可选，减稳态延迟）
    if a.compile:
        import torch
        t0 = time.perf_counter()
        try:
            eng._model.model.talker = torch.compile(
                eng._model.model.talker, mode="reduce-overhead")
            ts["compile"] = "尝试中"
        except Exception as e:
            ts["compile"] = f"跳过:{type(e).__name__}"
        ts["compile_setup_s"] = round(time.perf_counter() - t0, 3)

    # 2) 预热（可选）：固化 CUDA kernel，使实测反映稳态延迟
    if a.warm:
        t0 = time.perf_counter()
        eng.synthesize("好的", "平静")
        ts["warm_s"] = round(time.perf_counter() - t0, 3)

    # 3) 合成：普通 or 分句流式
    if a.flow:
        emotions = [x.strip() for x in a.emolist.split(",") if x.strip()]
        emo_lbl = "开心" if not a.flow else "/".join(emotions)
        wav, sr, meta = eng.synthesize_flow(a.text, emotions=emotions,
                                            adapter=a.adapter, tone_variation=a.vary, seed=a.seed)
        ts["synthesize_s"] = meta["seconds"]
        ts["rtf"] = meta["rtf"]
        ts["audio_seconds"] = meta["audio_seconds"]
        ts["n_segments"] = meta["n_segments"]
    else:
        wav, sr, meta = eng.synthesize(a.text, emo.label,
                                       adapter=a.adapter, tone_variation=a.vary, seed=a.seed)
        ts["synthesize_s"] = meta["seconds"]
        ts["rtf"] = meta["rtf"]
        ts["audio_seconds"] = meta["audio_seconds"]
        emo_lbl = emo.label

    # 4) 写盘 + 编码延迟
    out = os.path.join(OSD, a.out)
    t0 = time.perf_counter()
    write_wav(out, wav, sr)
    ts["wav_encode_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    # 5) 一致性分析（不依赖听感）
    import numpy as np
    arr = np.asarray(wav, dtype="float32")
    win = int(sr * 0.3)                      # 300ms 窗
    n = len(arr) // win
    frames = arr[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    rms = rms[rms > 1e-4]                    # 去静音帧，看有效发声段
    e_std = round(float(rms.std()), 4) if rms.size else 0.0
    e_mean = round(float(rms.mean()), 4) if rms.size else 0.0
    live = round(float(rms.std() / rms.mean()), 3) if rms.size and e_mean > 0 else 0.0
    # 尾端能量 vs 中段：检测是否异常截断/掉链子
    tail = round(float(rms[-max(3, rms.size // 8):].mean()), 4) if rms.size >= 6 else 0.0
    mid = round(float(rms[rms.size // 3: 2 * rms.size // 3].mean()), 4) if rms.size >= 6 else 0.0
    cons = {
        "win_rms_mean": e_mean, "win_rms_std": e_std,
        "liveliness_std_mean_ratio": live,
        "tail_energy": tail, "mid_energy": mid,
        "text_chars": len(a.text), "dur_per_char_s": round(meta["audio_seconds"] / max(len(a.text), 1), 4),
    }

    print("=" * 60)
    print(f"文本     : {a.text}")
    print(f"情感     : {emo_lbl}  来源={getattr(emo,'source','—')}  (模式={'分句流式' if a.flow else '整段'})")
    print(f"策略     : {meta['strategy']}  adapter={meta.get('adapter')}  adapter_attached={meta.get('adapter_attached',False)}")
    if a.flow:
        for m in meta.get("segments", []):
            print(f"  「{m['emotion']}」 {m['segment'][:26]}…  {m['audio_seconds']}s rtf={m.get('rtf')} gen={m.get('gen')}")
    else:
        print(f"随机语气 : {meta.get('gen')}")
    print("-" * 60)
    print("分阶段延迟(实测):")
    for k, v in ts.items():
        print(f"  {k:20s}: {v}")
    print("-" * 60)
    print(f"总(TTS链路,不含加载) ≈ {round(ts['synthesize_s'] + ts['wav_encode_ms']/1000, 3)} s   音频 {ts['audio_seconds']}s   RTF {ts['rtf']}")
    print("-" * 60)
    print("一致性分析(300ms 窗)：")
    for k, v in cons.items():
        print(f"  {k:24s}: {v}")
    print(f"  解读: liveliness>0.25 有起伏非机械; tail≈mid 未截断; dur/char≈0.12-0.25 语速正常")
    print(f"输出: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()