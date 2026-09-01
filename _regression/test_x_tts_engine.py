# -*- coding: utf-8 -*-
"""R20 音频族回归：EmoCompanion TTS 引擎封装（tts_engine）纯逻辑层

直接 import 真实模块（顶层仅 os/random/re/time/dataclass/typing）：
  1. EMOTION_VOCAB 结构：≥6 情感、键唯一、`[emotion]X[/emotion]` 自洽、兜底在词表。
  2. _compose_text：干净文本 strip，绝不塞情感标签（设计契约）。
  3. make_schedule：按句切分 + 情感分配（list 全给 / dict 局部覆盖），
     list 全指定时不再触发懒加载 emo_detect。
  4. TTSUnavailable 契约 + _check_paths 缺件检测（含 角色音色 embedding）。
  5. SR/多路 adapter/情感清单 元信息（list_adapter_names）。
运行：python3 /workspace/_regression/test_x_tts_engine.py
"""
import os, sys, tempfile, importlib.util

根 = "/workspace/_regression"
PASS = 0
FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")

mod_path = os.path.join(
    根, "..", "EmoCompanion_角色挂载与情感注入工程",
    "06_Qwen3TTS外挂", "serve", "tts_engine.py")
serve_dir = os.path.dirname(mod_path)
if serve_dir not in sys.path:  # 让 tts_engine 内 `from emo_detect import detect_emotion` 可解析
    sys.path.insert(0, serve_dir)
spec = importlib.util.spec_from_file_location("tts_engine_test", mod_path)
engmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engmod)

VOCAB = engmod.EMOTION_VOCAB
FB = engmod.EMOTION_FALLBACK

print("== EMOTION_VOCAB 结构 + 兜底 ==")
check("词表≥6 情感", len(VOCAB) >= 6, f"n={len(VOCAB)}")
check("键唯一", len(set(VOCAB)) == len(VOCAB))
自洽 = all(v == f"[emotion]{k}[/emotion]" for k, v in VOCAB.items())
check("情感围栏与键自洽", 自洽)
check("兜底在词表内", FB in VOCAB, f"FB={FB}")

print("== _compose_text（干净文本，不塞标签）==")
e = engmod.TTSEngine.__new__(engmod.TTSEngine)
check("纯文本 strip", e._compose_text("  哥哥你回来啦  ", "开心") == "哥哥你回来啦")
check("空串处理", e._compose_text("", "开心") == "")
check("不含情感标签", "[emotion]" not in e._compose_text("Hello", "悲伤"))

print("== make_schedule（分句 + 情感分配）==")
def _mk():
    return engmod.TTSEngine.__new__(engmod.TTSEngine)
# list 全给：每句逐一分配，不再触发 emo_detect 懒加载
sched = _mk().make_schedule("你好。世界！", ["开心", "悲伤"])
check("按句切分 2 段", [s for s, _ in sched] == ["你好", "世界"], f"got={sched}")
check("list 逐句情感", [emc for _, emc in sched] == ["开心", "悲伤"])

# 无标点单句
sched2 = _mk().make_schedule("没有标点一句", ["平静"])
check("无标点单句成段", [s for s, _ in sched2] == ["没有标点一句"])

# dict 局部覆盖：覆盖句0，其余自动（此处后续句将触发检测，返回非 None 即可）
sched3 = _mk().make_schedule("第一句。第二句。第三句", {0: "兴奋"})
check("dict 局部覆盖句0=兴奋", sched3[0][1] == "兴奋")
check("dict 覆盖最多3段", len(sched3) == 3)

# list 短于句数：缺失段回落自动检测（不炸）
sched4 = _mk().make_schedule("甲。乙。丙", ["开心"])
check("list 短于句数不炸", len(sched4) == 3 and sched4[0][1] == "开心")

print("== TTSUnavailable + _check_paths ==")
check("TTSUnavailable 是 RuntimeError 子类", issubclass(engmod.TTSUnavailable, RuntimeError))
with tempfile.TemporaryDirectory() as td:
    ok = engmod.TTSEngine(base_model=td, adapters={"voice": td, "emotion": td},
                          speaker_emb_path=os.path.join(td, "emb.pt"))
    open(ok.speaker_emb_path, "wb").close()
    ok._check_paths()  # 全部就绪不抛
    check("就绪 _check_paths 不抛", True)
    bad = engmod.TTSEngine(base_model=os.path.join(td, "nobase"),
                           adapters={"voice": td, "emotion": os.path.join(td, "noemo")},
                           speaker_emb_path=os.path.join(td, "noemb.pt"))
    try:
        bad._check_paths()
        check("缺件抛 TTSUnavailable", False, "未抛")
    except engmod.TTSUnavailable as ex:
        msg = str(ex)
        check("缺件抛 TTSUnavailable", True)
        check("错误含全部缺件名",
              "Base 模型" in msg and "外挂包[emotion]" in msg and "角色音色 embedding" in msg,
              f"msg={msg!r}")

print("== list_adapter_names 元信息 ==")
info = engmod.list_adapter_names()
check("SR=24000", info["samplerate"] == 24000)
check("adapter 含 voice+emotion", set(info["adapters"]) == {"voice", "emotion"})
check("emotions 与词表一致", set(info["emotions"]) == set(VOCAB))

print("== R21 emo_detect 词典路径 + make_schedule 自动识别 ==")
from emo_detect import _lexicon_detect, detect_emotion, DEFAULT
check("高兴文案→开心", _lexicon_detect("今天真的好开心啊").label == "开心")
r  = _lexicon_detect("我太难过想哭呜呜")
check("难过文案→悲伤", r.label == "悲伤", f"got={r.label} conf={r.confidence}")
check("无关键词→默认平静", _lexicon_detect("普通话题").label == DEFAULT)
check("和平静文案→默认(不在词典)", _lexicon_detect("今天状态很平静").label == DEFAULT)
# detect_emotion(text_chat_fn=None) → 走词典兜底，不炸
d = detect_emotion("他好悲伤，快哭出来了", None)
check("detect_emotion 词典兜底不炸", d.label == "悲伤", f"got={d.label}")
# make_schedule 自动识别路径（emotions=None → 逐句 detect，词典兜底）
sch_auto = engmod.TTSEngine.__new__(engmod.TTSEngine).make_schedule("我好难过。", None)
check("make_schedule 自动识别悲伤", sch_auto[0][1] == "悲伤", f"got={sch_auto}")

print("== R22 _merge_refs_to_single 多段 ICL 合并 ==")
import wave as _wave
import numpy as _np

def _写wav(path, sr, samples):
    with _wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(_np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes())

SR0 = engmod.SR
with tempfile.TemporaryDirectory() as td2:
    import os as _os
    a = _os.path.join(td2, "a.wav")
    b = _os.path.join(td2, "b.wav")
    _写wav(a, SR0, _np.ones(SR0))
    _写wav(b, SR0, _np.ones(SR0))
    eng = engmod.TTSEngine.__new__(engmod.TTSEngine)
    refs = [{"audio": a, "text": "[开心]啊", "emotion": "开心"},
            {"audio": b, "text": "[悲伤]呜", "emotion": "悲伤"}]
    m = eng._merge_refs_to_single(refs, gap_s=0.1)
    arr, sr = m["audio"]
    exp_len = SR0 + int(SR0 * 0.1) + SR0   # 两段 1s + 一个 gap 0.1s
    check("合并长度=两段+gap", arr.shape[0] == exp_len, f"got={arr.shape[0]} exp={exp_len}")
    check("target SR=24000", sr == SR0)
    check("多情感文本拼装", m["text"] == "[开心]啊 [悲伤]呜", f"got={m['text']!r}")
    check("emotions 透传", m["emotions"] == ["开心", "悲伤"])
    # 单段：无 gap（只加一段）
    m1 = eng._merge_refs_to_single([refs[0]], gap_s=0.1)
    check("单段长度=该段, 无 gap", m1["audio"][0].shape[0] == SR0, f"got={m1['audio'][0].shape[0]}")
    check("单段文本=该段文本", m1["text"] == "[开心]啊")

print(f"\n== 结果: {PASS} 通过 / {FAIL} 失败 ==")
sys.exit(1 if FAIL else 0)