# -*- coding: utf-8 -*-
"""用当前音色生成一段感情丰富的片段（情绪标签 + 自由指令，带自动降级）。"""
import os, requests, json, pathlib, base64, subprocess

API_KEY = os.environ["DASHSCOPE_API_KEY"]
WORK = pathlib.Path(__file__).parent
VOICE_ID = (WORK / "voice_id.txt").read_text(encoding="utf-8").strip()
URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

TEXT_TAGS = ("你怎么这么久都不理我？[gasp] 我还以为……以为你把我忘了呢。"
             "刚才一个人走夜路的时候，我真的好害怕，好想给你打个电话。"
             "现在看到你来了，[giggles] 心里一下子就踏实了。"
             "喂，笑什么？不许笑话我。走，我们回家，今晚我给你煮热汤，"
             "谁也不许再说分开的话。")
TEXT_PLAIN = ("你怎么这么久都不理我？我还以为……以为你把我忘了呢。"
              "刚才一个人走夜路的时候，我真的好害怕，好想给你打个电话。"
              "现在看到你来了，心里一下子就踏实了。"
              "喂，笑什么？不许笑话我。走，我们回家，今晚我给你煮热汤，"
              "谁也不许再说分开的话。")
INSTRUCTION = ("女声独白，情绪层层递进：开头是委屈和不安，声音微微发颤；"
               "说到害怕时呼吸加重；然后如释重负，声音放松带着笑意；"
               "最后转为俏皮撒娇，又带一点温柔。语速自然，感情真挚。")

def synth(text, instruction, out, tag=""):
    payload = {"model": "qwen-audio-3.0-tts-plus",
               "input": {"text": text, "voice": VOICE_ID,
                         "format": "wav", "sample_rate": 48000,
                         "instruction": instruction}}
    r = requests.post(URL, json=payload, headers=HEADERS, timeout=180)
    print(f"[{tag}] HTTP {r.status_code}")
    body = r.text[:1200]
    print(body)
    if r.status_code != 200:
        return False
    j = r.json()
    url = j.get("output", {}).get("audio", {}).get("url", "")
    if not url:
        return False
    data = requests.get(url, timeout=120).content
    out.write_bytes(data)
    return True

# 优先带情绪标签，失败则降级为纯指令
ok = synth(TEXT_TAGS, INSTRUCTION, WORK / "tts_emotion.wav", "带标签版")
if not ok:
    print("标签版失败，降级为纯指令版……")
    ok = synth(TEXT_PLAIN, INSTRUCTION, WORK / "tts_emotion.wav", "纯指令版")
if not ok:
    raise SystemExit("合成失败")

subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(WORK / "tts_emotion.wav"),
                "-b:a", "192k", str(WORK / "tts_emotion.mp3")], check=True)
print("SAVED", WORK / "tts_emotion.mp3")
