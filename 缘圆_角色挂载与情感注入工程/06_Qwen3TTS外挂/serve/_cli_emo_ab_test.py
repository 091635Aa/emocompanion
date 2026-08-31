# -*- coding: utf-8 -*-
"""情感路由 开/关 对比 + 强情感输入测试"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://127.0.0.1:8000"  # 直接打 04 引擎，便于精确控制 emotion/scale_emo


def chat(msgs, emotion=None, scale=1.0, max_new=200, persona=None):
    pay = {"messages": [{"role": "user", "content": msgs}], "max_new": max_new}
    if emotion:
        pay["emotion"] = emotion
    pay["scale_emo"] = scale
    if persona:
        pay["persona"] = persona
    req = urllib.request.Request(
        BASE + "/chat", data=json.dumps(pay).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


# 强情感输入：明确表达情绪的词
CASES = [
    ("我哭了一晚上，感觉天都塌了", "悲伤"),
    ("太开心了！！我中奖了哈哈哈哈哈", "开心"),
    ("你别理我了，我没事，真的没事", "悲伤"),
    ("家人们我今天好激动啊！！终于过了！", "激动"),
    ("好想抱抱你，能不能陪我一下", "撒娇"),
]

print("=" * 72)
print("A. 情感路由 开(scale=1) vs 关(scale=0) 对比 —— 相同输入「我今天好难过」")
print("=" * 72)
on = chat("我今天好难过，感觉撑不下去了", "悲伤", 1.0, 160)
off = chat("我今天好难过，感觉撑不下去了", "悲伤", 0.0, 160)
print(f"[路由开] {on['reply'][:120]}")
print(f"[路由关] {off['reply'][:120]}")

print("\n" + "=" * 72)
print("B. 强情感输入 逐情感路由测试")
print("=" * 72)
for text, emo in CASES:
    r = chat(text, emo, 1.0, 160)
    print(f"\n输入: {text}  (route={emo})")
    print(f"回复: {r['reply'][:150]}")
