# -*- coding: utf-8 -*-
"""缘圆 主播化长线对话 CLI 测试
模拟直播间观众 → 主播 多轮互动（同一会话，保持上下文连续），验证：
  - 人设一致（缘圆 温柔/撒娇/口语化/主播话术）
  - 情感路由生效（emo_detect → 04 引擎 emotion 偏置）
  - 长度自适应（安慰→小作文，日常→几句）
  - 句句有回应
"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://127.0.0.1:8071"


def post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


# 直播间预设提示词（模拟一位观众从进直播间到离开的完整长线互动）
SCENES = [
    ("观众进直播间", "主播晚上好呀！第一次来看你~"),
    ("观众夸主播", "哇你声音好好听，人也好可爱，我要待着不走了！"),
    ("观众求关注", "那个…关注和灯牌是干嘛的呀？"),
    ("观众低落", "我今天工作被骂了，好难过啊……"),
    ("观众被安慰后", "谢谢你安慰我，感觉好多了，你真好~"),
    ("观众撩主播", "你这么可爱，有没有男朋友呀？"),
    ("观众起哄", "主播主播，给我们唱首歌呗！"),
    ("观众夸变化", "感觉你今天心情特别好呀，是不是遇到好事了？"),
    ("观众要离开", "我要去吃饭啦，一会儿再来看你哦~"),
    ("观众回来", "我回来啦！想我了吗？"),
    ("观众深夜关怀", "夜深了，主播还不下播吗？要注意休息呀"),
]

s = post("/api/sessions", {"title": "主播化长线测试"})
sid = s["id"]
print("=" * 70)
print(f"会话: {sid}  |  主播: 缘圆（温柔撒娇主播 persona）")
print("=" * 70)

for label, user_msg in SCENES:
    r = post(f"/api/sessions/{sid}/talk", {
        "content": user_msg, "want_tts": False, "role": "缘圆"})
    emo = r.get("emotion", "?")
    src = (r.get("emotion_info") or {}).get("source", "?")
    reply = r.get("reply", "")
    print(f"\n【{label}】")
    print(f"  观众: {user_msg}")
    print(f"  缘圆: {reply}")
    print(f"  情感: {emo}({src}) | 长度: {len(reply)} 字")
