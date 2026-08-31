# -*- coding: utf-8 -*-
"""步骤1：从6951条打标数据提取'缘圆真实口癖'正例词袋 + 构建AI空泛词抑制表
- 输入: 微调训练集.jsonl(6951条, transcript=缘圆真实口播)
- 输出: deai_bag.json {positive_bag: {词: 计数}, hollow_table: [短语], meta}
用法: python extract_deai_bag.py
"""
import os, json, re
from collections import Counter

HERE = os.path.dirname(__file__)
MICRO = os.path.join(os.path.dirname(HERE), "02_角色参数与数据", "微调数据", "微调训练集.jsonl")
OUT = os.path.join(HERE, "deai_bag.json")

# 候选口癖粒子(单字, 高频优先) —— 数据里缘圆真实语气
PARTICLES = ["吗", "呢", "呀", "啊", "嘛", "啦", "呗", "哦", "嗯", "哟", "哈", "噢", "诶", "喂", "啦~"]
# 候选口癖短词(整词/整串, 续接增强用)
PHRASES = ["可以吗", "点点关注", "加个灯牌", "家人们", "哎呀", "我去", "有感觉吗",
           "谢谢", "爱你", "想你了", "欢迎", "欢迎来到", "新人主播", "拜拜", "晚安",
           "嘻嘻", "嘿嘿", "哈哈哈哈", "喵", "宝贝", "宝宝", "亲亲", "抱抱", "么么"]

# AI空泛词/模板化表达抑制表(多字短语, 续接压制)
HOLLOW = [
    "太好了", "太棒了", "太厉害了", "太优秀了", "真棒", "很棒", "非常棒", "完美",
    "首先", "其次", "最后", "再次", "总之", "因此", "然而", "此外", "另外",
    "其实", "真的", "确实", "实际上", "其实呢", "可以说",
    "不是", "不仅", "更是", "重要的是", "关键是", "真正", "本质上",
    "我想说的是", "我觉得吧", "那么", "总之呢", "所以说", "也就是说",
    "好的呢", "明白了吗", "懂了吗", "你知道吧", "对不对呀",
    "让我们一起", "希望你能", "希望大家", "一定可以", "相信你",
    "感谢你", "非常感谢", "很荣幸", "倍感荣幸",
    "接下来", "下面", "现在让我们", "我们来",
    "同时", "我们需要", "我们应该", "我的理解是", "换句话说", "总而言之",
    "一般来说", "综上所述", "首先呢", "无论如何", "众所周知",
    "哈哈哈哈哈", "嗯嗯嗯", "啊啊啊", "呜呜呜",
]

def transcripts():
    n = 0
    with open(MICRO, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = (d.get("输出") or {}).get("transcript") or ""
            if t.strip():
                n += 1
                yield t
    return n

def main():
    tlist = list(transcripts())
    n = len(tlist)
    text = "".join(tlist)
    total_chars = len(text)

    # 1) 单字粒子计数(相对词频 = 出现次数/文本字数, 便于跨数据量比较)
    pc = Counter()
    for ch in PARTICLES:
        pc[ch] = text.count(ch)
    positive_bag = {}
    for ch, c in pc.items():
        if c >= 10:  # 至少出现10次才算稳定口癖
            positive_bag[ch] = {"count": c, "per10k": round(c / total_chars * 10000, 2)}

    # 2) 口癖短词计数
    phrase_counts = {}
    for p in PHRASES:
        c = text.count(p)
        if c >= 10:
            phrase_counts[p] = {"count": c, "per10k": round(c / total_chars * 10000, 2)}
    # 合并：单字粒子入 bag，多字短语入 bag(带续接标志)
    for p, info in phrase_counts.items():
        positive_bag[p] = info

    # 3) 空泛词在真实语料中的出现率(越低越说明'真人不用' → 抑制权重越高)
    #    数据驱动分级：count<100 强抑制(1.0)；100<=count<600 弱抑制(0.5)；>=600 真人也在用 → 剔除
    hollow_check = {}
    for h in HOLLOW:
        hollow_check[h] = text.count(h)
    hollow_weighted = []   # [(phrase, weight)]
    hollow_skip = []       # 因真实语料高频被剔除
    for h, c in hollow_check.items():
        if c >= 600:
            hollow_skip.append(h)
        elif c >= 100:
            hollow_weighted.append([h, 0.5])
        else:
            hollow_weighted.append([h, 1.0])

    out = {
        "meta": {"n_transcripts": n, "total_chars": total_chars, "source": os.path.basename(MICRO)},
        "positive_bag": dict(sorted(positive_bag.items(), key=lambda kv: -kv[1]["count"])),
        "hollow_weighted": hollow_weighted,
        "hollow_skip": hollow_skip,
        "hollow_in_real_corpus": hollow_check,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[data] {n} 条 transcript, 共 {total_chars} 字")
    print(f"[positive_bag] {len(positive_bag)} 项(单字粒子+口癖短词):")
    for k, v in sorted(positive_bag.items(), key=lambda kv: -kv[1]["count"])[:30]:
        print(f"    {k!r:>8}  {v['count']:>6} 次  {v['per10k']}/万")
    print(f"[hollow_weighted] {len(hollow_weighted)} 个有效抑制目标(强1.0/弱0.5):")
    for h, w in hollow_weighted:
        print(f"    {h!r:>10}  w={w}  真实语料{hollow_check[h]}次")
    print(f"[hollow_skip] {len(hollow_skip)} 个因真人高频剔除: {hollow_skip}")
    print(f"saved: {OUT}")

if __name__ == "__main__":
    main()
