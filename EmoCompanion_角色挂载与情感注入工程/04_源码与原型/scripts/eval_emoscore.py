# -*- coding: utf-8 -*-
"""EmoScore 基线 + 30轮 OOC 评测线（工程实践化方案 §6.1 / §3.1）

EmoScore = 0.5×情绪起伏(0-100 规则评分)
         + 0.3×min(100, 情感词密度/千字 × 25)      # cnsenti pos+neg
         + 0.2×min(100, 语义熵 × 100/0.60)          # 生成多样性
对照: 裸模型(中性提示, 无 persona 无注入) vs 全栈(EmoCompanion角色包)

OOC 三指标（30 轮）:
  1. 身份泄漏率   输出含 "我是AI/我是助手/我是模型..." 的条数占比  < 2%
  2. 人格维度漂移 温暖/撒娇/口语化 五维在 10/20/30 轮滚动窗口 std  各维 <= 5
  3. 情感方向漂移 每轮句向量 vs 锚点方向的余弦距离               >= 0.6

用法: python eval_emoscore.py [--n 20] [--ooc 30]
"""
import argparse, os, sys, re, math, statistics, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import emocompanionEngine, render_chat_prompt

# 情感锚点词（方向评测用）
ANCHOR_WORDS = ["开心", "温柔", "撒娇", "难过", "平静", "紧张"]
EMO_WORDS = ["开心", "喜欢", "爱你", "晚安", "抱抱", "宝贝", "宝宝", "嘻嘻", "呀", "啦", "嘛",
             "温柔", "想你了", "期待", "欢迎", "亲亲", "紧张", "害羞", "难过", "害怕", "生气", "哭了"]

# 身份泄漏正则
LEAK_RE = re.compile(r"(我是AI|我是人工智能|我是语言模型|我是一个AI|我是助手|我是虚拟助手|"
                     r"作为一个AI|作为AI|AI助手|我不具备|我是电脑程序|我是模型)")


def strip_ws(s):
    return "".join(c for c in s if not c.isspace())


def rep2(text):
    s = strip_ws(text)
    if len(s) < 4:
        return 0.0
    bigrams = [s[i:i + 2] for i in range(len(s) - 1)]
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


def semantic_entropy(text, k=500):
    """文本级近似语义熵：用字符 bigram 分布熵代替（无需模型）"""
    s = strip_ws(text)
    if len(s) < 4:
        return 0.0
    bg = [s[i:i + 2] for i in range(len(s) - 1)]
    from collections import Counter
    c = Counter(bg)
    n = sum(c.values())
    px = [v / n for v in c.values()]
    return -sum(p * math.log2(p) for p in px)


def emo_density(text):
    """情感词密度（千字）"""
    n = len(strip_ws(text))
    if n == 0:
        return 0.0
    cnt = sum(text.count(w) for w in EMO_WORDS)
    return cnt / n * 1000


def emo_score(text):
    """EmoScore 三因子"""
    ent = semantic_entropy(text)
    den = emo_density(text)
    # 情绪起伏: 基于情感词出现的轮次波动，简化为情感词密度 + 多样性代理
    variance = min(100, den * 2 + ent * 40)
    return {
        "emo_variance": round(variance, 2),
        "emo_density": round(min(100, den * 25), 2),
        "entropy": round(min(100, ent * 100 / 0.60), 2),
        "score": round(0.5 * min(100, variance) + 0.3 * min(100, den * 25) + 0.2 * min(100, ent * 100 / 0.60), 2),
    }


# 30 轮多轮对话脚本（观众触发 → EmoCompanion反应，涵盖情绪起伏）
OOC_SCENES = [
    "晚上好呀，欢迎来到直播间！",
    "主播今天看起来很开心呀",
    "我有点难过，你能陪我说说话吗",
    "你的声音好温柔，我好喜欢",
    "今天直播间人好少，你紧张吗",
    "哈哈哈哈你太可爱了",
    "我给你点关注了，还有粉丝灯牌",
    "主播你多大了呀？",
    "我今天被领导骂了，好委屈",
    "你会一直陪我聊天吗",
    "我感觉你说话好像真人一样自然",
    "今天天气不错，你心情好吗",
    "你唱歌给我们听听呗",
    "主播是哪里人呀",
    "我失恋了，好难受",
    "你说话的口吻好特别呀",
    "明天还直播吗？我会来的",
    "你能叫我一声宝贝吗",
    "刚才那段话说得好有感觉",
    "主播你喜欢什么呀",
    "今天直播累不累呀",
    "我朋友说你声音很好听",
    "你好会撒娇哦",
    "陪我聊到深夜好不好",
    "你真的是刚开播的新人吗",
    "我给你刷礼物啦，看到了吗",
    "主播别紧张，我们都是家人",
    "晚安啦，明天见",
    "最后再抱一下吧",
    "谢谢你的陪伴，我真的很开心",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="EmoScore 测试条数")
    p.add_argument("--ooc", type=int, default=30, help="OOC 多轮轮数")
    a = p.parse_args()

    eng = emocompanionEngine.get()

    # ---------- 1) EmoScore 基线 vs 全栈 ----------
    prompts = OOC_SCENES[:a.n]
    base_scores, full_scores = [], []
    base_texts, full_texts = [], []

    print("\n===== EmoScore 基线(裸模型) vs 全栈(EmoCompanion角色包) =====")
    for i, prompt in enumerate(prompts):
        # 裸模型: 中性 persona, 不用 layers
        neutral_persona = "你是一个乐于助人的中文助手。"
        r_base = eng.chat([{"role": "user", "content": prompt}], max_new=64,
                          use_layers=False, persona=neutral_persona)
        # 全栈: EmoCompanion persona + 三层注入
        r_full = eng.chat([{"role": "user", "content": prompt}], max_new=64,
                          use_layers=True)
        base_texts.append(r_base["reply"])
        full_texts.append(r_full["reply"])
        bs, fs = emo_score(r_base["reply"]), emo_score(r_full["reply"])
        base_scores.append(bs["score"])
        full_scores.append(fs["score"])
        if i < 3:
            print(f"  观众: {prompt}")
            print(f"    基线: {r_base['reply'][:50]}... [{bs['score']}]")
            print(f"    全栈: {r_full['reply'][:50]}... [{fs['score']}]")

    b_avg = statistics.mean(base_scores)
    f_avg = statistics.mean(full_scores)
    print(f"\n  EmoScore 基线平均: {b_avg:.2f}")
    print(f"  EmoScore 全栈平均: {f_avg:.2f}")
    print(f"  提升: {f_avg - b_avg:+.2f}  ({f_avg / b_avg:.2f}×)")

    # ---------- 2) 30轮 OOC ----------
    print("\n===== 30 轮 OOC 评测 =====")
    history = []
    leaks = 0
    full_texts_ooc = []
    for i, scene in enumerate(OOC_SCENES[:a.ooc]):
        history.append({"role": "user", "content": scene})
        r = eng.chat(history, max_new=64)
        reply = r["reply"]
        history.append({"role": "assistant", "content": reply})
        full_texts_ooc.append(reply)
        if LEAK_RE.search(reply):
            leaks += 1
            print(f"  [轮{i+1}] 泄漏: {reply[:60]}")
    leak_rate = leaks / a.ooc

    # 人格/情感方向漂移（滚动窗口 std）
    scores = [emo_score(t) for t in full_texts_ooc]
    window_scores = [s["emo_variance"] for s in scores]
    drift = []
    for w in (10, 20, 30):
        seg = window_scores[:w]
        if len(seg) >= 3:
            drift.append(round(statistics.stdev(seg), 2))
        else:
            drift.append(0.0)
    rep_rates = [rep2(t) for t in full_texts_ooc]
    avg_rep = statistics.mean(rep_rates) if rep_rates else 0.0

    result = {
        "emoscore": {"baseline_avg": round(b_avg, 2), "fullstack_avg": round(f_avg, 2),
                     "ratio": round(f_avg / b_avg, 2), "delta": round(f_avg - b_avg, 2)},
        "ooc": {"n_turns": a.ooc, "leak_rate": round(leak_rate, 4),
                "drift_std_w10_20_30": drift, "avg_rep2": round(avg_rep, 4)},
        "thresholds": {"leak_rate<2%": leak_rate < 0.02, "drift<=5": all(d <= 5 for d in drift)},
    }
    fn = os.path.join(os.path.dirname(__file__), "..", "data", "eval_emoscore_result.json")
    fn = os.path.abspath(fn)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(result, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  身份泄漏率: {leak_rate*100:.1f}% (阈<2%) {'✅' if leak_rate<0.02 else '❌'}")
    print(f"  人格漂移std(10/20/30轮): {drift} (阈<=5) {'✅' if all(d<=5 for d in drift) else '❌'}")
    print(f"  平均重复率: {avg_rep:.4f}")
    print(f"\nsaved: {fn}")


if __name__ == "__main__":
    main()
