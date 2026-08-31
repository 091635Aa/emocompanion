# -*- coding: utf-8 -*-
"""私有命名脱敏 v2：生成专利公有申请版（更精细）。

长词优先替换；代码类名替换为合法标识符；角色名替换为通用词。
仅改命名，不改技术内容。
"""
from pathlib import Path

SRC = Path(r"d:\AI情感\输出文档\专利申请")
DST = Path(r"d:\AI情感\输出文档\专利申请_公有脱敏版")

REPLACEMENTS = [
    # —— 代码/类名（替换为合法标识符）——
    ("class 情感注入引擎（EIE）:", "class EmotionEngine:"),
    ("emocompanionEngine", "EmotionEngine"),
    ("class EmotionEngine（EIE）:", "class EmotionEngine:"),
    # —— 引擎/系统名 ——
    ("EmoCompanion引擎", "情感注入引擎"),
    ('"EmoCompanion"系统', "本系统"),
    ("EmoCompanion系统", "本系统"),
    ('"EmoCompanion"', "本系统"),
    ("EmoCompanion", "本系统"),
    # —— 角色名通用化 ——
    ('character_name = "本系统"', 'character_name = "情感伙伴"'),
    ('character_name = "EmoCompanion"', 'character_name = "情感伙伴"'),
    ('character_name="EmoCompanion"', 'character_name="情感伙伴"'),
    # —— 私有方法名通用化 ——
    ("Ultra Fusion Dynamics", "动态超融合方法"),
    ("Emotional Director Dispatch", "情感导演调度方法"),
    ("emocompanion", "EmotionEngine"),
]

def deidentify(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text

def main():
    DST.mkdir(parents=True, exist_ok=True)
    total = 0
    for md in sorted(SRC.glob("*.md")):
        raw = md.read_text(encoding="utf-8")
        new = deidentify(raw)
        hits = sum(raw.count(old) for old, _ in REPLACEMENTS)
        total += hits
        (DST / md.name).write_text(new, encoding="utf-8")
        print(f"{md.name}: 替换 {hits} 处")
    # 残留检查
    print("\n残留私有词检查（应为空）：")
    for md in sorted(DST.glob("*.md")):
        t = md.read_text(encoding="utf-8")
        for w in ("EmoCompanion", "emocompanion", "emocompanion"):
            if w in t:
                print(f"  !! {md.name}: 仍含 '{w}'  {t.count(w)} 处")
        if md.name == "P6_情感导演调度专利.md":
            # 抽查类名合法性
            for line in t.splitlines():
                if "class " in line and "（" in line:
                    print(f"  !! {md.name}: 非法类名 '{line.strip()[:60]}'")
    print(f"\n共替换 {total} 处，输出: {DST}")

if __name__ == "__main__":
    main()