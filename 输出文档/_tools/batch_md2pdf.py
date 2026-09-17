# -*- coding: utf-8 -*-
"""批量 Markdown → PDF 转换（复用 md2pdf 引擎，Edge 无头打印学术风格）。

用法: python batch_md2pdf.py
输出: d:\AI情感\输出文档\PDF转化\ 下按同名目录结构存放。
"""
import sys
from pathlib import Path

SRC = Path(r"d:\AI情感\输出文档")
DST = Path(r"d:\AI情感\输出文档\PDF转化")
MD2PDF = Path(r"d:\AI情感\091635Aa_商业化推进\工具\md2pdf.py")
STYLE = "academic"          # academic: 学术灰，打印友好
COVER = False               # 不额外加封面（专利/论文本身有标题）
SKIP_DIRS = {"PDF转化", "_tools"}     # 避免递归转换已有输出/工具

def main():
    files = sorted(p for p in SRC.rglob("*.md") if not any(d in p.parts for d in SKIP_DIRS))
    ok, fail = [], []
    for md in files:
        rel = md.relative_to(SRC)
        out = DST / rel.with_suffix(".pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, str(MD2PDF), str(md), str(out), "--style", STYLE]
        if COVER:
            args.append("--cover")
        import subprocess
        print(f"[转] {rel}")
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if out.exists() and out.stat().st_size > 0:
            ok.append((rel, out.stat().st_size))
        else:
            fail.append((rel, r.stderr.strip()[:300]))
    print("\n==== 汇总 ====")
    print(f"成功 {len(ok)} / 失败 {len(fail)}")
    for rel, size in ok:
        print(f"  OK  {rel}  ({size//1024} KB)")
    for rel, err in fail:
        print(f"  FAIL {rel}  {err}")
    return 1 if fail else 0

if __name__ == "__main__":
    sys.exit(main())