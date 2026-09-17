# -*- coding: utf-8 -*-
"""
091635Aa 商业化推进系统 · 排版输出插件 md2pdf
Markdown → PDF（Edge 无头打印），多风格排版，80g A4 打印友好。

用法:
  python md2pdf.py <输入.md> <输出.pdf> [--style business|academic|print] [--cover] [--title 标题] [--subtitle 副标题] [--footer 页脚]

风格:
  business  商务蓝（默认）：适合商业提案，蓝色标题 + 浅蓝表头
  academic  学术灰：适合技术报告/论文附录，黑字白底素净
  print     极简省墨：纯黑白、无背景色块，最省墨水、80g 纸最友好
"""
import sys, time, argparse, subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import date

import markdown

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# ── 三种风格 CSS ──────────────────────────────────────────────
CSS_BASE = """
@page {{ size: A4; margin: 1.8cm 2cm 1.8cm 2cm;
  @bottom-center {{ content: "{footer} | 第 " counter(page) " / " counter(pages) " 页";
                    font-size: 8pt; color: #666; }} }}
html, body {{ font-family: "Microsoft YaHei", "SimSun", sans-serif; font-size: 10.5pt;
             line-height: 1.7; color: {text}; margin: 0; padding: 0; }}
h1 {{ font-size: 16pt; text-align: center; margin: 0 0 8pt 0; page-break-after: avoid; }}
h2 {{ font-size: 13pt; {h2} border-bottom: 1pt solid {h2line}; padding-bottom: 2pt;
     margin-top: 16pt; page-break-after: avoid; }}
h3 {{ font-size: 11.5pt; margin-top: 12pt; page-break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }}
th {{ {th} border: 0.5pt solid {line}; padding: 3pt 5pt; text-align: center; }}
td {{ border: 0.5pt solid {line2}; padding: 3pt 5pt; }}
tr {{ page-break-inside: avoid; }}
blockquote {{ color: #555; border-left: 3pt solid {line}; padding-left: 8pt; margin: 6pt 0; }}
code {{ font-family: Consolas, monospace; background: #f4f4f4; font-size: 8.5pt; padding: 0 2pt; }}
pre {{ background: #f4f4f4; border: 0.5pt solid #ddd; padding: 6pt; font-size: 8.5pt;
      white-space: pre-wrap; page-break-inside: avoid; }}
hr {{ border: 0; border-top: 1pt solid #ccc; margin: 12pt 0; }}
p {{ margin: 6pt 0; }}
ul, ol {{ margin: 6pt 0; padding-left: 22pt; }}
li {{ margin: 2pt 0; }}
.cover {{ text-align: center; padding-top: 6cm; page-break-after: always; }}
.cover h1 {{ font-size: 24pt; margin-bottom: 12pt; }}
.cover .subtitle {{ font-size: 14pt; color: {sub}; margin-bottom: 36pt; }}
.cover .meta {{ font-size: 10.5pt; color: #666; }}
"""

STYLES = {
    "business": CSS_BASE.format(
        text="#222222", h2="color: #1a3c6e;", h2line="#c8d4e4",
        th="background: #eef3f9;", line="#9db2c8", line2="#b7c6d6",
        sub="#1a3c6e", footer="091635Aa 商业化推进系统",
    ),
    "academic": CSS_BASE.format(
        text="#111111", h2="color: #111111;", h2line="#999999",
        th="background: #f5f5f5;", line="#888888", line2="#aaaaaa",
        sub="#333333", footer="Semantic Echo · 技术报告",
    ),
    "print": CSS_BASE.format(
        text="#000000", h2="color: #000000;", h2line="#000000",
        th="background: #ffffff;", line="#000000", line2="#555555",
        sub="#000000", footer="Semantic Echo",
    ),
}


def 构建封面(标题, 副标题) -> str:
    今天 = date.today().isoformat()
    return (
        f'<div class="cover"><h1>{标题}</h1>'
        f'<div class="subtitle">{副标题}</div>'
        f'<div class="meta">091635Aa 商业化推进系统 · {今天}</div></div>'
    )


def 主():
    parser = argparse.ArgumentParser(description="Markdown → PDF 多风格排版")
    parser.add_argument("输入", help="输入 .md 文件")
    parser.add_argument("输出", help="输出 .pdf 文件")
    parser.add_argument("--style", choices=list(STYLES), default="business", help="排版风格")
    parser.add_argument("--cover", action="store_true", help="生成封面页")
    parser.add_argument("--title", default=None, help="封面/文档标题（默认取首行）")
    parser.add_argument("--subtitle", default="", help="封面副标题")
    parser.add_argument("--footer", default=None, help="页脚文字（默认随风格）")
    args = parser.parse_args()

    md文本 = Path(args.输入).read_text(encoding="utf-8")
    body_html = markdown.markdown(md文本, extensions=["tables", "fenced_code", "sane_lists", "nl2br"])
    标题 = args.title or md文本.splitlines()[0].lstrip("# ").strip()

    css = STYLES[args.style]
    if args.footer:
        css = css.replace("{footer}", args.footer)
    封面 = 构建封面(标题, args.subtitle) if args.cover else ""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{标题}</title>
<style>{css}</style></head>
<body>{封面}{body_html}</body></html>"""

    with TemporaryDirectory(prefix="md2pdf_", ignore_cleanup_errors=True) as 临时:
        临时 = Path(临时)
        html文件 = 临时 / "doc.html"
        html文件.write_text(html, encoding="utf-8")
        pdf文件 = Path(args.输出)
        pdf文件.parent.mkdir(parents=True, exist_ok=True)

        命令 = [
            EDGE,
            "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--user-data-dir={临时 / 'profile'}",
            f"--print-to-pdf={pdf文件}",
            html文件.as_uri(),
        ]
        proc = subprocess.Popen(命令, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            if pdf文件.exists() and pdf文件.stat().st_size > 0:
                break
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        else:
            proc.kill()
            print("超时：Edge 未在 60s 内完成"); sys.exit(1)
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        time.sleep(1)

        if not pdf文件.exists() or pdf文件.stat().st_size == 0:
            print("错误：未生成 PDF"); sys.exit(1)

    print(f"PDF 已生成: {pdf文件} ({pdf文件.stat().st_size/1024:.0f} KB, style={args.style})")


if __name__ == "__main__":
    主()
