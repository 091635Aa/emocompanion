# -*- coding: utf-8 -*-
"""md -> html -> pdf 构建脚本（高密度排版）"""
import os, glob, sys
import markdown
from weasyprint import HTML

DOC = "/workspace/rwkv_intro_doc"
PARTS_DIR = os.path.join(DOC, "parts")
STYLE = os.path.join(DOC, "style.css")

CSS = """
@page {
  size: A4;
  margin: 1.35cm 1.35cm 1.5cm 1.35cm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #94a3b8; font-family: "Noto Sans CJK SC"; }
  @top-right { content: "RWKV 架构详解 · 超全入门手册"; font-size: 7.5pt; color: #cbd5e1; font-family: "Noto Sans CJK SC"; }
}
@page cover { margin: 0; @bottom-center { content: none; } @top-right { content: none; } }
* { box-sizing: border-box; }
body {
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 9.8pt;
  line-height: 1.52;
  color: #1e293b;
  margin: 0;
  text-align: justify;
}
h1 { font-size: 17pt; color: #1e3a8a; margin: 0.9em 0 0.5em; padding-bottom: 0.25em; border-bottom: 2.5px solid #2563eb; page-break-before: always; line-height: 1.25; }
h1:first-of-type { page-break-before: auto; }
h2 { font-size: 13pt; color: #1d4ed8; margin: 0.85em 0 0.4em; padding-left: 0.45em; border-left: 5px solid #2563eb; line-height: 1.3; page-break-after: avoid; }
h3 { font-size: 11pt; color: #0f172a; margin: 0.7em 0 0.3em; page-break-after: avoid; }
h4 { font-size: 10.2pt; color: #334155; margin: 0.6em 0 0.25em; page-break-after: avoid; }
p { margin: 0.35em 0; }
ul, ol { margin: 0.3em 0 0.45em; padding-left: 1.6em; }
li { margin: 0.18em 0; }
strong { color: #0f172a; }
a { color: #2563eb; text-decoration: none; }
code {
  font-family: "DejaVu Sans Mono", monospace;
  background: #f1f5f9; border: 0.5px solid #e2e8f0;
  padding: 0.05em 0.3em; border-radius: 3px; font-size: 8.6pt; color: #b91c1c;
}
pre {
  background: #0f172a; color: #e2e8f0; border-radius: 6px;
  padding: 0.55em 0.7em; font-size: 8.3pt; line-height: 1.42;
  page-break-inside: avoid; white-space: pre-wrap; word-break: break-all;
}
pre code { background: transparent; border: none; color: inherit; padding: 0; font-size: inherit; }
blockquote {
  margin: 0.45em 0; padding: 0.4em 0.9em;
  background: #eff6ff; border-left: 4px solid #2563eb; color: #1e3a8a; border-radius: 0 6px 6px 0;
  page-break-inside: avoid;
}
blockquote p { margin: 0.2em 0; }
table {
  border-collapse: collapse; width: 100%; margin: 0.5em 0; font-size: 8.7pt;
  page-break-inside: auto;
}
tr { page-break-inside: avoid; }
th, td { page-break-inside: avoid; }
th { background: #1e3a8a; color: #fff; padding: 0.32em 0.5em; text-align: left; font-weight: 600; }
td { border: 0.5px solid #cbd5e1; padding: 0.28em 0.5em; vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }
img { max-width: 100%; height: auto; display: block; margin: 0.45em auto; page-break-inside: avoid; }
figure { margin: 0.45em auto; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.3pt; color: #64748b; margin-top: 0.2em; text-align: center; }
hr { border: none; border-top: 1px dashed #cbd5e1; margin: 0.7em 0; }
/* 提示卡片 */
.note { background:#f0fdf4; border:1px solid #86efac; border-left:4px solid #16a34a; padding:0.45em 0.8em; border-radius:6px; margin:0.5em 0; page-break-inside:avoid; }
.warn { background:#fefce8; border:1px solid #fde047; border-left:4px solid #ca8a04; padding:0.45em 0.8em; border-radius:6px; margin:0.5em 0; page-break-inside:avoid; }
.tip { background:#eff6ff; border:1px solid #bfdbfe; border-left:4px solid #2563eb; padding:0.45em 0.8em; border-radius:6px; margin:0.5em 0; page-break-inside:avoid; }
.key { background:#fef2f2; border:1px solid #fca5a5; border-left:4px solid #dc2626; padding:0.45em 0.8em; border-radius:6px; margin:0.5em 0; page-break-inside:avoid; }
.toc { font-size: 8.6pt; line-height: 1.35; columns: 2; column-gap: 1.6em; }
.toc ul { list-style: none; padding-left: 0; margin: 0; }
.toc li { margin: 0.12em 0; }
.toc a { color: #1e293b; text-decoration: none; }
.toc a::after { content: leader('.') target-counter(attr(href), page); color: #64748b; font-size: 8.2pt; }
.toc .toc-chapter { font-weight: 600; color: #1e3a8a; margin-top: 0.5em; }
.toc .toc-sub { padding-left: 1.15em; }
.toc .toc-sub a { color: #475569; font-size: 8.2pt; }
"""

COVER = """
<div style="page: cover; width:100%; height:100%; padding:0; margin:0; position:relative; page-break-after: always;">
  <div style="background:#0f172a; color:#fff; height:38%; padding:5cm 2cm 0 2cm;">
    <div style="font-size:15pt; color:#93c5fd; letter-spacing:3px;">RECEPTANCE WEIGHTED KEY VALUE</div>
    <div style="font-size:52pt; font-weight:bold; margin-top:0.6cm;">RWKV 架构详解</div>
    <div style="font-size:22pt; color:#fbbf24; margin-top:0.4cm;">从入门到落地 · 超全图文手册</div>
  </div>
  <div style="padding:0.8cm 2cm;">
    <div style="font-size:11pt; color:#334155; line-height:2.1;">
      给完全零基础的小白：用大白话 + 24 张示意图 + 代码实战，<br/>
      讲清楚 RWKV 是什么、为什么特殊、和 Transformer 有什么区别、怎么部署使用、<br/>
      以及如何套用到我们自己的 AI 陪伴 / 情感智能体工程里。
    </div>
    <div style="font-size:10pt; color:#64748b; margin-top:0.6cm; line-height:1.9;">
      涵盖内容：RNN / Transformer 基础 · RWKV 核心机制 · 架构拆解 · 与 Transformer 全面对比 ·<br/>
      版本进化史（v1→v8）· 部署教程（Ollama / llama.cpp / HuggingFace / vLLM）· 代码实战 ·<br/>
      微调与 State Tuning · 应用场景与工程套用 · 术语表 · 参考资料
    </div>
    <div style="border-top:1px solid #e2e8f0; margin:1cm 0 0.5cm;"></div>
    <div style="font-size:9pt; color:#94a3b8;">版本：2026-09 · 本文基于公开资料整理，仅供学习参考</div>
  </div>
</div>
"""

def toc_html(parts_tokens):
    """用 markdown toc 扩展生成的真实锚点 id 构建紧凑 TOC"""
    items = []
    for title, tokens in parts_tokens:
        # 章节目录项
        items.append(f'<li class="toc-chapter"><a href="#{title["id"]}">{title["name"]}</a></li>')
        subs = []
        for tok in tokens:
            if tok.get("level") == 2:
                subs.append(f'<li class="toc-sub"><a href="#{tok["id"]}">{tok["name"]}</a></li>')
        if subs:
            items.append("<ul>" + "".join(subs) + "</ul>")
    return '<div class="toc"><ul>' + "".join(items) + "</ul></div>"

def main():
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "*.md")))
    print("parts:", [os.path.basename(p) for p in parts])
    # 自定义 slugify：给每个标题生成唯一锚点 id，避免多章 h1 重名导致目录页码错乱
    idx = [0]
    def slugify(value, separator="-", **kwargs):
        idx[0] += 1
        return f"h{idx[0]}"
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "attr_list", "toc", "sane_lists"],
        extension_configs={"toc": {"slugify": slugify}})
    body_parts = []
    parts_tokens = []
    for p in parts:
        html = md.reset().convert(open(p, encoding="utf-8").read())
        body_parts.append(html)
        # 收集标题 token：第一个 h1 作为章节名，h2 作为子项
        toks = md.toc_tokens
        h1 = next((t for t in toks if t.get("level") == 1), None)
        parts_tokens.append((h1, toks))
    body = "\n".join(body_parts)
    toc = toc_html(parts_tokens)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{COVER}
<h1 style="page-break-before:always;">目录</h1>
{toc}
{body}
</body></html>"""
    with open(os.path.join(DOC, "doc.html"), "w", encoding="utf-8") as f:
        f.write(html)
    HTML(string=html, base_url=DOC).write_pdf(os.path.join(DOC, "RWKV架构详解_超全入门手册.pdf"))
    print("PDF DONE")

if __name__ == "__main__":
    main()
