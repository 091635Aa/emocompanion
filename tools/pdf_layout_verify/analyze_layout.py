# -*- coding: utf-8 -*-
"""布局分析：输出 PDF 文本块/图形坐标，定位错位问题"""
import sys
import pymupdf

pdf = sys.argv[1] if len(sys.argv) > 1 else r'D:\AI情感\091635Aa_商业化推进\提案\PDF\补充说明页_腾讯.pdf'
doc = pymupdf.open(pdf)
page = doc[0]
W, H = page.rect.width, page.rect.height
print(f'页面: {W:.0f}x{H:.0f} pt (A4=595x842)')
print('--- 文本 spans ---')
for b in page.get_text('dict')['blocks']:
    for l in b.get('lines', []):
        for s in l['spans']:
            txt = s['text'][:40]
            if txt.strip():
                print(f'({s["origin"][0]:6.1f},{s["origin"][1]:6.1f}) {s["font"][:14]:14} sz{s["size"]:4.1f}  {txt}')
print('--- 图形 ---')
for d in page.get_drawings():
    r = d['rect']
    print(f'({r.x0:6.1f},{r.y0:6.1f})-({r.x1:6.1f},{r.y1:6.1f}) type={d["type"]} color={d.get("color")} fill={d.get("fill")}')
