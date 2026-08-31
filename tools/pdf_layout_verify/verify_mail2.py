# -*- coding: utf-8 -*-
"""验证地址封面 PNG：内容 / 无图形元素 / 居中"""
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, r'D:\AI情感\_py')
sys.path.insert(0, r'D:\AI情感\091635Aa_商业化推进\工具')

import pymupdf
import 生成地址封面 as G

OUT = Path(r'D:\AI情感\091635Aa_商业化推进\提案\地址封面')

with TemporaryDirectory(prefix='addr_verify_', ignore_cleanup_errors=True) as t:
    tmp = Path(t) / 'm.pdf'
    G.生成横向PDF(tmp)
    doc = pymupdf.open(str(tmp))
    print(f'页数: {len(doc)} (预期 7)')
    ok = True
    for i, p in enumerate(doc):
        tag = G.标签列表[i]
        W, H = p.rect.width, p.rect.height
        print(f'--- 第{i+1}页 {tag["短名"]}  页面 {W:.0f}x{H:.0f}pt (横向={W>H})')
        txt = re.sub(r'\s+', '', p.get_text() or '')
        checks = [tag['收件人'], tag['电话'], '邓斯键', '13660110716', '街口街雅景居']
        miss = [c for c in checks if re.sub(r'\s+', '', c) not in txt]
        # 地址多行分别检查
        for 行 in tag['地址'].split('\n'):
            if re.sub(r'\s+', '', 行) not in txt:
                miss.append(行)
        # 图形检查（应无裁剪框等矢量图形）
        draws = p.get_drawings()
        # 文本水平居中检查：所有正文 span 的 x 中心应接近页面中心
        spans = []
        for b in p.get_text('dict')['blocks']:
            for l in b.get('lines', []):
                for s in l['spans']:
                    if s['text'].strip():
                        spans.append((s['origin'][0], s['size'], s['text']))
        cx = W / 2
        if spans:
            body = [s for s in spans if s[1] >= 14]  # 大字号（收件人/地址/电话）
            off = [round(abs(s[0] + s[1] * 0.5 - cx)) for s in body] if body else []
        else:
            off = []
        print(f'  内容缺失: {miss if miss else "无"} | 矢量图形数: {len(draws)} (预期 0) | 大字行居中偏差: {off if off else "N/A"}')
        if miss or draws or (off and max(off) > 80):
            ok = False
    # PNG 文件检查
    pngs = sorted(OUT.glob('地址封面_*.png'))
    print(f'PNG 文件数: {len(pngs)} (预期 7)')
    for f in pngs:
        im = pymupdf.open(str(f))
        r = im[0].rect
        print(f'  {f.name}: {r.width:.0f}x{r.height:.0f}px')
    print('ALL OK' if ok and len(pngs) == 7 else 'PROBLEM')
