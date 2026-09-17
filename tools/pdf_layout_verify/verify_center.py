# -*- coding: utf-8 -*-
"""行级居中验证"""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, r'D:\AI情感\_py')
sys.path.insert(0, r'D:\AI情感\091635Aa_商业化推进\工具')

import pymupdf
import 生成地址封面 as G

with TemporaryDirectory(prefix='v3_', ignore_cleanup_errors=True) as t:
    tmp = Path(t) / 'm.pdf'
    G.生成横向PDF(tmp)
    doc = pymupdf.open(str(tmp))
    CX = 842 / 2
    ok = True
    for i, p in enumerate(doc):
        lines = []
        for b in p.get_text('dict')['blocks']:
            for l in b.get('lines', []):
                spans = [s for s in l['spans'] if s['text'].strip()]
                if not spans:
                    continue
                sz = max(s['size'] for s in spans)
                if sz >= 14:
                    x0, y0, x1, y1 = l['bbox']
                    lines.append((round((x0 + x1) / 2, 1), round(x0, 1), round(x1, 1), sz))
        bad = [ln for ln in lines if abs(ln[0] - CX) > 25]
        print(f'第{i+1}页 大字行数={len(lines)} 行中心偏离>25pt: {bad if bad else "无"}')
        if bad:
            ok = False
    print('居中检查:', 'PASS' if ok else 'FAIL')
