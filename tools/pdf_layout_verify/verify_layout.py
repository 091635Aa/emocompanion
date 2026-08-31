# -*- coding: utf-8 -*-
"""v2 布局验证：页数 / span 越界 / 内容关键词"""
import re
import sys
from pathlib import Path
import pymupdf

d = Path(r'D:\AI情感\091635Aa_商业化推进\提案\PDF')
公司列表 = ["腾讯", "字节跳动", "阿里云", "米哈游", "深度求索"]

关键词 = [
    "技术复现说明与 PoC 邀约",
    "独立创作者",
    "初中毕业",
    "0.7046",
    "+21.5%",
    "+122%",
    "-2%",
    "E1–E13",
    "TuringBench",
    "DeepSeek",
    "30 天内",
    "客服 / 销售",
    "dypubg@dingtalk.com",
    "13660110716@163.com",
    "13660110716",
    "3795423641",
    "github.com/091635Aa/SemanticEcho",
    "SemanticEcho-EDD-OpenSource",
]


def norm(s):
    return re.sub(r'\s+', '', s)


all_ok = True
for 公司 in 公司列表:
    f = d / f'补充说明页_{公司}.pdf'
    doc = pymupdf.open(str(f))
    page = doc[0]
    W, H = page.rect.width, page.rect.height
    n = len(doc)
    # span 越界检查
    spans = []
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                if s['text'].strip():
                    x0, y0 = s['origin']
                    spans.append((x0, y0, s['text']))
    越界 = [s for s in spans if s[0] < 40 or s[0] > W - 40 or s[1] > H - 30]
    text = norm(page.get_text() or '')
    missing = [k for k in 关键词 if norm(k) not in text]
    status = f'{公司}: 页数={n} spans={len(spans)} 越界={len(越界)}'
    if 越界:
        status += ' 越界详情=' + str([(round(x,1), round(y,1), t[:12]) for x, y, t in 越界[:5]])
    status += ' 缺失关键词=' + str(missing) if missing else ' 内容OK'
    print(status)
    if n != 1 or 越界 or missing:
        all_ok = False
print('ALL OK' if all_ok else 'PROBLEM')
