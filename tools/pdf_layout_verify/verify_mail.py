# -*- coding: utf-8 -*-
"""验证地址封面 PDF 与 DOCX"""
import re
import zipfile
from pathlib import Path
import pymupdf

d = Path(r'D:\AI情感\091635Aa_商业化推进\提案\地址封面')

# ── PDF ──
pdf = d / '邮件地址封面_打印版.pdf'
doc = pymupdf.open(str(pdf))
print(f'PDF 页数: {len(doc)} (预期 2)')
expect_pdf = [
    '腾讯 AI Lab / 元宝项目组', '北京市海淀区西北旺东路8号', '腾讯北京总部大楼',
    '腾讯 AI Lab', '海天二路33号', '腾讯滨海大厦',
    '豆包大模型团队 / Seed 项目组', '中关村大街11号',
    '通义大模型技术团队 / 达摩院', '灯彩街1008号', '云谷园区',
    '逆熵研究部', '战略投资部', '宜山路700号',
    '技术评测组', '环城北路169号', '汇金国际大厦西1幢1201室（转：技术评测组）',
    '0755-86013388', '010-58341796', '0571-85022088', '021-64710018',
    '邓斯键', '13660110716', '街口街雅景居',
]
txt_all = re.sub(r'\s+', '', ''.join(p.get_text() or '' for p in doc))
miss = [k for k in expect_pdf if re.sub(r'\s+', '', k) not in txt_all]
print(f'PDF 内容缺失: {miss if miss else "无"}')

# 越界检查
bad = []
for pi, p in enumerate(doc):
    W, H = p.rect.width, p.rect.height
    for b in p.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                if s['text'].strip():
                    x0, y0 = s['origin']
                    if x0 < 0 or x0 > W - 2 or y0 < 0 or y0 > H - 2:
                        bad.append((pi + 1, round(x0, 1), round(y0, 1), s['text'][:16]))
print(f'PDF 越界 span: {bad if bad else "无"}')

# ── DOCX ──
dx = d / '邮件地址封面_可编辑版.docx'
with zipfile.ZipFile(str(dx)) as z:
    names = z.namelist()
    xml = z.read('word/document.xml').decode('utf-8')
print(f'DOCX 结构: {names}')
exp_docx = ['邓斯键', '13660110716', '雅景居', '元宝项目组', '豆包大模型团队',
            '达摩院', '逆熵研究部', '战略投资部', '技术评测组', '0755-86013388',
            '010-58341796', '0571-85022088', '021-64710018', 'E世界财富中心',
            '普天科创产业园', '汇金国际大厦']
miss2 = [k for k in exp_docx if k not in xml]
print(f'DOCX 内容缺失: {miss2 if miss2 else "无"}')
page_br = 'w:type="page"'
print(f'DOCX 分页符数: {xml.count(page_br)} (预期 6)')
