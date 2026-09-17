# -*- coding: utf-8 -*-
"""
091635Aa 商业化推进系统 · 物理邮件地址封面生成器（PNG 打印版）
为 5 家目标公司生成邮寄地址封面图片，输出格式：
  1. PNG 图片（A4 横向 297×210mm，一页一个地址，居中布局，无裁剪框，300 DPI 打印质量）
  2. Word docx（每地址一页，可编辑；无依赖手写 OOXML）

寄件人：邓斯键　13660110716　广东省广州市从化区街口街雅景居
收件人：腾讯（北京/深圳） / 字节跳动 / 阿里云 / 米哈游（两部门）/ 深度求索（转技术评测组）

用法:
  python 生成地址封面.py
产出: 提案/地址封面/地址封面_*.png（7 张）+ 邮件地址封面_可编辑版.docx（不修改任何提案 PDF）
"""
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from fpdf import FPDF

# PyMuPDF 渲染 PNG（安装于工作区 _py 目录时自动挂载）
_PY = Path(__file__).resolve().parents[1].parent / "_py"
if _PY.exists() and str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

import pymupdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]          # 091635Aa_商业化推进
OUT_DIR = ROOT / "提案" / "地址封面"

FONT_SONG = r"C:\Windows\Fonts\simsun.ttc"          # 宋体
FONT_HEI = r"C:\Windows\Fonts\simhei.ttf"           # 黑体

# ── 寄件人 ──────────────────────────────────────────────
寄件人 = {
    "姓名": "邓斯键",
    "手机": "13660110716",
    "地址": "广东省广州市从化区街口街雅景居",
}

# ── 收件人标签（腾讯/米哈游按部门拆分，共 7 个）───────────────
标签列表 = [
    {"短名": "腾讯_北京元宝项目组", "公司": "腾讯 · 北京（首选）",
     "收件人": "腾讯 AI Lab / 元宝项目组",
     "地址": "北京市海淀区西北旺东路8号\n腾讯北京总部大楼", "电话": "0755-86013388"},
    {"短名": "腾讯_深圳AI Lab", "公司": "腾讯 · 深圳",
     "收件人": "腾讯 AI Lab",
     "地址": "广东省深圳市南山区海天二路33号\n腾讯滨海大厦", "电话": "0755-86013388"},
    {"短名": "字节跳动_豆包Seed", "公司": "字节跳动",
     "收件人": "豆包大模型团队 / Seed 项目组",
     "地址": "北京市海淀区中关村大街11号\nE世界财富中心A座12F", "电话": "010-58341796"},
    {"短名": "阿里云_通义达摩院", "公司": "阿里云",
     "收件人": "通义大模型技术团队 / 达摩院",
     "地址": "浙江省杭州市西湖区三墩镇灯彩街1008号\n云谷园区", "电话": "0571-85022088"},
    {"短名": "米哈游_逆熵研究部", "公司": "米哈游 · 逆熵研究部",
     "收件人": "逆熵研究部",
     "地址": "上海市徐汇区宜山路700号\n普天科创产业园C4号楼", "电话": "021-64710018"},
    {"短名": "米哈游_战略投资部", "公司": "米哈游 · 战略投资部",
     "收件人": "战略投资部",
     "地址": "上海市徐汇区宜山路700号\n普天科创产业园C4号楼", "电话": "021-64710018"},
    {"短名": "深度求索_技术评测组", "公司": "深度求索",
     "收件人": "技术评测组",
     "地址": "浙江省杭州市拱墅区环城北路169号\n汇金国际大厦西1幢1201室（转：技术评测组）",
     "电话": "13660110716（寄件人手机）"},
]

灰 = (130, 130, 130)


# ══════════════════════ 一、A4 横向地址页（中间 PDF）════════════
def 生成横向PDF(输出: Path) -> int:
    """A4 横向（297×210mm），每页一个地址，居中布局，无裁剪框。"""
    pdf = FPDF("L", "mm", "A4")                     # 297 × 210
    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(20, 12, 20)
    pdf.add_font("Song", "", FONT_SONG)
    pdf.add_font("Song", "B", FONT_HEI)
    内容宽 = 297 - 40

    for 标签 in 标签列表:
        pdf.add_page()

        # 左上角：寄件人（小五 10pt）
        pdf.set_font("Song", "", 10)
        pdf.set_text_color(90, 90, 90)
        pdf.set_xy(20, 14)
        for 行 in ["寄：{0}　{1}".format(寄件人["姓名"], 寄件人["手机"]),
                   "广东省广州市从化区", "街口街雅景居"]:
            pdf.cell(内容宽, 4.6, 行)
            pdf.ln(4.6)

        # 中央：收件人（大字，水平居中；垂直居中于 y=72 起）
        cy = 72
        pdf.set_xy(30, cy)
        pdf.set_font("Song", "B", 26)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(内容宽 - 20, 12.5, 标签["收件人"], align="C")

        pdf.set_xy(30, pdf.get_y() + 6)
        pdf.set_font("Song", "B", 18)
        pdf.multi_cell(内容宽 - 20, 9.5, 标签["地址"], align="C")

        pdf.set_xy(30, pdf.get_y() + 5)
        pdf.set_font("Song", "B", 16)
        pdf.multi_cell(内容宽 - 20, 8.5, "TEL：{0}".format(标签["电话"]), align="C")

        # 底部：寄件人姓名手机（小字居中）
        pdf.set_font("Song", "", 10)
        pdf.set_text_color(*灰)
        pdf.set_xy(0, 190)
        pdf.cell(297, 4.6, "寄件人：{0}　{1}".format(寄件人["姓名"], 寄件人["手机"]), align="C")

    输出.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(输出))
    return len(标签列表)


# ══════════════════════ 二、PNG 渲染 ══════════════════════
def 渲染PNG(pdf文件: Path, dpi: int = 300) -> list:
    doc = pymupdf.open(str(pdf文件))
    结果 = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        名 = "地址封面_{0:02d}_{1}.png".format(i + 1, 标签列表[i]["短名"])
        输出 = OUT_DIR / 名
        pix.save(str(输出))
        结果.append((名, 输出, pix.width, pix.height))
    return 结果


# ══════════════════════ 三、Word docx（手写 OOXML）══════════════
def 生成docx(输出: Path) -> None:
    def run(text, sz, 黑体=False, 居中=False, 加粗=False, before=0, after=0):
        rpr = "<w:rPr>"
        rpr += '<w:rFonts w:ascii="{0}" w:eastAsia="{0}"/>'.format("黑体" if 黑体 else "宋体")
        if 加粗:
            rpr += "<w:b/>"
        rpr += "<w:sz w:val=\"{0}\"/><w:szCs w:val=\"{0}\"/></w:rPr>".format(sz * 2)
        ppr = "<w:pPr><w:spacing w:before=\"{0}\" w:after=\"{1}\"/>{2}</w:pPr>".format(
            before, after, "<w:jc w:val=\"center\"/>" if 居中 else "")
        return "<w:p>{0}<w:r>{1}<w:t xml:space=\"preserve\">{2}</w:t></w:r></w:p>".format(ppr, rpr, text)

    body = []
    for i, 标签 in enumerate(标签列表):
        body.append(run("寄：{0}　{1}".format(寄件人["姓名"], 寄件人["手机"]), 9, after=60))
        body.append(run("广东省广州市从化区街口街雅景居", 9, after=240))
        body.append(run("", 9))
        body.append(run(标签["收件人"], 18, 黑体=True, 居中=True, 加粗=True, before=480, after=200))
        for 行 in 标签["地址"].split("\n"):
            body.append(run(行, 14, 黑体=True, 居中=True, after=80))
        body.append(run("TEL：{0}".format(标签["电话"]), 13, 黑体=True, 居中=True, before=160))
        body.append(run("寄件人：{0}　{1}".format(寄件人["姓名"], 寄件人["手机"]), 9, 居中=True, before=480))
        if i < len(标签列表) - 1:
            body.append("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>" + "".join(body) +
        '<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr>'
        "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )

    输出.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(输出), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)


def 主():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 清理旧的 PDF 打印版（改为 PNG 输出后不再需要；被占用时跳过并提示）
    for 旧 in OUT_DIR.glob("*.pdf"):
        try:
            旧.unlink()
        except PermissionError:
            print("提示：旧文件 {0} 正被其他程序占用，未能自动删除，可稍后手动删除。".format(旧.name))

    with TemporaryDirectory(prefix="地址封面_", ignore_cleanup_errors=True) as 临时:
        pdf文件 = Path(临时) / "mail.pdf"
        生成横向PDF(pdf文件)
        结果 = 渲染PNG(pdf文件, dpi=300)

    print("已生成 {0} 张 PNG（A4 横向 · 一页一地址 · 300 DPI）：".format(len(结果)))
    for 名, 路径, w, h in 结果:
        print("  {0}  ({1}x{2}px, {3:.0f} KB)".format(名, w, h, 路径.stat().st_size / 1024))

    docx文件 = OUT_DIR / "邮件地址封面_可编辑版.docx"
    生成docx(docx文件)
    print("DOCX: {0} ({1:.0f} KB)".format(docx文件, docx文件.stat().st_size / 1024))


if __name__ == "__main__":
    主()
