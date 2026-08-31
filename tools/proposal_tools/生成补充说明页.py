# -*- coding: utf-8 -*-
"""
091635Aa 商业化推进系统 · 补充说明页生成器（fpdf2 版 · v2 布局修复）
为已打印的提案邮件生成「补充说明页」PDF（单页 A4，插在邮件最上方）。

v2 变更：
  - 修复 v1 布局错位：弃用 fpdf2 multi_cell(markdown=True)（行内加粗解析 bug
    导致列表项被渲染到页面外），改用 write() 手动分片渲染，逐 span 坐标可控。
  - 内容扩充：身份背景 / 实验概览 / 13 组核心对照实验 E1–E13 测试方案 /
    转交要求 / 完整联系渠道（主备邮箱、电话、微信、QQ）/ 参考仓库入口。

用法:
  python 生成补充说明页.py [--公司 腾讯] [--邮箱 xxx@dingtalk.com] [--电话 13xxxxxxxxx]

产出: 提案/PDF/补充说明页_[公司名].pdf（与原提案同目录，不修改原 PDF）
"""
import argparse, sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]          # 091635Aa_商业化推进
PDF_DIR = ROOT / "提案" / "PDF"

FONT_SONG = r"C:\Windows\Fonts\simsun.ttc"          # 宋体（正文）
FONT_HEI = r"C:\Windows\Fonts\simhei.ttf"           # 黑体（标题/强调，充当粗体）

# ── 公司 → 定制产品名（依据各公司情报文件）──────────────────────
公司列表 = [
    ("腾讯",      "贵司腾讯元宝/混元模型"),
    ("字节跳动",  "贵司豆包模型"),
    ("阿里云",    "贵司通义千问（Qwen）模型"),
    ("米哈游",    "贵司游戏角色 AI / Glossa 模型"),
    ("深度求索",  "贵司 DeepSeek 模型"),
]

默认邮箱 = "dypubg@dingtalk.com"
默认备用邮箱 = "13660110716@163.com / dypubg2025@163.com"
默认电话 = "13660110716"
默认QQ = "3795423641"

# 配色（打印友好：深蓝标题 + 灰线，正文深灰）
深蓝 = (22, 48, 88)
正文灰 = (40, 40, 40)
浅灰 = (130, 130, 130)
线灰 = (185, 195, 210)

正文字号 = 10.5
行高 = 5.6                                 # mm
内容宽 = 210 - 16 - 16                       # 左右边距 16mm

# ── 内容模板（{产品}/{公司} 按需定制）────────────────────────
导语 = ("本人为**语义回响（Semantic Echo）**项目独立创作者，个人开发者（初中毕业）。"
        "2026 年暑假期间独立完成本方案的研究、实现与复现；全部源码与实验数据已开源上传 "
        "GitHub 仓库（公开可复核）。对复现结果有绝对把握，愿以 30 天内实测交付自证。")

实验概览 = [
    "**P6 情感导演**（解码期零权重插件）：5 项图灵基准综合 **0.7046**（裸基座 0.5800，**+21.5%**），全家族最高",
    "真人胜率（AI vs 真人盲评）提升 **+122%**（0.30 → 0.667）；Qwen3-1.7B 达 0.833，评分超真人均值",
    "显存**零增加**（峰值 3.78GB 不变），吞吐仅下降 **-2%**，零权重、零新增参数",
    "已在 Qwen / DeepSeek / Gemma / Phi / SmolLM 等 **7+ 系列模型**上完成验证，泛化提升 **+67%~+233%**",
]

测试方案 = [
    "**13 组核心对照实验 E1–E13**：三轮设计（λ/γ 强度扫描 → 情感筛选 + 思考阶段分离 → 保留策略），同种子可复现",
    "**5 项图灵基准**：TuringBench（人似度翻倍）/ EmoCharacter / HeartBench / HEART-BENCH / LLM-Judge 盲评",
    "**8 个跨架构模型泛化**：Qwen2.5 全系 / Qwen3 / Gemma / Phi / SmolLM / DeepSeek-R1-Distill（7/8 正向）",
    "**全流程 7 模式 × 5 基准**（P1~P6 五代 + 裸基座，2026 最终版口径）：P6 综合 0.7046 全场第一",
]

请求段 = ("无需费用、无需接入生产。我方可在 **30 天内**，针对{产品}交付一份"
         "「人味维度对比评测报告」，供内部技术参考；如验证后认可价值，再谈后续合作。"
         "恳请将本函转交至贵司**客服 / 销售 / 相关技术团队**，感谢垂阅。")

联系渠道 = [
    "邮箱（主）：**dypubg@dingtalk.com**",
    "邮箱（备用）：**13660110716@163.com** / dypubg2025@163.com",
    "电话 / 微信：**13660110716**（微信同号）",
    "QQ：**3795423641**（如需微信 / 钉钉等即时渠道，可回复本邮箱另行告知）",
]

仓库入口 = [
    "主仓库：**github.com/091635Aa/SemanticEcho**",
    "家族主页：github.com/091635Aa/SemanticEcho-Home　·　冠军插件 P6：github.com/091635Aa/SemanticEcho-EDD-OpenSource",
]


class 补充页(FPDF):
    """A4 单页；页脚用标准 footer() 绘制，避免被自动分页挤出。"""

    def footer(self):
        self.set_y(-12)
        self.set_font("Song", "", 8)
        self.set_text_color(*浅灰)
        self.cell(0, 4, "补充说明页 · 语义回响 PoC 邀约 · 第 {0} 页".format(self.page_no()), align="C")


def 写流(pdf, 文本, x):
    """按 ** 分片，交替宋体/黑体连续写入（write 自动换行，不依赖 markdown 解析）。"""
    pdf.set_text_color(*正文灰)
    pdf.set_x(x)
    for i, seg in enumerate(文本.split("**")):
        if not seg:
            continue
        pdf.set_font("Song", "B" if i % 2 == 1 else "", 正文字号)
        pdf.write(行高, seg)


def 段落(pdf, 文本):
    写流(pdf, 文本, pdf.l_margin)
    pdf.ln(行高 + 1.2)


def 列表项(pdf, 文本, 缩进=3.5):
    pdf.set_font("Song", "", 正文字号)
    pdf.set_text_color(*正文灰)
    pdf.set_x(pdf.l_margin + 缩进)
    pdf.write(行高, "•  ")
    写流(pdf, 文本, pdf.get_x())
    pdf.ln(行高 + 0.6)


def 区块标题(pdf, 标题):
    pdf.set_font("Song", "B", 12)
    pdf.set_text_color(*深蓝)
    pdf.set_x(pdf.l_margin)
    pdf.cell(内容宽, 7, 标题)
    pdf.ln(7.2)
    pdf.set_draw_color(*线灰)
    pdf.set_line_width(0.3)
    y = pdf.get_y() + 0.6
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(4.2)


def 生成(公司: str, 产品: str, 邮箱: str, 备用邮箱: str, 电话: str, qq: str, 输出: Path) -> Path:
    pdf = 补充页("P", "mm", "A4")                   # 210 × 297 mm
    pdf.set_auto_page_break(auto=False, margin=16)
    pdf.set_margins(16, 12, 16)
    pdf.add_font("Song", "", FONT_SONG)
    pdf.add_font("Song", "B", FONT_HEI)
    pdf.add_page()

    # ── 标题 ──────────────────────────────────────────────
    pdf.set_font("Song", "B", 15.5)
    pdf.set_text_color(*深蓝)
    pdf.multi_cell(内容宽, 8.6, "关于《语义回响》技术方案的\n技术复现说明与 PoC 邀约", align="C")
    pdf.ln(1.5)
    # 致公司 + 转交要求
    pdf.set_font("Song", "B", 10)
    pdf.set_text_color(60, 75, 100)
    pdf.cell(内容宽, 6, "致：{0}　｜　恳请转交贵司客服 / 销售 / 技术团队".format(公司), align="C")
    pdf.ln(7)
    pdf.set_draw_color(*线灰)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

    # ── 一、关于我 ────────────────────────────────────────
    区块标题(pdf, "一、关于我（个人创作者背景）")
    段落(pdf, 导语)

    # ── 二、实验概览 ──────────────────────────────────────
    区块标题(pdf, "二、实验概览（核心数据）")
    for 行 in 实验概览:
        列表项(pdf, 行)

    # ── 三、测试方案 ──────────────────────────────────────
    区块标题(pdf, "三、测试方案（13 组核心对照实验 + 5 项图灵基准 + 多模型泛化）")
    for 行 in 测试方案:
        列表项(pdf, 行)

    # ── 四、我的请求（PoC 邀约）───────────────────────────
    区块标题(pdf, "四、我的请求（PoC 邀约）")
    段落(pdf, 请求段.format(产品=产品))

    # ── 五、联系渠道 ──────────────────────────────────────
    区块标题(pdf, "五、联系渠道")
    for 行 in 联系渠道:
        列表项(pdf, 行)

    # ── 六、参考仓库入口 ──────────────────────────────────
    区块标题(pdf, "六、参考仓库入口（可复现）")
    for 行 in 仓库入口:
        列表项(pdf, 行)

    输出.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(输出))
    return 输出


def 主():
    parser = argparse.ArgumentParser(description="为提案生成补充说明页 PDF（fpdf2，A4 单页）")
    parser.add_argument("--公司", default=None, help="只生成指定公司（如 腾讯），默认全部")
    parser.add_argument("--邮箱", default=默认邮箱, help="主邮箱（钉钉）")
    parser.add_argument("--备用邮箱", default=默认备用邮箱, help="备用邮箱")
    parser.add_argument("--电话", default=默认电话, help="联系电话（微信同号）")
    parser.add_argument("--qq", default=默认QQ, help="QQ 号")
    args = parser.parse_args()

    结果 = []
    for 公司, 产品 in 公司列表:
        if args.公司 and 公司 != args.公司:
            continue
        输出 = 生成(公司, 产品, args.邮箱, args.备用邮箱, args.电话, args.qq,
                 PDF_DIR / "补充说明页_{0}.pdf".format(公司))
        结果.append((公司, 输出, 输出.stat().st_size))

    print("\n完成：")
    for 公司, 输出, 大小 in 结果:
        print("  {0}: {1} ({2:.0f} KB)".format(公司, 输出, 大小 / 1024))
    if not 结果:
        print("未匹配任何公司，请检查 --公司 参数。")
        sys.exit(1)


if __name__ == "__main__":
    主()
