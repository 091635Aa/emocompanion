# -*- coding: utf-8 -*-
"""RWKV 介绍文档 - 示意图批量生成脚本"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
import os

# ---------- 字体与全局样式 ----------
FDIR = "/workspace/rwkv_intro_doc/fonts"
for f in os.listdir(FDIR):
    if f.endswith((".otf", ".ttf")):
        fm.fontManager.addfont(os.path.join(FDIR, f))
plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

OUT = "/workspace/rwkv_intro_doc/img"
os.makedirs(OUT, exist_ok=True)

# 配色
C_BLUE   = "#2563eb"   # 主蓝
C_BLUE_L = "#dbeafe"
C_ORANGE = "#f59e0b"
C_ORANGE_L = "#fef3c7"
C_GREEN  = "#10b981"
C_GREEN_L = "#d1fae5"
C_RED    = "#ef4444"
C_RED_L  = "#fee2e2"
C_PURPLE = "#8b5cf6"
C_PURPLE_L = "#ede9fe"
C_GRAY   = "#64748b"
C_GRAY_L = "#e2e8f0"
C_DARK   = "#0f172a"
C_WHITE  = "#ffffff"
C_TEAL   = "#0d9488"
C_TEAL_L = "#ccfbf1"

def _font(size=11, weight='normal'):
    return dict(fontsize=size, fontweight=weight, family='Noto Sans CJK SC')

def card(ax, x, y, w, h, text, fc=C_WHITE, ec=C_BLUE, tc=C_DARK, fs=11, weight='normal', radius=0.02, lw=1.4, sub=None, sub_fs=8.5, sub_tc=C_GRAY):
    """圆角矩形卡片，text 居中；sub 为右下角小字"""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    cy = y + h / 2
    if sub:
        cy = y + h * 0.58
        ax.text(x + w / 2, y + h * 0.20, sub, ha='center', va='center', **_font(sub_fs, 'normal'), color=sub_tc)
    ax.text(x + w / 2, cy, text, ha='center', va='center', **_font(fs, weight), color=tc)
    return p

def arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.6, style='-|>', ms=14, conn="arc3,rad=0.0", ls='-', zorder=3):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, connectionstyle=conn, linestyle=ls, zorder=zorder)
    ax.add_patch(a)

def txt(ax, x, y, s, fs=10.5, color=C_DARK, weight='normal', ha='center', va='center', alpha=1.0):
    ax.text(x, y, s, ha=ha, va=va, **_font(fs, weight), color=color, alpha=alpha)

def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    return fig, ax

def chart_fig(w, h, xlim, ylim):
    """真正的数据图表：使用数据坐标系，而不是 0-100 画布"""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return fig, ax

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches='tight', facecolor='white', pad_inches=0.08)
    plt.close(fig)
    print("ok", name)

# ============================================================
# 1. RNN 循环结构
# ============================================================
def fig_rnn_loop():
    fig, ax = new_fig(9.5, 4.2)
    txt(ax, 50, 94, "RNN 的循环结构：网络会把“上一次的记忆”一起吞进来", fs=15, weight='bold')
    ax.add_patch(Circle((28, 50), 12, fc=C_BLUE_L, ec=C_BLUE, lw=2, zorder=2))
    txt(ax, 28, 50, "RNN\n网络 A", fs=13, weight='bold', color=C_BLUE)
    # 输入
    for y, lab in [(33, "输入 x[0]"), (50, "输入 x[1]"), (67, "输入 x[2]")]:
        card(ax, 55, y - 6, 20, 12, lab, fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=11)
        arrow(ax, 55, y, 40, 50)
    # 循环箭头
    a = FancyArrowPatch((28 + 12, 58), (28 + 12, 64), arrowstyle='-|>', mutation_scale=16, color=C_RED, lw=2.2)
    a.set_connectionstyle("arc3,rad=-0.5")
    ax.add_patch(a)
    txt(ax, 44, 76, "把上一次的状态\n(h) 传回给网络", fs=9.5, color=C_RED, ha='left')
    # 输出
    card(ax, 6, 44, 16, 12, "预测输出\ny[1]/y[2]/y[3]", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=10.5)
    arrow(ax, 16, 50, 28 - 12, 50)
    txt(ax, 50, 8, "每吃进一个词，网络都会更新自己的“记忆状态”，并用新状态去预测下一个词。", fs=11, color=C_GRAY)
    save(fig, "fig01_rnn_loop.png")

# ============================================================
# 2. RNN 按时间展开
# ============================================================
def fig_rnn_unfold():
    fig, ax = new_fig(10.5, 4.4)
    txt(ax, 50, 94, "RNN 按时间展开：一排重复的“同一个小网络”", fs=15, weight='bold')
    xs = [12, 32, 52, 72, 92]
    for i, x in enumerate(xs):
        ax.add_patch(Circle((x, 58), 8, fc=C_BLUE_L, ec=C_BLUE, lw=2, zorder=2))
        txt(ax, x, 58, f"A", fs=13, weight='bold', color=C_BLUE)
        # 输入
        card(ax, x - 7, 26, 14, 10, f"x{i}", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=10)
        arrow(ax, x, 36, x, 50)
        # 输出
        card(ax, x - 7, 74, 14, 10, f"y[{i}]", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=10)
        arrow(ax, x, 66, x, 74)
        if i < 4:
            arrow(ax, x + 8, 58, xs[i + 1] - 8, 58)
    txt(ax, 50, 12, "同一个权重 A 被反复使用 → 参数少、有记忆；但只能“串行”算，不能并行 → 训练慢、容易忘", fs=11, color=C_GRAY)
    save(fig, "fig02_rnn_unfold.png")

# ============================================================
# 3. Transformer 解码块
# ============================================================
def fig_transformer_block():
    fig, ax = new_fig(10, 6.4)
    txt(ax, 50, 96, "Transformer（Decoder）单层结构：靠“注意力”翻旧账", fs=15, weight='bold')
    # 输入
    card(ax, 40, 88, 20, 8, "输入向量 x", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=11, weight='bold')
    # QKV
    for i, (x, lab, c) in enumerate([(10, "Query\n查询", C_BLUE), (50, "Key\n键", C_BLUE), (90, "Value\n值", C_BLUE)]):
        card(ax, x - 9, 68, 18, 14, lab, fc=C_BLUE_L, ec=c, tc=c, fs=10.5)
        arrow(ax, 50, 88, x, 82 if i else 82, color=C_GRAY)
    # 注意力核心
    card(ax, 40, 46, 20, 14, "自注意力\nQ·K^T / √d → softmax → ·V", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=10)
    for x in [10, 50, 90]:
        arrow(ax, x, 68, 40 + (x - 10) * 0.5 if x == 10 else (50 if x == 50 else 60), 60, color=C_GRAY)
    # 简化箭头
    arrow(ax, 10, 68, 42, 60); arrow(ax, 50, 68, 50, 60); arrow(ax, 90, 68, 58, 60)
    # 残差
    card(ax, 40, 30, 20, 8, "残差 + LayerNorm", fc=C_GRAY_L, ec=C_GRAY, tc=C_GRAY, fs=9.5)
    arrow(ax, 50, 46, 50, 38)
    # FFN
    card(ax, 40, 16, 20, 8, "前馈网络 FFN", fc=C_TEAL_L, ec=C_TEAL, tc=C_TEAL, fs=10)
    arrow(ax, 50, 30, 50, 24)
    card(ax, 40, 4, 20, 7, "残差 + LayerNorm", fc=C_GRAY_L, ec=C_GRAY, tc=C_GRAY, fs=9.5)
    arrow(ax, 50, 16, 50, 11)
    # 说明
    txt(ax, 50, 92, "", fs=10)
    save(fig, "fig03_transformer_block.png")

# ============================================================
# 4. 注意力机制与 N×N 矩阵
# ============================================================
def fig_attention():
    fig, ax = new_fig(10.5, 5.4)
    txt(ax, 50, 95, "自注意力：每个词都要“看一遍”历史上所有词 → 产生 N×N 矩阵", fs=15, weight='bold')
    # 左侧 token
    words = ["我", "很", "喜欢", "RWKV", "模型"]
    x0, y0 = 12, 14
    for i, w in enumerate(words):
        card(ax, x0, y0 + i * 12, 16, 9, f"词 {w}", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=9.5)
    # 矩阵
    for i in range(5):
        for j in range(5):
            if i >= j:
                fc = C_ORANGE_L if i == j else C_GRAY_L
                ax.add_patch(Rectangle((52 + j * 8, 76 - i * 8), 7.6, 7.6, fc=fc, ec=C_GRAY, lw=0.8))
    txt(ax, 52 + 5 * 4, 82, "N×N 注意力矩阵", fs=11, weight='bold', color=C_DARK)
    txt(ax, 52 + 5 * 4, 16, "序列越长，这个矩阵越大\n长度 N → 计算量 N²", fs=10.5, color=C_RED)
    arrow(ax, 30, 40, 50, 55, color=C_GRAY)
    txt(ax, 40, 46, "每个词与\n每个词打分", fs=9.5, color=C_GRAY, ha='center')
    txt(ax, 50, 4, "举个例子：1 万词的文本，就要算 1 万 × 1 万 = 1 亿个“配对分数”，代价随长度平方增长。", fs=11, color=C_GRAY)
    save(fig, "fig04_attention.png")

# ============================================================
# 5. KV Cache 增长 vs RWKV 恒定状态
# ============================================================
def fig_kv_cache():
    fig, ax = chart_fig(9.5, 5.2, (0, 17), (0, 10.6))
    lens = [1, 2, 4, 8, 16]
    tr = [0.2, 0.6, 1.6, 4.0, 9.0]
    rw = [0.6, 0.6, 0.6, 0.6, 0.6]
    ax.plot(lens, tr, marker='o', color=C_RED, lw=2.5, label="Transformer（KV Cache 线性增长）")
    ax.plot(lens, rw, marker='s', color=C_GREEN, lw=2.5, label="RWKV（隐藏状态恒定）")
    ax.fill_between(lens, rw, tr, color=C_RED, alpha=0.08)
    ax.set_xlabel("上下文长度（千 token）", **_font(11))
    ax.set_ylabel("推理显存（相对单位）", **_font(11))
    ax.set_xticks(lens)
    ax.tick_params(labelsize=10)
    ax.legend(prop={'size': 10, 'family': 'Noto Sans CJK SC'}, loc='upper left', framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title("推理时显存/内存占用：Transformer 越聊越重，RWKV 始终不变", **_font(15, 'bold'), pad=14)
    ax.text(1.2, 7.6, "同样 7B 模型：上下文越长，\nTransformer 缓存越大，\nRWKV 始终只要一份固定状态",
            ha='left', va='center', **_font(10.5), color=C_GRAY)
    fig.tight_layout()
    save(fig, "fig05_kv_cache.png")

# ============================================================
# 6. 复杂度曲线 O(N²) vs O(N)
# ============================================================
def fig_complexity():
    import numpy as np
    fig, ax = chart_fig(9.5, 5.2, (0, 21), (0, 230))
    N = np.linspace(1, 20, 200)
    ax.plot(N, N ** 2, color=C_RED, lw=2.5, label="Transformer 注意力：O(N²) 平方增长")
    ax.plot(N, 8 * N, color=C_GREEN, lw=2.5, label="RWKV 时间混合：O(N) 线性增长")
    ax.fill_between(N, 8 * N, N ** 2, color=C_RED, alpha=0.06)
    ax.set_xlabel("序列长度 N（千 token）", **_font(11))
    ax.set_ylabel("计算量（相对）", **_font(11))
    ax.legend(prop={'size': 10, 'family': 'Noto Sans CJK SC'}, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title("计算复杂度：序列长度翻倍，代价怎么涨？", **_font(15, 'bold'), pad=14)
    ax.text(1.4, 195, "序列越长，两者差距越大：\n4K 时约 4 倍，32K 时约 30 倍以上",
            ha='left', va='center', **_font(10.5), color=C_GRAY)
    fig.tight_layout()
    save(fig, "fig06_complexity.png")

# ============================================================
# 7. RWKV 名字含义
# ============================================================
def fig_rwkv_name():
    fig, ax = new_fig(10.5, 4.6)
    txt(ax, 50, 94, "RWKV 这个名字，来自它核心的四个组件", fs=16, weight='bold')
    data = [
        ("R", "Receptance", "感受态", "控制“过去的信息”\n被接受多少 → 类似门控", C_BLUE, C_BLUE_L),
        ("W", "Weight", "权重衰减", "位置相关的衰减向量\n可学习，控制记忆遗忘快慢", C_ORANGE, C_ORANGE_L),
        ("K", "Key", "键", "类似注意力里的 K\n代表“当前词是什么”", C_GREEN, C_GREEN_L),
        ("V", "Value", "值", "类似注意力里的 V\n代表“携带什么信息”", C_PURPLE, C_PURPLE_L),
    ]
    for i, (letter, en, zh, desc, c, cl) in enumerate(data):
        x = 6 + i * 24
        ax.add_patch(FancyBboxPatch((x, 20), 21, 60, boxstyle="round,pad=0.01,rounding_size=1.2",
                                    fc=cl, ec=c, lw=1.6, zorder=2))
        ax.add_patch(Circle((x + 10.5, 66), 8, fc=c, ec=c, zorder=3))
        txt(ax, x + 10.5, 66, letter, fs=20, weight='bold', color=C_WHITE)
        txt(ax, x + 10.5, 54, en, fs=11, weight='bold', color=c)
        txt(ax, x + 10.5, 47, zh, fs=10.5, color=c)
        txt(ax, x + 10.5, 34, desc, fs=9, color=C_DARK, va='center')
    txt(ax, 50, 10, "把 R、W、K、V 四个字母拼起来 → RWKV（中文名：元始智能）", fs=12, weight='bold', color=C_DARK)
    save(fig, "fig07_rwkv_name.png")

# ============================================================
# 8. RWKV 残差块结构
# ============================================================
def fig_rwkv_block():
    fig, ax = new_fig(10, 7.6)
    txt(ax, 50, 97, "RWKV 的一个残差块：时间混合 + 通道混合", fs=15, weight='bold')
    # 输入
    card(ax, 40, 88, 20, 7, "输入 x", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=11, weight='bold')
    # Token shift
    card(ax, 40, 76, 20, 7, "Token Shift（短卷积）", fc=C_GRAY_L, ec=C_GRAY, tc=C_GRAY, fs=9.5)
    arrow(ax, 50, 88, 50, 83)
    # Time mixing
    card(ax, 40, 58, 20, 13, "Time Mixing 时间混合\n（线性注意力 / WKV）\n← 负责“记过去”", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=9.8)
    arrow(ax, 50, 76, 50, 71)
    # 残差
    card(ax, 40, 48, 20, 6, "残差连接 + LayerNorm", fc=C_GRAY_L, ec=C_GRAY, tc=C_GRAY, fs=9)
    arrow(ax, 50, 58, 50, 54)
    # Token shift 2
    card(ax, 40, 37, 20, 7, "Token Shift", fc=C_GRAY_L, ec=C_GRAY, tc=C_GRAY, fs=9.5)
    arrow(ax, 50, 48, 50, 44)
    # Channel mixing
    card(ax, 40, 21, 20, 12, "Channel Mixing 通道混合\n（类似前馈 FFN）\n← 负责“加工特征”", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=9.8)
    arrow(ax, 50, 37, 50, 33)
    card(ax, 40, 11, 20, 6, "残差连接 + LayerNorm", fc=C_GRAY_L, ec=C_GRAY, tc=C_GRAY, fs=9)
    arrow(ax, 50, 21, 50, 17)
    card(ax, 40, 2, 20, 6, "输出", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=10, weight='bold')
    # 右侧说明
    txt(ax, 82, 66, "时间混合 ≈ 注意力\n（在“时间轴”上\n混合信息）", fs=9.5, color=C_BLUE, ha='center')
    txt(ax, 82, 30, "通道混合 ≈ FFN\n（在“特征通道”上\n做非线性变换）", fs=9.5, color=C_GREEN, ha='center')
    txt(ax, 50, 97, "", fs=10)
    save(fig, "fig08_rwkv_block.png")

# ============================================================
# 9. Token Shift 示意图
# ============================================================
def fig_token_shift():
    fig, ax = new_fig(10, 4.6)
    txt(ax, 50, 94, "Token Shift：把“上一个词”和“当前词”各取一部分混合起来", fs=15, weight='bold')
    card(ax, 10, 34, 20, 22, "上一个词\nx[t-1]", fc=C_GRAY_L, ec=C_GRAY, tc=C_GRAY, fs=11)
    card(ax, 38, 34, 20, 22, "当前词\nx[t]", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=11)
    # 混合
    ax.add_patch(Circle((66, 45), 9, fc=C_ORANGE_L, ec=C_ORANGE, lw=1.8, zorder=3))
    txt(ax, 66, 45, "μ 混合", fs=9, color=C_ORANGE, weight='bold')
    txt(ax, 66, 30, "Shift = μ·x[t] + (1−μ)·x[t-1]", fs=10, color=C_ORANGE, weight='bold')
    arrow(ax, 30, 45, 55, 45); arrow(ax, 58, 45, 57, 45)
    card(ax, 78, 34, 16, 22, "混合结果\n（带一点\n“上一步记忆”）", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=9)
    arrow(ax, 75, 45, 78, 45)
    txt(ax, 50, 12, "它像“一层很短、免费的卷积”，让模型在不增加计算量的前提下，摸到“上一个时刻”的信息。", fs=11, color=C_GRAY)
    save(fig, "fig09_token_shift.png")

# ============================================================
# 10. WKV 计算流程
# ============================================================
def fig_wkv_flow():
    fig, ax = new_fig(10.5, 5.2)
    txt(ax, 50, 95, "WKV 时间混合：用“指数衰减的记忆”代替注意力", fs=15, weight='bold')
    # 左侧输入
    card(ax, 4, 66, 16, 14, "输入 x[t]", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=11)
    card(ax, 4, 44, 16, 12, "K（键）", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=11)
    card(ax, 4, 24, 16, 12, "V（值）", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=11)
    card(ax, 4, 4, 16, 12, "R（感受态）", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=11)
    arrow(ax, 12, 66, 26, 50)
    arrow(ax, 12, 50, 26, 50); arrow(ax, 12, 30, 26, 30); arrow(ax, 12, 10, 26, 10)
    # 记忆状态
    ax.add_patch(FancyBboxPatch((28, 34), 20, 34, boxstyle="round,pad=0.01,rounding_size=1.4", fc=C_PURPLE_L, ec=C_PURPLE, lw=1.8, zorder=2))
    txt(ax, 38, 62, "记忆状态 S", fs=11.5, weight='bold', color=C_PURPLE)
    txt(ax, 38, 54, "（K·V 不断写入）", fs=8.8, color=C_PURPLE)
    txt(ax, 38, 47, "W：衰减因子", fs=10, color=C_ORANGE, weight='bold')
    txt(ax, 38, 40, "旧的记忆按指数\n逐步“淡忘”", fs=8.8, color=C_GRAY)
    a = FancyArrowPatch((48, 55), (60, 55), arrowstyle='-|>', mutation_scale=14, color=C_PURPLE, lw=1.6)
    a.set_connectionstyle("arc3,rad=-0.4"); ax.add_patch(a)
    txt(ax, 56, 62, "再写入\n新记忆", fs=8.5, color=C_PURPLE)
    # 输出
    card(ax, 66, 40, 18, 16, "加权求和\n读出\n（带衰减的记忆）", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=9.5)
    arrow(ax, 48, 55, 66, 50)
    # R 门控
    card(ax, 88, 40, 12, 16, "× R\n门控", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=10)
    arrow(ax, 84, 48, 88, 48)
    arrow(ax, 12, 10, 88, 42, color=C_GRAY, lw=1.4)
    # 输出
    card(ax, 88, 70, 12, 12, "输出\ny[t]", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=10.5, weight='bold')
    arrow(ax, 94, 56, 94, 70)
    txt(ax, 50, 2, "本质：把“全部历史”压缩成一个不断更新的状态向量，读出来时按“时间远近”自动打折（衰减）。", fs=11, color=C_GRAY)
    save(fig, "fig10_wkv_flow.png")

# ============================================================
# 11. 双模式：训练并行 / 推理循环
# ============================================================
def fig_dual_mode():
    fig, ax = new_fig(10.5, 5.6)
    txt(ax, 50, 96, "RWKV 的“双模式”：训练像 Transformer，推理像 RNN", fs=15, weight='bold')
    # 左侧训练
    ax.add_patch(FancyBboxPatch((2, 20), 46, 66, boxstyle="round,pad=0.01,rounding_size=1.4", fc=C_BLUE_L, ec=C_BLUE, lw=1.6, zorder=1))
    txt(ax, 25, 80, "① 训练：GPT 模式（并行）", fs=12.5, weight='bold', color=C_BLUE)
    words = ["t1", "t2", "t3", "t4", "t5"]
    for i, w in enumerate(words):
        card(ax, 6 + i * 8.2, 56, 7, 12, w, fc=C_WHITE, ec=C_BLUE, tc=C_BLUE, fs=9)
    txt(ax, 25, 46, "整段文本一次性喂入，\n所有 token 同时计算 → 可用 GPU 并行", fs=9.8, color=C_DARK)
    txt(ax, 25, 32, "→ 训练效率高，和 Transformer 一样", fs=9.8, weight='bold', color=C_BLUE)
    # 右侧推理
    ax.add_patch(FancyBboxPatch((52, 20), 46, 66, boxstyle="round,pad=0.01,rounding_size=1.4", fc=C_GREEN_L, ec=C_GREEN, lw=1.6, zorder=1))
    txt(ax, 75, 80, "② 推理：RNN 模式（循环）", fs=12.5, weight='bold', color=C_GREEN)
    # 循环
    ax.add_patch(Circle((75, 56), 9, fc=C_WHITE, ec=C_GREEN, lw=1.8, zorder=2))
    txt(ax, 75, 56, "状态 S", fs=9.5, weight='bold', color=C_GREEN)
    a = FancyArrowPatch((84, 58), (84, 52), arrowstyle='-|>', mutation_scale=12, color=C_GREEN, lw=1.6)
    a.set_connectionstyle("arc3,rad=-0.5"); ax.add_patch(a)
    txt(ax, 75, 40, "一次只处理一个词\n携带上一轮状态 S 往下传", fs=9.8, color=C_DARK)
    txt(ax, 75, 28, "→ 内存固定、速度恒定，\n  想聊多长就聊多长", fs=9.8, weight='bold', color=C_GREEN)
    txt(ax, 50, 12, "同一套权重、两种跑法：训练时摊开并行，推理时卷起来循环。", fs=11.5, weight='bold', color=C_DARK)
    save(fig, "fig11_dual_mode.png")

# ============================================================
# 12. 显存对比柱状图
# ============================================================
def fig_memory_compare():
    import numpy as np
    fig, ax = chart_fig(9.5, 5.4, (-0.7, 3.7), (0, 28.5))
    ax.set_title("推理内存对比（7B 模型，fp16）", **_font(15, 'bold'), pad=14)
    cats = ["1K", "4K", "16K", "64K"]
    tr = [0.4, 1.6, 6.4, 25.6]
    rw = [1.2, 1.2, 1.2, 1.2]
    x = np.arange(len(cats))
    ax.bar(x - 0.2, tr, 0.35, label="Transformer（KV Cache）", color=C_RED, alpha=0.85)
    ax.bar(x + 0.2, rw, 0.35, label="RWKV（隐藏状态）", color=C_GREEN, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_xlabel("上下文长度（token）", **_font(11))
    ax.set_ylabel("显存（GB，示意）", **_font(11))
    ax.legend(prop={'size': 10, 'family': 'Noto Sans CJK SC'}, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    for xi, v in zip(x - 0.2, tr):
        ax.text(xi, v + 0.5, f"{v}", ha='center', **_font(8.5), color=C_RED)
    for xi, v in zip(x + 0.2, rw):
        ax.text(xi, v + 0.5, f"{v}", ha='center', **_font(8.5), color=C_GREEN)
    fig.tight_layout()
    save(fig, "fig12_memory_compare.png")

# ============================================================
# 13. 版本进化时间线
# ============================================================
def fig_timeline():
    fig, ax = new_fig(11, 4.8)
    txt(ax, 50, 95, "RWKV 版本进化时间线（2021 → 2026）", fs=15, weight='bold')
    ax.plot([4, 96], [55, 55], color=C_GRAY, lw=2.5, zorder=1)
    data = [
        (8,  "v1\n2021", "长卷积\n原型验证", C_GRAY),
        (22, "v2/v3\n2022", "实现 RNN 模式\n首个模型发布", C_GRAY),
        (36, "v4\n2023", "首个正式大模型\n与 GPT 同级别\nEMNLP 收录", C_BLUE),
        (50, "v5 Eagle\n2024", "多粒度注意力\n矩阵状态", C_BLUE),
        (64, "v6 Finch\n2024", "数据相关衰减\nWorld 14B 开源", C_BLUE),
        (78, "v7 Goose\n2025", "动态状态演化\n超越 TC0\n3B 多语 SOTA", C_ORANGE),
        (92, "v8\n2025~", "实验阶段\nDeepEmbed 等", C_PURPLE),
    ]
    for x, t, sub, c in data:
        ax.add_patch(Circle((x, 55), 2.2, fc=c, ec=c, zorder=3))
        up = x % 2 == 0
        ty = 78 if up else 30
        ax.plot([x, x], [55, ty], color=C_GRAY, lw=1, ls=':')
        txt(ax, x, ty + (5 if up else -5), t, fs=9.5, weight='bold', color=c)
        txt(ax, x, ty + (-5 if up else 5), sub, fs=8, color=C_DARK)
    txt(ax, 50, 8, "v1–v3 是概念验证；v4 让 RNN 真正达到 GPT 级；v7 “Goose” 是目前社区公认的基准；v8 仍在实验。", fs=11, color=C_GRAY)
    save(fig, "fig13_timeline.png")

# ============================================================
# 14. RWKV-7 状态演化
# ============================================================
def fig_rwkv7_state():
    fig, ax = new_fig(11, 5.6)
    txt(ax, 50, 96, "RWKV-7「Goose」：动态状态演化（广义 Delta 规则）", fs=15, weight='bold')
    # 左侧旧式
    ax.add_patch(FancyBboxPatch((2, 30), 40, 54, boxstyle="round,pad=0.01,rounding_size=1.4", fc=C_GRAY_L, ec=C_GRAY, lw=1.5))
    txt(ax, 22, 78, "旧版线性注意力（v4–v6）", fs=11.5, weight='bold', color=C_GRAY)
    txt(ax, 22, 64, "S[t] = 衰减·S[t-1] + K·V", fs=11, weight='bold', color=C_DARK)
    txt(ax, 22, 52, "记忆只能“写进去”\n不能“改错”", fs=10, color=C_GRAY)
    txt(ax, 22, 42, "像：一直往账本里\n记，但不修改旧账", fs=9.5, color=C_GRAY)
    # 右侧新式
    ax.add_patch(FancyBboxPatch((58, 30), 40, 54, boxstyle="round,pad=0.01,rounding_size=1.4", fc=C_ORANGE_L, ec=C_ORANGE, lw=1.8))
    txt(ax, 78, 78, "RWKV-7 动态状态演化", fs=11.5, weight='bold', color=C_ORANGE)
    txt(ax, 78, 66, "S[t] = S[t-1]·(D[t] + 更新) + K·误差", fs=10, weight='bold', color=C_DARK)
    txt(ax, 78, 55, "先“读出”预测，算出误差\n再按误差修正记忆", fs=9.8, color=C_DARK)
    txt(ax, 78, 45, "像：根据预测错误\n主动“改作业”\n→ 更强记忆力 / 少样本学习", fs=9.5, color=C_ORANGE)
    # 底部结论
    txt(ax, 50, 16, "论文证明：RWKV-7 能完成“状态追踪”，可识别所有正则语言 → 表达力超出注意力受限的 TC0 复杂度类", fs=11, weight='bold', color=C_RED)
    txt(ax, 50, 6, "通俗说：它能做很多 Transformer 理论上“做不到”的推理，同时训练仍可并行、推理仍是常数开销。", fs=10, color=C_GRAY)
    save(fig, "fig14_rwkv7_state.png")

# ============================================================
# 15. 部署选型流程图
# ============================================================
def fig_deploy_flow():
    fig, ax = new_fig(10.5, 6.8)
    txt(ax, 50, 97, "怎么跑起来？一张部署选型流程图", fs=15, weight='bold')
    card(ax, 40, 88, 20, 8, "选择 RWKV 模型", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=11.5, weight='bold')
    # 决策
    card(ax, 8, 70, 24, 10, "有 GPU 且要\n高吞吐服务？", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=10)
    card(ax, 38, 70, 24, 10, "要本地一键\n图形界面？", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=10)
    card(ax, 68, 70, 24, 10, "普通 CPU /\n低内存设备？", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=10)
    arrow(ax, 50, 88, 20, 80); arrow(ax, 50, 88, 50, 80); arrow(ax, 50, 88, 80, 80)
    # 方案
    card(ax, 2, 46, 26, 14, "方案 A\nvLLM / SGLang\nOpenAI 兼容 API", fc=C_WHITE, ec=C_BLUE, tc=C_BLUE, fs=9.8)
    card(ax, 34, 46, 26, 14, "方案 B\nRWKV-Runner /\nOllama / WebUI", fc=C_WHITE, ec=C_GREEN, tc=C_GREEN, fs=9.8)
    card(ax, 66, 46, 26, 14, "方案 C\nllama.cpp GGUF\n量化（Q5/Q8）", fc=C_WHITE, ec=C_PURPLE, tc=C_PURPLE, fs=9.8)
    arrow(ax, 12, 70, 12, 60); arrow(ax, 50, 70, 50, 60); arrow(ax, 80, 70, 80, 60)
    # 代码路径
    card(ax, 10, 26, 30, 10, "Python：pip install rwkv\n或 transformers 加载", fc=C_WHITE, ec=C_GRAY, tc=C_GRAY, fs=9.2)
    card(ax, 56, 26, 30, 10, "命令行：ollama run / llama-cli", fc=C_WHITE, ec=C_GRAY, tc=C_GRAY, fs=9.2)
    txt(ax, 50, 12, "从最小模型（0.1B）试起 → 再逐步换大模型 / 更高精度量化，按你的显存与延迟目标选。", fs=10.5, color=C_GRAY)
    save(fig, "fig15_deploy_flow.png")

# ============================================================
# 16. 推理调用流程
# ============================================================
def fig_api_flow():
    fig, ax = new_fig(10.5, 4.8)
    txt(ax, 50, 94, "一次推理的完整数据流（多轮对话会一直传递“状态 S”）", fs=15, weight='bold')
    steps = [
        (4,  "用户输入\n文本", C_ORANGE),
        (24, "Tokenize\n切成 token", C_GRAY),
        (44, "RWKV 前向\nforward", C_BLUE),
        (64, "采样\nsample", C_GREEN),
        (84, "Decode\n还原文本", C_PURPLE),
    ]
    for x, t, c in steps:
        card(ax, x, 44, 16, 16, t, fc=C_WHITE, ec=c, tc=c, fs=10)
    for i in range(len(steps) - 1):
        arrow(ax, steps[i][0] + 16, 52, steps[i + 1][0], 52)
    # 状态回路
    a = FancyArrowPatch((60, 44), (60, 30), arrowstyle='-|>', mutation_scale=12, color=C_RED, lw=1.6)
    a.set_connectionstyle("arc3,rad=0.4"); ax.add_patch(a)
    txt(ax, 66, 28, "状态 S 传给下一轮\n（RWKV 的记忆）", fs=9, color=C_RED, ha='left')
    # 温度等
    txt(ax, 50, 12, "采样的关键参数：temperature（温度）、top_p（核采样）、top_k、重复惩罚 → 决定“随机性 vs 稳定”", fs=10.5, color=C_GRAY)
    save(fig, "fig16_api_flow.png")

# ============================================================
# 17. 微调流程
# ============================================================
def fig_finetune_flow():
    fig, ax = new_fig(10.5, 5.4)
    txt(ax, 50, 95, "RWKV 微调 / 个性化：让模型学会你的数据", fs=15, weight='bold')
    steps = [
        (3,  "准备数据集\n（问答/对话/指令）", C_ORANGE),
        (26, "格式化 + Tokenize", C_GRAY),
        (49, "微调训练\n（全参 / LoRA / PEFT）", C_BLUE),
        (72, "导出权重\n（.pth / GGUF）", C_GREEN),
        (95, "部署 + 验证", C_PURPLE),
    ]
    for x, t, c in steps:
        w = 17
        card(ax, x, 50, w, 18, t, fc=C_WHITE, ec=c, tc=c, fs=9.8)
    for i in range(len(steps) - 1):
        arrow(ax, steps[i][0] + 17, 59, steps[i + 1][0], 59)
    # 特色 state tuning
    card(ax, 20, 18, 60, 18, "RWKV 特色：State Tuning（状态微调）—— 不动权重，直接微调“隐藏状态”，\n就像给模型“调整心理状态”，任务就能变好 → 又快又省", fc=C_RED_L, ec=C_RED, tc=C_RED, fs=9.8)
    arrow(ax, 58, 50, 44, 36, color=C_RED, lw=1.4)
    txt(ax, 50, 6, "训练依旧可以并行（GPT 模式），显存远小于同规模 Transformer → 单卡也能微调。", fs=10.5, color=C_GRAY)
    save(fig, "fig17_finetune_flow.png")

# ============================================================
# 18. RAG 应用架构
# ============================================================
def fig_app_rag():
    fig, ax = new_fig(10.5, 5.6)
    txt(ax, 50, 96, "应用①：RAG 检索增强（让模型“查资料”再回答）", fs=15, weight='bold')
    # 文档入库
    card(ax, 4, 68, 18, 12, "企业文档 /\n知识库", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=10)
    card(ax, 28, 68, 18, 12, "分块 +\n向量化", fc=C_GRAY_L, ec=C_GRAY, tc=C_GRAY, fs=10)
    card(ax, 52, 68, 18, 12, "向量数据库", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=10)
    arrow(ax, 22, 74, 28, 74); arrow(ax, 46, 74, 52, 74)
    # 查询
    card(ax, 4, 40, 18, 12, "用户提问", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=10)
    card(ax, 28, 40, 18, 12, "检索相关片段", fc=C_GRAY_L, ec=C_GRAY, tc=C_GRAY, fs=10)
    arrow(ax, 22, 46, 28, 46); arrow(ax, 37, 68, 37, 52)
    # 注入 + 生成
    card(ax, 52, 40, 18, 12, "片段注入提示词\n（上下文）", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=9.5)
    arrow(ax, 46, 46, 52, 46)
    card(ax, 76, 40, 20, 12, "RWKV 生成回答\n（长上下文优势）", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=9.5)
    arrow(ax, 70, 46, 76, 46)
    arrow(ax, 94, 46, 94, 70, color=C_PURPLE)
    card(ax, 76, 70, 20, 12, "高质量回答", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=10.5, weight='bold')
    txt(ax, 50, 12, "RWKV 天然适合：上下文可以很长、显存恒定，把整篇文档片段一起塞给模型也不怕。", fs=11, color=C_GRAY)
    save(fig, "fig18_app_rag.png")

# ============================================================
# 19. 多轮对话 / 状态持久化
# ============================================================
def fig_app_chat():
    fig, ax = new_fig(10.5, 5.2)
    txt(ax, 50, 95, "应用②：长会话 / 角色扮演 / 陪伴对话（核心卖点）", fs=15, weight='bold')
    # 中间模型
    card(ax, 40, 36, 20, 20, "RWKV 模型\n（固定状态）", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=12, weight='bold')
    # 左侧用户
    card(ax, 4, 50, 16, 12, "用户：今天\n心情不好…", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=9.5)
    card(ax, 4, 20, 16, 12, "用户：讲个笑话？", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=9.5)
    arrow(ax, 20, 56, 40, 50); arrow(ax, 20, 26, 40, 42)
    # 右侧回复
    card(ax, 80, 50, 16, 12, "模型：我陪你\n聊聊～", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=9.5)
    card(ax, 80, 20, 16, 12, "模型：哈哈\n当然可以", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=9.5)
    arrow(ax, 60, 50, 80, 56); arrow(ax, 60, 42, 80, 26)
    # 状态持久化
    card(ax, 40, 6, 20, 10, "状态 S 序列化保存\n（跨会话也能续上）", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=9)
    arrow(ax, 50, 36, 50, 16)
    txt(ax, 50, 95, "", fs=10)
    txt(ax, 50, 92, "", fs=10)
    save(fig, "fig19_app_chat.png")

# ============================================================
# 20. 边缘设备部署
# ============================================================
def fig_app_edge():
    fig, ax = new_fig(10.5, 4.8)
    txt(ax, 50, 94, "应用③：边缘 / 低配设备（树莓派、旧电脑、手机）", fs=15, weight='bold')
    devs = [
        (6, "树莓派 / 开发板\nCPU", "2.9B Q8_0\n约 3GB"),
        (34, "旧笔记本 / 迷你主机\n8GB 内存", "2.9B 量化\n稳定 4.7 tok/s"),
        (62, "嵌入式 / 离线\n无网环境", "完全本地\n隐私安全"),
    ]
    for x, t, sub in devs:
        ax.add_patch(FancyBboxPatch((x, 40), 26, 30, boxstyle="round,pad=0.01,rounding_size=1.4", fc=C_PURPLE_L, ec=C_PURPLE, lw=1.6))
        txt(ax, x + 13, 60, t, fs=10, weight='bold', color=C_PURPLE)
        txt(ax, x + 13, 49, sub, fs=9, color=C_DARK)
    card(ax, 38, 12, 24, 12, "量化：Q5_1 / Q8_0\nGGUF 格式", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=9.5)
    arrow(ax, 30, 25, 38, 24, color=C_GRAY)
    txt(ax, 50, 3, "实测：在 8GB 内存、无 GPU 的迷你主机上，RWKV-7 2.9B 十轮对话速度保持恒定，而同样大小的 Transformer 中途内存耗尽崩溃。", fs=10, color=C_GRAY)
    save(fig, "fig20_app_edge.png")

# ============================================================
# 21. 套用我们的场景（AI 陪伴/情感智能体）
# ============================================================
def fig_app_us():
    fig, ax = new_fig(11, 6.4)
    txt(ax, 50, 97, "应用④：套用我们自己的 AI 陪伴 / 情感智能体工程", fs=15, weight='bold')
    # 用户
    card(ax, 2, 40, 16, 20, "用户\n（长期对话\n陪伴场景）", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=10)
    # 应用层
    ax.add_patch(FancyBboxPatch((22, 30), 40, 52, boxstyle="round,pad=0.01,rounding_size=1.4", fc=C_BLUE_L, ec=C_BLUE, lw=1.6))
    txt(ax, 42, 76, "智能体应用层", fs=11.5, weight='bold', color=C_BLUE)
    card(ax, 26, 58, 15, 11, "角色人设\n挂载", fc=C_WHITE, ec=C_BLUE, tc=C_BLUE, fs=9)
    card(ax, 43, 58, 15, 11, "情感状态\n追踪", fc=C_WHITE, ec=C_BLUE, tc=C_BLUE, fs=9)
    card(ax, 26, 42, 15, 11, "长期记忆\n（状态持久化）", fc=C_WHITE, ec=C_BLUE, tc=C_BLUE, fs=9)
    card(ax, 43, 42, 15, 11, "注入控制\n（prompt/LoRA）", fc=C_WHITE, ec=C_BLUE, tc=C_BLUE, fs=9)
    # 模型层
    card(ax, 68, 40, 16, 24, "RWKV-7\n（边缘部署\n低显存 / 长会话）", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=10, weight='bold')
    # 外设
    card(ax, 88, 30, 10, 40, "TTS\n语音\n/多模态\n接口", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=8.5)
    arrow(ax, 18, 50, 22, 50)
    arrow(ax, 62, 50, 68, 50)
    arrow(ax, 84, 50, 88, 50)
    # 说明
    txt(ax, 50, 24, "RWKV 的优势正好对上我们的痛点：", fs=11, weight='bold', color=C_DARK)
    txt(ax, 50, 15, "① 8GB 级低显存也能本地跑 → 隐私好、成本低　② 恒定内存 → 超长多轮会话不崩", fs=9.3, color=C_GRAY)
    txt(ax, 50, 7, "③ 状态可保存 → 角色“记忆”可持续　④ State Tuning → 角色个性化微调更省", fs=9.3, color=C_GRAY)
    save(fig, "fig21_app_us.png")

# ============================================================
# 22. 速度对比（多轮对话稳定性）
# ============================================================
def fig_speed_stable():
    import numpy as np
    fig, ax = chart_fig(9.5, 5.2, (0.6, 10.4), (0, 7))
    ax.set_title("多轮对话中的生成速度：恒定 vs 越来越慢", **_font(15, 'bold'), pad=14)
    turns = np.arange(1, 11)
    rw = np.full_like(turns, 4.7, dtype=float)
    tr = np.array([5.7, 5.4, 5.3, 5.2, 5.2, 0, 0, 0, 0, 0])  # turn6 崩溃
    ax.plot(turns, rw, marker='o', color=C_GREEN, lw=2.5, label="RWKV-7 2.9B（恒定 4.7 tok/s）")
    ax.plot(turns[:5], tr[:5], marker='s', color=C_RED, lw=2.5, label="Transformer 4B（第 6 轮内存耗尽）")
    ax.set_xlabel("对话轮数", **_font(11)); ax.set_ylabel("生成速度（tok/s）", **_font(11))
    ax.set_xticks(turns); ax.set_ylim(0, 7)
    ax.legend(prop={'size': 10, 'family': 'Noto Sans CJK SC'}, loc='lower left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(6.6, 2.2, "没有 KV Cache 的 RWKV，\n聊到第 10 轮和第 1 轮一样快。", ha='left', va='center', **_font(10.5), color=C_GRAY)
    fig.tight_layout()
    save(fig, "fig22_speed_stable.png")

# ============================================================
# 23. 主流架构对比一图流
# ============================================================
def fig_family_map():
    fig, ax = new_fig(10.5, 5.6)
    txt(ax, 50, 96, "RWKV 在“高效序列模型”家族中的位置", fs=15, weight='bold')
    # 三大类
    card(ax, 3, 60, 28, 22, "Transformer\n（GPT / LLaMA…）\nO(N²) 注意力\n最强但最贵", fc=C_RED_L, ec=C_RED, tc=C_RED, fs=10.5)
    card(ax, 36, 60, 28, 22, "RNN / SSM\n（LSTM / Mamba…）\n线性 / 常数开销\n但训练并行难", fc=C_GRAY_L, ec=C_GRAY, tc=C_GRAY, fs=10.5)
    card(ax, 69, 60, 28, 22, "RWKV\n（Mamba / RetNet 同族）\n线性复杂度 +\n并行训练 +\n常数推理", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=10.5, weight='bold')
    arrow(ax, 31, 71, 36, 71, color=C_GRAY)
    arrow(ax, 64, 71, 69, 71, color=C_GRAY)
    txt(ax, 33, 48, "左（贵但有表达力）", fs=9, color=C_RED)
    txt(ax, 84, 48, "右（便宜且高效）", fs=9, color=C_BLUE)
    txt(ax, 50, 34, "RWKV 想同时拿到：Transformer 的“训练效率 + 表达力” 与 RNN 的“推理效率 + 恒定内存”。", fs=11, weight='bold', color=C_DARK)
    card(ax, 10, 10, 80, 16, "同族对比：Mamba（SSM）、RetNet、Gated DeltaNet、RWKV-7 —— 都在“用状态代替注意力”这条路上卷效率；RWKV 是其中唯一“注意力 100% 去除”且从 0.1B 到 7.2B 全系开源可跑。", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=9.8)
    save(fig, "fig23_family_map.png")

# ============================================================
# 24. 信息流：为什么“无注意力”也能行
# ============================================================
def fig_why_works():
    fig, ax = new_fig(10.5, 5.0)
    txt(ax, 50, 95, "RWKV 为什么“没有注意力”也能记住上下文？", fs=15, weight='bold')
    # 三个机制
    items = [
        ("Token Shift", "每个词都带一点\n“上一刻”的影子\n→ 短程记忆", C_BLUE),
        ("指数衰减记忆 W", "K·V 写入状态 S\n旧记忆按指数淡忘\n→ 长程记忆可调", C_GREEN),
        ("感受态 R 门控", "决定每条信息\n“要不要影响我”\n→ 选择性注意", C_ORANGE),
    ]
    for i, (t, sub, c) in enumerate(items):
        x = 4 + i * 33
        ax.add_patch(FancyBboxPatch((x, 44), 30, 34, boxstyle="round,pad=0.01,rounding_size=1.4", fc=C_WHITE, ec=c, lw=1.8))
        txt(ax, x + 15, 70, t, fs=11.5, weight='bold', color=c)
        txt(ax, x + 15, 58, sub, fs=9.3, color=C_DARK)
    # 汇总
    card(ax, 18, 14, 64, 18, "三个机制合力 → 用“固定大小的状态”装下整个上下文历史，\n需要哪段就“按权重调出来”，不必每次和全部历史两两比对", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=10)
    arrow(ax, 50, 44, 50, 32, color=C_PURPLE)
    txt(ax, 50, 4, "代价：极端细节的“精准回忆”弱于完整注意力；所以长上下文版 RWKV-X 会再混入稀疏注意力来补强。", fs=9.8, color=C_GRAY)
    save(fig, "fig24_why_works.png")

# ============================================================
# 25. 生态核心项目地图
# ============================================================
def fig_ecosystem():
    fig, ax = new_fig(10.5, 6.2)
    txt(ax, 50, 96, "RWKV 生态核心项目地图：从“跑起来”到“改模型”一整套工具", fs=15, weight='bold')
    # 顶部入口
    card(ax, 34, 84, 32, 8, "RWKV 模型（0.1B → 14B 全系开源）", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=11, weight='bold')
    # 中间工具
    card(ax, 4, 58, 27, 12, "RWKV-Runner\n一键图形界面", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=9.8)
    card(ax, 37, 58, 27, 12, "Ollama / llama.cpp\nGGUF 通用推理", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=9.8)
    card(ax, 70, 58, 27, 12, "vLLM / Albatross\n高并发服务", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=9.8)
    arrow(ax, 50, 84, 17, 70); arrow(ax, 50, 84, 50, 70); arrow(ax, 50, 84, 83, 70)
    # 开发层
    card(ax, 4, 34, 27, 12, "pip: rwkv\n官方推理库", fc=C_WHITE, ec=C_TEAL, tc=C_TEAL, fs=9.8)
    card(ax, 37, 34, 27, 12, "RWKV-PEFT\nLoRA / State Tuning", fc=C_WHITE, ec=C_RED, tc=C_RED, fs=9.8)
    card(ax, 70, 34, 27, 12, "RWKV-LM\n源码与训练", fc=C_WHITE, ec=C_GRAY, tc=C_GRAY, fs=9.8)
    # 社区底座
    card(ax, 16, 14, 68, 12, "社区支撑：GitHub · Discord · rwkv.cn · Hugging Face · 官方 Wiki", fc=C_GRAY_L, ec=C_GRAY, tc=C_DARK, fs=10.5)
    txt(ax, 50, 5, "用法口诀：个人体验→Runner/Ollama；Python 集成→pip rwkv；生产高并发→vLLM；要改模型→RWKV-LM + RWKV-PEFT", fs=9.6, color=C_GRAY)
    save(fig, "fig25_ecosystem.png")

# ============================================================
# 26. 生产部署架构
# ============================================================
def fig_prod_arch():
    fig, ax = new_fig(10.5, 6.0)
    txt(ax, 50, 96, "三种生产部署形态：按并发与预算选型", fs=15, weight='bold')
    # 形态一
    card(ax, 2, 66, 30, 18, "形态一 · 单机轻服务\nllama.cpp / llama-server\nOpenAI 兼容 · 内网够用", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=9.6)
    card(ax, 2, 54, 30, 9, "适用：小团队 / 个人助手", fc=C_WHITE, ec=C_GREEN, tc=C_GRAY, fs=9)
    # 形态二
    card(ax, 35, 66, 30, 18, "形态二 · vLLM 高并发\ncontinuous batching\n多租户 SaaS 首选", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=9.6)
    card(ax, 35, 54, 30, 9, "适用：对外公众产品", fc=C_WHITE, ec=C_BLUE, tc=C_GRAY, fs=9)
    # 形态三
    card(ax, 68, 66, 30, 18, "形态三 · 混合路由\n小模型兜底 + 大模型兜顶\n路由层按意图分发", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=9.6)
    card(ax, 68, 54, 30, 9, "适用：成本敏感产品", fc=C_WHITE, ec=C_PURPLE, tc=C_GRAY, fs=9)
    # 共同底座
    card(ax, 16, 28, 68, 15, "共同底座：状态持久化（Redis/DB）+ 监控告警 + 内容安全 + 兜底策略", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=10.5)
    arrow(ax, 50, 54, 50, 43, color=C_ORANGE)
    txt(ax, 50, 10, "进阶提示：80% 简单请求走 RWKV 小模型量化版 → 20% 复杂请求路由到大模型 → 综合成本降一个量级", fs=9.8, color=C_GRAY)
    save(fig, "fig26_prod_arch.png")

# ============================================================
# 27. 状态管理与多角色记忆
# ============================================================
def fig_state_mgmt():
    fig, ax = new_fig(10.5, 5.8)
    txt(ax, 50, 96, "状态即身份：RWKV 把“记忆”变成一份可持久化、可隔离的文件", fs=15, weight='bold')
    # 三个角色
    roles = [
        (6,  "角色 A\n“温柔姐姐”", C_ORANGE),
        (39, "角色 B\n“毒舌损友”", C_GREEN),
        (72, "角色 C\n“工作助手”", C_PURPLE),
    ]
    for x, t, c in roles:
        card(ax, x, 70, 24, 14, t, fc=C_WHITE, ec=c, tc=c, fs=10.5)
        card(ax, x, 52, 24, 10, "独立状态\nstate_A.st", fc=C_WHITE, ec=c, tc=C_GRAY, fs=9.2)
        arrow(ax, x + 12, 70, x + 12, 62)
    # 数据库
    card(ax, 16, 30, 68, 13, "持久化层：Redis / 数据库 —— 一个角色一个 key，天然多用户隔离", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=10.5)
    for x in (18, 51, 84):
        arrow(ax, x, 52, x, 43, color=C_BLUE)
    # 恢复
    card(ax, 24, 8, 52, 12, "用户再次进入 → 读回状态 → 无缝续聊，“它还记得我”", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=10.5)
    arrow(ax, 50, 30, 50, 20, color=C_GREEN)
    txt(ax, 92, 40, "对比 Transformer：要存整段\nKV Cache，成本高得多", fs=8.8, color=C_RED, ha='right')
    save(fig, "fig27_state_mgmt.png")

# ============================================================
# 28. 未来演进方向
# ============================================================
def fig_future():
    fig, ax = new_fig(10.5, 5.4)
    txt(ax, 50, 96, "RWKV 未来演进：状态表达力越来越强，与 Transformer 走向融合", fs=15, weight='bold')
    # 演进时间线
    steps = [
        (2,  "v1–v4\n固定衰减\n证明可行", C_GRAY),
        (24, "v5/v6\n数据相关衰减\n懂上下文", C_GREEN),
        (46, "v7 Goose\nDSE 可修正\n逼近注意力", C_BLUE),
        (68, "v8 / RWKV-X\n更强状态\n混合架构", C_PURPLE),
        (88, "多模态\n长视频音频\n世界模型", C_ORANGE),
    ]
    for x, t, c in steps:
        card(ax, x, 62, 16, 18, t, fc=C_WHITE, ec=c, tc=c, fs=9.2)
    for i in range(len(steps) - 1):
        arrow(ax, steps[i][0] + 16, 71, steps[i + 1][0], 71)
    # 趋势框
    card(ax, 6, 30, 88, 20, "三大趋势：① 混合化（线性注意力 + 少量稀疏注意力）  ② 状态可编辑、可迁移的“记忆即产品”  ③ 恒定内存 → 长视频/长音频流式理解成为优势场", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=9.6)
    txt(ax, 50, 8, "理性判断：任务决定架构 —— 长上下文、低资源、私有化、流式 → 状态模型占优；极致效果 → Transformer；二者共存且融合", fs=9.8, color=C_GRAY)
    save(fig, "fig28_future.png")

# ============================================================
# 29. 进阶学习路线图
# ============================================================
def fig_learning_path():
    fig, ax = new_fig(10.5, 6.0)
    txt(ax, 50, 96, "从“会用”到“能研究”：三阶段学习路线", fs=15, weight='bold')
    # 阶段一
    card(ax, 2, 70, 30, 18, "阶段一 · 使用者（1-2周）\nOllama/Runner 跑通 2.9B\n玩熟状态保存 · 会调用", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=9.6)
    card(ax, 2, 58, 30, 8, "产出：能独立部署+对话", fc=C_WHITE, ec=C_GREEN, tc=C_GRAY, fs=8.8)
    # 阶段二
    card(ax, 35, 70, 30, 18, "阶段二 · 工程派（1-2月）\n精读 Time Mixing / WKV / DSE\n做一次 LoRA + State Tuning", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=9.6)
    card(ax, 35, 58, 30, 8, "产出：接入真实小产品", fc=C_WHITE, ec=C_BLUE, tc=C_GRAY, fs=8.8)
    # 阶段三
    card(ax, 68, 70, 30, 18, "阶段三 · 研究者（半年+）\n精读论文 · 复现简化 RWKV\n横向对比 Mamba / RetNet", fc=C_PURPLE_L, ec=C_PURPLE, tc=C_PURPLE, fs=9.6)
    card(ax, 68, 58, 30, 8, "产出：找到研究切入点", fc=C_WHITE, ec=C_PURPLE, tc=C_GRAY, fs=8.8)
    arrow(ax, 32, 79, 35, 79); arrow(ax, 65, 79, 68, 79)
    # 地基
    card(ax, 10, 34, 80, 14, "地基四件套：线性代数 → RNN 反向传播 → 注意力机制 → 状态空间模型（SSM）→ 先啃本手册第 17 章公式通俗课", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=9.8)
    txt(ax, 50, 12, "必读论文：RWKV-7 Goose · EMNLP 2023 · AFT · Mamba · RetNet · Gated DeltaNet（难度递增）", fs=9.8, color=C_GRAY)
    save(fig, "fig29_learning_path.png")

# ============================================================
# 30. Memo 伙伴项目架构
# ============================================================
def fig_memo_arch():
    fig, ax = new_fig(10.5, 6.2)
    txt(ax, 50, 96, "“Memo 伙伴”项目架构：前端 → 后端 → RWKV 推理 → 记忆/知识库", fs=15, weight='bold')
    # 前端
    card(ax, 3, 74, 22, 14, "前端\nHTML + JS\n聊天界面", fc=C_GREEN_L, ec=C_GREEN, tc=C_GREEN, fs=9.8)
    # 后端
    card(ax, 30, 74, 24, 14, "后端\nFastAPI\n编排与路由", fc=C_BLUE_L, ec=C_BLUE, tc=C_BLUE, fs=9.8)
    # 推理
    card(ax, 59, 74, 22, 14, "推理\nRWKV 2.9B\n(状态 S)", fc=C_ORANGE_L, ec=C_ORANGE, tc=C_ORANGE, fs=9.8)
    arrow(ax, 25, 81, 30, 81); arrow(ax, 54, 81, 59, 81)
    # 记忆
    card(ax, 30, 46, 24, 12, "记忆层\nSQLite\n存 state 文件", fc=C_WHITE, ec=C_PURPLE, tc=C_PURPLE, fs=9.6)
    # 知识库
    card(ax, 59, 46, 22, 12, "知识库\n本地检索\nRAG 外挂", fc=C_WHITE, ec=C_TEAL, tc=C_TEAL, fs=9.6)
    a = FancyArrowPatch((42, 74), (42, 58), arrowstyle='-|>', mutation_scale=13, color=C_PURPLE, lw=1.7)
    a.set_connectionstyle("arc3,rad=0.25"); ax.add_patch(a)
    txt(ax, 36, 60, "读回/保存\n状态", fs=9, color=C_PURPLE, ha='right')
    a2 = FancyArrowPatch((70, 74), (70, 58), arrowstyle='-|>', mutation_scale=13, color=C_TEAL, lw=1.7)
    a2.set_connectionstyle("arc3,rad=0.25"); ax.add_patch(a2)
    txt(ax, 64, 60, "检索资料", fs=9, color=C_TEAL, ha='right')
    # 回流
    a3 = FancyArrowPatch((59, 74), (54, 74), arrowstyle='-|>', mutation_scale=13, color=C_BLUE, lw=1.7)
    ax.add_patch(a3)
    txt(ax, 50, 70, "回复", fs=9, color=C_BLUE, va='top')
    # 底部说明
    card(ax, 8, 22, 84, 14, "三件套 = 角色卡（prompt）+ 状态记忆（state 存库）+ 知识库（RAG）——RWKV 的“状态”让记忆成为一等公民", fc=C_GRAY_L, ec=C_GRAY, tc=C_DARK, fs=10)
    txt(ax, 50, 6, "用户体验：同一 uid → 同一份记忆 → 重启也不失忆；“它还记得我”就是本项目的核心卖点", fs=9.8, color=C_GRAY)
    save(fig, "fig30_memo_arch.png")

# ============================================================
# 31. 报错排查决策树
# ============================================================
def fig_troubleshoot():
    fig, ax = new_fig(10.5, 6.2)
    txt(ax, 50, 96, "报错排查决策树：从最可能的根因开始查", fs=15, weight='bold')
    card(ax, 40, 86, 20, 8, "遇到报错", fc=C_RED_L, ec=C_RED, tc=C_RED, fs=11.5, weight='bold')
    rows = [
        (2,  "① 版本与环境\npip/torch/ollama 版本\npython -c 检查", C_GRAY),
        (26, "② 显存/内存\nnvidia-smi · free -h\n换量化 · 减上下文", C_BLUE),
        (50, "③ 模型文件\n.pth/.gguf 是否完整\n重新下载校验", C_GREEN),
        (74, "④ 参数设置\ntemperature/top_p\n重复惩罚 · 上下文长度", C_ORANGE),
        (96, "⑤ 代码逻辑\n最小复现 · 二分排除\n社区提问带三件套", C_PURPLE),
    ]
    # 用圆角矩形，宽度压到 x+14
    for x, t, c in rows:
        ax.add_patch(FancyBboxPatch((x, 60), 14, 20, boxstyle="round,pad=0.01,rounding_size=1.2", fc=C_WHITE, ec=c, lw=1.8))
        txt(ax, x + 7, 70, t, fs=8.8, color=c)
    arrow(ax, 50, 86, 9, 80); arrow(ax, 50, 86, 33, 80); arrow(ax, 50, 86, 57, 80); arrow(ax, 50, 86, 81, 80)
    # 90% 提示
    card(ax, 14, 32, 72, 16, "统计经验：90% 的问题集中在“版本不匹配”与“显存/内存不足”两类\n→ 永远先确认这两项，再往下排查", fc=C_GRAY_L, ec=C_GRAY, tc=C_DARK, fs=10.5)
    txt(ax, 50, 8, "求助模板：报错原文 + 模型版本 + 框架版本 + 硬件（显存/内存）+ 最小复现代码", fs=10, color=C_GRAY)
    save(fig, "fig31_troubleshoot.png")

if __name__ == "__main__":
    fig_rnn_loop()
    fig_rnn_unfold()
    fig_transformer_block()
    fig_attention()
    fig_kv_cache()
    fig_complexity()
    fig_rwkv_name()
    fig_rwkv_block()
    fig_token_shift()
    fig_wkv_flow()
    fig_dual_mode()
    fig_memory_compare()
    fig_timeline()
    fig_rwkv7_state()
    fig_deploy_flow()
    fig_api_flow()
    fig_finetune_flow()
    fig_app_rag()
    fig_app_chat()
    fig_app_edge()
    fig_app_us()
    fig_speed_stable()
    fig_family_map()
    fig_why_works()
    fig_ecosystem()
    fig_prod_arch()
    fig_state_mgmt()
    fig_future()
    fig_learning_path()
    fig_memo_arch()
    fig_troubleshoot()
    print("ALL DIAGRAMS DONE")
