# -*- coding: utf-8 -*-
"""
091635Aa 商业化推进系统 · 提案生成器 V3
每家厂商一份完整内容 × 三种颜色样式输出（第一代/第二代/第三代）

核心逻辑：
  - 内容完全一致（5 步链路：①行业问题 ②项目介绍 ③对症下药 ④复现 ⑤价格）
  - 第一代/第二代/第三代 = 三种样式（颜色方案不同，内容相同）
    第一代 黑白极简   —— 纯黑白色系，最保守打印友好
    第二代 商务蓝     —— 深蓝主色 + 蓝灰表头，商务打印
    第三代 墨绿沉稳   —— 墨绿主色 + 绿灰表头，沉稳打印

视觉铁律：彩色仅用于标题/边框/表头等结构性元素，正文纯黑；
          无渐变、无花哨字体、80g A4 打印清晰。
"""
import sys, time, subprocess, argparse, re
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import date

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = Path(__file__).resolve().parent.parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# ══════════════════════════════════════════════════════════════
# 一、统一内容（行业问题 / 项目介绍 / 复现 / 价格）
# ══════════════════════════════════════════════════════════════
行业问题 = [
    ("情感陪伴赛道爆发", "全球 AI 陪伴市场 3179 亿美元（2033E，CAGR 31%）；中国情绪经济 2.72 万亿。豆包、星野、猫箱等把「人味」变成留客核心，用户为情感体验付费的意愿已获验证（付费率 18.7%）。"),
    ("「人味」是全行业公认缺口", "腾讯公开要「活人感」；DeepSeek 官方承认「效率 vs 情感」两难；用户吐槽千问「阿里味重」；猫箱被骂「人设崩坏、句式重复」。堆参数解决不了表达温度问题。"),
    ("2026-07 拟人化 AI 新规施行", "人格型陪伴智能体集中下架（豆包/千问等），行业急需「合规框架内提升人味」的技术方案——不制造情感依赖、不触碰安全边界。"),
]

项目介绍 = [
    ("零权重推理期情感增强", "不改模型权重、不重训练、不新增参数——只在推理时回收被丢弃的 Token 嵌入（「概率暴政」），把被压制的情绪信息作为「情感底色」叠加进下一次选词，让 AI「先想好情绪，再开口」。"),
    ("核心实测证据", "1.5B 小模型在 5 项图灵基准 4/5 反超裸基座；P6 情感导演综合 0.7046 全场第一（裸 0.5800，+21.5%）；LLM-Judge 对真人胜率 +122%；多模型泛化 +67%~+400%；峰值显存零增加（3.78GB 持平）、吞吐 -2%。"),
    ("开源可溯源", "全部源码 + 实验数据 + 论文已开源（GitHub: 091635Aa / SemanticEcho 家族 9 个仓库），同种子对照 + 裁判 AB 双投校准，可独立复现验证。"),
]

复现逻辑 = [
    ("环境", "Python 3.10 + PyTorch 2.5 + transformers；消费级 GPU 即可（1.5B fp16 约 3GB，RTX 16GB 满足 7B 裁判）。"),
    ("步骤", "克隆 GitHub 仓库 → pip install → 运行统一生成器（8 模式：裸/P1~P6）→ 5 大基准评测（同种子 42 对照）→ 裁判 AB 双投消位置偏差 → 输出对比报告。"),
    ("零权重断言", "全程 sum/data_ptr 校验权重零修改；在线坍缩检测自动兜底，任何模型不因注入退化。"),
]

价格表 = [
    ("PoC 验证合作", "30 天内交付「人味维度对比评测报告」，费用面议（可先看在线演示验证）。"),
    ("年费授权（大厂档）", "500 万元/年起，按年续约（年收入 ≥100 亿或员工 ≥1 万人企业）。"),
    ("永久 / 独家授权", "永久非独占 1,500 万元起；独家授权 1,000 万元/年；永久独家买断 1 亿~10 亿元。"),
]

技术家族 = [
    ["P1 语义回响", "表示空间", "回收被丢弃 Token 嵌入，注入「情感底色」"],
    ["P1.5 通用兼容层", "配置空间", "λ 四级归一化公式，任意模型自动适配"],
    ["P2.5 情感潮汐 ETD", "概率空间", "采样分布乘性重加权，数学上有界不崩"],
    ["P3 锚点回响 AE", "嵌入空间", "模型自身「情感词典」打分，轻推输出"],
    ["P4 KV 情感共振", "注意力缓存", "注意力向情绪词共振，越说越走心"],
    ["P5 超融合 UFD", "五空间合成", "P1~P4 一体化合成（综合 0.6255）"],
    ["P6 情感导演 EDD", "解码期插件", "TAD/PIS/OQC 三位导演×五通道（综合 0.7046 第一）"],
]

图灵数据 = [
    ["裸模型", "0.4783", "0.5692", "0.4100", "0.5333", "0.9090", "0.5800"],
    ["P1 语义回响", "0.4682", "0.5873", "0.2266", "0.9333", "0.8162", "0.6063"],
    ["P2.5 潮汐", "0.4982", "0.3219", "0.2967", "0.9333", "0.8887", "0.5878"],
    ["P3 锚点回响", "0.4949", "0.3075", "0.3300", "0.8000", "0.8500", "0.5565"],
    ["P4 KV 共振", "0.4720", "0.3141", "0.2767", "0.9333", "0.8778", "0.5748"],
    ["P5 超融合", "0.4909", "0.3203", "0.5067", "0.9333", "0.8762", "0.6255"],
    ["P6 情感导演", "0.4898", "0.5217", "0.7067", "0.9333", "0.8713", "0.7046"],
]

泛化数据 = [
    ["Qwen2.5-1.5B（主）", "0.200 → 0.667", "+233%"],
    ["Qwen2.5-3B", "0.417 → 0.750", "+80%"],
    ["Qwen3-1.7B", "0.500 → 0.833", "+67%"],
    ["Qwen2.5-0.5B", "0.250 → 0.750", "+200%"],
    ["Qwen3-0.6B", "0.250 → 0.750", "+200%"],
    ["SmolLM2-1.7B", "0.083 → 0.417", "+400%"],
    ["Phi-3.5-mini", "0.083 → 0.417", "+400%"],
    ["gemma-2-2b", "0.250 → 0.833", "+233%"],
]

性能数据 = [
    ["裸模型", "20.1", "3.78 GB", "—"],
    ["P6 情感导演", "19.7（-2%）", "3.78 GB（持平）", "零增加"],
]

技术原理详述 = (
    "「概率暴政」：自回归每步只在约 15 万词表中采样 1 个 token，其余候选的隐藏状态被系统性丢弃——"
    "这些被丢弃的表示并非噪声，而是锚定整个语义子空间的结构信息（「开心」被选中时「快乐/愉悦/难过」一并被弃）。"
    "语义回响做法：① 模型最后 4 层注册 forward hook，捕获 hidden_state 平均为「语义场」；"
    "② cnsenti 情感词库筛选，命中才入回响池（指数衰减维护）；"
    "③ 池质心经固定随机投影矩阵（Kaiming 缩放 √(2/hidden_dim) 保方差）映射到 logits 空间；"
    "④ logits += 池质心 @ 投影矩阵 × λ。全程零训练、零参数增量，数学依据为 Johnson-Lindenstrauss 引理。"
    "工程兜底：λ 四级归一化公式（hidden_dim × 架构族 × 量化）自动适配任意模型；"
    "思考链中断把坍缩模型拉回健康区间（重复率 0.84 → 0.0036）；在线坍缩检测自动降 λ；"
    "长上下文 λ 步数衰减（2048 tokens 实测不坍缩）；GPU 投影直分配（大词表模型不 OOM）。"
)

合作流程 = [
    "初步沟通：确认业务场景与评测口径",
    "PoC 验证：30 天内交付「人味维度对比评测报告」（我方出人出力）",
    "授权方案确认：年费授权 / 永久 / 独家，参考价格见第 5 节",
    "接入部署：零权重插件挂载推理服务，显存零增加、吞吐影响 <7%",
    "持续优化：任务自适应调优 + 版本迭代支持",
]

项目背景 = [
    "作者：邓同学（项目主导，独立完成概念构思→技术路线→实验执行→论文撰写），DeepSeek V4 辅助实现。",
    "论文：《语义回响：1.5B 模型情感表达增强与图灵测试实证研究》《决策的温度：底层记忆化 AI 扮演架构》。",
    "评测方法：5 大图灵基准（TuringBench / EmoCharacter / HeartBench / HEART-BENCH / LLM-as-Judge），同种子 42 严格对照，裁判 AB 双投校准消位置偏差。",
    "联系：DYPUBG2025@QQ.COM ｜ GitHub: 091635Aa ｜ 魔塔社区：DYSLPUBG/SemanticEcho",
]

开源清单 = [
    "SemanticEcho（P1 核心源码 + 实验数据 + 论文）",
    "SemanticEcho-ETD-OpenSource（P2.5 情感潮汐）",
    "SemanticEcho-AnchorEcho（P3 锚点回响）",
    "SemanticEcho-KVResonance（P4 KV 共振 + P5 超融合）",
    "SemanticEcho-EDD-OpenSource（P6 情感导演 · 全流程 2026 数据）",
    "SemanticEcho-Hub（总入口 · 大白话版 · 授权价目表）",
    "1.5B-beats-big-labs（图灵测试挑战 · 在线演示）",
]

# ══════════════════════════════════════════════════════════════
# 二、五家厂商定制内容（对症下药）
# ══════════════════════════════════════════════════════════════
CONTENT = {
"米哈游": {
    "en": "miHoYo", "副标题": "游戏内角色 AI 情感增强",
    "现状": [
        ["帕姆 AI 体验", "玩家吐槽「剧情设定防御型太强」，想聊设定被拒答", "NGA 玩家帖"],
        ["chatNPC 对比", "玩家称「游戏 chatNPC 不如酒馆」（第三方工具碾压）", "机核文章"],
        ["AI 陪伴试水", "两款陪伴产品停运，BSide 上线不足一月关停", "南方都市报"],
    ],
    "诊断": "帕姆 AI「剧情和设定方面防御型太强」，玩家想自由聊设定却被拒答；玩家直言「游戏 chatNPC 不如酒馆」（第三方角色扮演工具碾压官方）；玩家呼吁「随时召唤派蒙和旅行者聊天」；两款 AI 陪伴产品停运（BSide 未满月）——赛道缺的不是算力，是「情感真实感」。",
    "现在能用": [
        ("游戏内角色 AI 情感增强", "帕姆帮帮 / 星布谷地 NPC 挂载零权重情感增强层，守设定边界的同时有人味"),
        ("角色一致性不 OOC", "P6 OQC 在线质量纠正 + 角色感知锚定（v=0.7·角色+0.3·用户），角色基调跨轮稳定（3-run std=0.0）"),
    ],
    "未来能用": [
        ("全游戏 NPC 情感底座", "原神/崩铁/绝区零角色 AI 统一接入，N 个角色 N 种人格"),
        ("IP 角色陪伴产品", "「随时召唤角色聊天」的官方产品化——玩家诉求已在社区公开出现"),
    ],
    "怎么接入": "零权重插件挂载推理服务：不修改模型权重、不触碰内容安全与审核链路；P6 情感导演 + P3 锚点回响组合，即插即用，显存零增加。",
    "怎么收益": "角色陪伴付费、IP 情感资产增值、游戏内 AI 功能差异化（留住「被酒馆抢走的玩家」）。",
    "数据行": [["P6 综合", "0.7046", "裸 0.5800 · +21.5%"],
              ["LLM-Judge 胜率", "+122%", "0.30 → 0.667"],
              ["角色一致性", "std=0.0", "三轮种子完全一致"],
              ["峰值显存", "3.78GB", "与裸持平"]],
},
"深度求索": {
    "en": "DeepSeek", "副标题": "大模型情感增强层",
    "现状": [
        ["「变冷淡」风波", "2026-02 用户集体吐槽「冷漠、凶凶的」", "正观/IT之家/澎湃"],
        ["官方承认两难", "「为平衡效率与情感，正为 V4 做压力测试」", "太博快讯"],
        ["GitHub 诉求密集", "#703/#850/#1206/#1245 情感诉求 Issue", "deepseek-ai 官方仓库"],
    ],
    "诊断": "2026-02「变冷淡」风波：用户集体吐槽「冷漠、凶凶的、程式化」，官方回应「效率优先和边界意识」；官方公开承认「效率 vs 情感」两难并承诺 V4 恢复情感互动；官方 GitHub 仓库情感诉求 Issue 密集（#703/#850/#1206/#1245）；留存承压（「半年使用率暴跌 94%」讨论）。",
    "现在能用": [
        ("R1-Distill 已实测通过", "DeepSeek-R1-Distill-7B 全链路验证：未注册兜底命中 0.2166 ≥ 已注册定制 0.1572，即插即用"),
        ("V4/V4-Pro 情感补强", "以 V4 为实测对象交付「人味维度对比评测」作敲门砖，零权重不动效率与边界感"),
    ],
    "未来能用": [
        ("App 情感陪伴模式", "「不冷淡」的默认体验——直击变冷淡风波与留存问题"),
        ("Harness 生态插件", "DeepSeek Harness Agent 生态全插件化，情感增强中间件可建事实标准"),
        ("API 分层定价增值层", "情感增强作为差异化增值项（API 已从价格战转向分层定价）"),
    ],
    "怎么接入": "零权重推理期增强层，与模型解耦：不动权重、不重训练、不涉及训练数据——理论上可在不牺牲效率与边界感的前提下按需增强情感表达。",
    "怎么收益": "API 分层增值、C 端留存提升、Harness 生态事实标准带来的生态价值。",
    "数据行": [["情感命中率", "0.2166", "未注册兜底 ≥ 已注册 0.1572"],
              ["语义熵", "0.79 → 1.35", "+71% 表达更丰富"],
              ["重复率", "0.018 → 0.008", "无坍缩更健康"],
              ["即插即用", "26/30 条触发", "未注册模型直接可用"]],
},
"阿里云": {
    "en": "Alibaba Cloud", "副标题": "通义情感底座 + 百炼 MaaS 情感增强 API",
    "现状": [
        ["用户直评", "「千问阿里味很重」「豆包情感细腻远超千问」", "小红书用户对比贴"],
        ["行业评论", "「C 端大模型只剩工具人」，情感连接缺失", "钛媒体"],
        ["数字人战略", "小酒窝官方口径「能谈心，更能办事」", "IT之家"],
    ],
    "诊断": "用户直评「千问阿里味很重」「豆包情感细腻远超千问，更像是真人」；行业评论「C 端大模型只剩工具人」；36氪实测千问「接住情绪」未占优；数字人「小酒窝」官方口径「能谈心，更能办事」——情感陪伴已入战略，但情感能力是短板。",
    "现在能用": [
        ("Qwen 全系实测通过", "Qwen2.5 0.5B~7B + Qwen3 全系 19 配置实测，λ 注入公式四级归一化自动适配"),
        ("百炼 MaaS 情感增强 API", "「情感增强接口」封装为按 Token 计费能力，随 100 万企业客户分发"),
    ],
    "未来能用": [
        ("通义 App / 小酒窝情感底座", "数字人「能谈心」的落地引擎——直击留存痛点"),
        ("百炼 Agent Store / 魔搭生态", "上架情感增强模型服务，走官方客户案例复制路径"),
    ],
    "怎么接入": "Qwen 全系原生适配（hidden_dim×架构族×量化四级归一化），百炼/魔搭上架或 API 服务接入；1.5B 级轻量 GPU 可跑，单次对话成本约为大模型 API 的 1/30~1/300。",
    "怎么收益": "API 按 Token 计费、企业客户分发、通义 App 留存与数字人差异化。",
    "数据行": [["19 配置对照", "13 有效", "Qwen 全系回响有效（熵 +40~45%）"],
              ["Qwen3-4B", "熵 +51%", "λ=0.098 未坍缩"],
              ["未注册兜底", "命中 0.2166", "任意新模型即插即用"],
              ["单次成本", "1/30~1/300", "1.5B 级轻量 GPU"]],
},
"腾讯": {
    "en": "Tencent", "副标题": "混元 / 元宝 / 微信 AI 智能体「人味引擎」",
    "现状": [
        ["元宝翻车", "AI 生成拜年海报出现辱骂文字，多次致歉", "重庆日报/新浪财经"],
        ["QQ AI 体验", "群聊 AI 机器人被实测吐槽「太假、太能装」", "AI锄客实测"],
        ["战略定调", "马化腾：「AI 要有活人感」，微信落地 AI 智能体", "光明网/澎湃"],
    ],
    "诊断": "元宝 AI 生成辱骂文字翻车多次致歉；QQ 群 AI 机器人被实测吐槽「太假、太能装」；「元宝辜负了腾讯」增长失速；马化腾公开定调「AI 要有活人感」，微信将落地 AI 智能体——战略方向明确，缺的是落地引擎。",
    "现在能用": [
        ("元宝对话情感增强", "直击元宝体验短板与「活人感」战略缺口"),
        ("QQ 伙伴 / AI 聊天搭子", "复活 QQ 宠物、AI 聊天搭子测试中——角色陪伴需要情感一致性"),
    ],
    "未来能用": [
        ("微信 AI 智能体情感底座", "马化腾定调的「活人感」在微信生态的工程化落地"),
        ("GiiNEX 游戏 NPC", "元梦之星 UGC + 游戏 AI 队友——1.5B 高效小参数契合游戏内低成本实时对话"),
    ],
    "怎么接入": "腾讯云 TI 平台挂载 / 游戏 AI 引擎接入；零权重插件即插即用，1.5B 契合高效小参数路线，显存零增加。",
    "怎么收益": "元宝会员/广告转化提升、游戏内购与 UGC 活跃、MaaS 差异化（对标豆包 3.4 亿月活的体验差距）。",
    "数据行": [["P6 综合", "0.7046", "裸 0.5800 · +21.5%"],
              ["LLM-Judge 胜率", "+122%", "0.30 → 0.667"],
              ["情感推理质量", "0.78", "P1~P5 均 0.56"],
              ["峰值显存", "3.78GB", "与裸持平"]],
},
"字节跳动": {
    "en": "ByteDance", "副标题": "猫箱 / 豆包角色功能「人味补丁」",
    "现状": [
        ["猫箱抱怨", "句式重复、人设同质化（「无论什么人设都感觉一样」）", "小红书氪金用户"],
        ["OOC 崩坏", "「猫箱人设完全崩了」等高频投诉", "小红书"],
        ["模型波动", "「还我十天前的模型」「语言模型又变了」", "小红书"],
        ["收费压力", "「用户越多，字节越穷」，豆包开始收费抽佣", "36氪/界面"],
    ],
    "诊断": "猫箱用户抱怨高度集中：句式重复、人设同质化（「无论什么人设都感觉一样」）、OOC 崩坏（「猫箱人设完全崩了」）、模型反复更换（「还我十天前的模型」）；豆包收费压力（「用户越多，字节越穷」）——需要不换模型、不加成本的体验升级。",
    "现在能用": [
        ("猫箱角色功能情感增强", "不换模型只增强情感——直击「模型反复更换导致体验波动」的痛点"),
        ("OOC 抑制", "P6 OQC 角色漂移拉回锚点 + 身份暴露硬拦截，专治人设崩坏"),
    ],
    "未来能用": [
        ("火山引擎 MaaS 情感增强服务", "对外能力出口，按 Token 计费"),
        ("扣子 Coze 插件生态", "情感增强插件进入扣子生态，开发者即插即用"),
        ("AI 乙女方向", "智能体退潮后字节转向 AI 恋爱陪伴——情感一致性是刚需"),
    ],
    "怎么接入": "零权重插件与底层模型解耦——豆包/猫箱换模型不影响情感增强层；火山/扣子平台化接入；显存零增加、吞吐 -2%，契合降本增效窗口。",
    "怎么收益": "猫粮付费留存提升、日均时长（125 分钟）进一步变现、MaaS 计费、角色差异化付费。",
    "数据行": [["角色一致性", "std=0.0", "三轮种子完全一致"],
              ["LLM-Judge 胜率", "+122%", "0.30 → 0.667"],
              ["P6 综合", "0.7046", "裸 0.5800 · +21.5%"],
              ["AI 平均分", "3.73/5", "真人基线 3.9"]],
},
}

# ══════════════════════════════════════════════════════════════
# 三、每家厂商专属柔和配色（浅色背景 + 居中主色，打印观感好）
# ══════════════════════════════════════════════════════════════
THEMES = {
    "米哈游":   {"名称": "星际浅紫", "bg": "#F5F0FC", "bg2": "#EBE2F8", "primary": "#6B5BC9", "headbg": "#E9E2F7", "text_primary": "#4A3FA0"},
    "深度求索": {"名称": "深海浅蓝", "bg": "#EEF2FE", "bg2": "#E0E9FC", "primary": "#3D5AF0", "headbg": "#E1E9FB", "text_primary": "#2B3FA0"},
    "阿里云":   {"名称": "云霞浅橙", "bg": "#FDF4EC", "bg2": "#F9E7D6", "primary": "#D96A1F", "headbg": "#F7E6D6", "text_primary": "#A84F12"},
    "腾讯":     {"名称": "冰川浅青", "bg": "#EDF6F8", "bg2": "#DCEEF2", "primary": "#1B8AC9", "headbg": "#DCEDF3", "text_primary": "#0F6FA0"},
    "字节跳动": {"名称": "豆包暖金", "bg": "#FEF8E7", "bg2": "#FAEEC9", "primary": "#D9A400", "headbg": "#F7ECC9", "text_primary": "#A87F00"},
}

# 通用备选样式（--style 指定时用，颜色居中柔和）
GENERIC_STYLES = {
    "黑白":   {"bg": "#FFFFFF", "bg2": "#F2F2F2", "primary": "#000000", "headbg": "#E8E8E8", "text_primary": "#000000"},
    "商务蓝": {"bg": "#F4F8FC", "bg2": "#E8F0F8", "primary": "#1A5C8E", "headbg": "#E2ECF5", "text_primary": "#12405F"},
    "墨绿":   {"bg": "#F4F8F5", "bg2": "#E6F0EA", "primary": "#2E6B4F", "headbg": "#E0ECE5", "text_primary": "#1D4A36"},
}

CSS_BASE = """
@page {{ size: A4; margin: 1.6cm 1.8cm; @bottom-center {{ content: "{footer} · {主题名} | 第 " counter(page) " / " counter(pages) " 页"; font-size: 7.5pt; color: #888; }} }}
* {{ box-sizing: border-box; }}
html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ font-family: "Microsoft YaHei","SimHei","SimSun",sans-serif; color: #111; font-size: 10.5pt; line-height: 1.75; margin: 0; background: {bg}; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
/* 封面：上下主色条 + 中央内容，彩打精致 */
.cover {{ border: 2pt solid {primary}; background: {bg2}; padding: 0; text-align: center; margin-bottom: 14pt; page-break-after: always; position: relative; }}
.cover .bar-t {{ height: 10pt; background: {primary}; }}
.cover .inner {{ padding: 1.1cm 1cm 1.3cm; }}
.cover .brand {{ font-size: 11pt; letter-spacing: 8px; font-weight: bold; color: {text_primary}; }}
.cover .rule {{ width: 3cm; height: 3pt; background: {primary}; margin: 10pt auto; }}
.cover h1 {{ font-size: 22pt; font-weight: bold; color: {text_primary}; margin: 8pt 0 6pt; }}
.cover .sub {{ font-size: 12pt; color: #333; }}
.cover .meta {{ font-size: 9pt; color: #666; margin-top: 16pt; }}
.cover .bar-b {{ height: 10pt; background: {primary}; }}
h2 {{ font-size: 13pt; font-weight: bold; color: {text_primary}; border-bottom: 2pt solid {primary}; padding-bottom: 3pt; margin: 14pt 0 7pt; page-break-after: avoid; }}
h2 .no {{ display: inline-block; background: {primary}; color: #fff; font-size: 10pt; padding: 1pt 7pt; margin-right: 6pt; border-radius: 2pt; }}
h3 {{ font-size: 11pt; font-weight: bold; color: {text_primary}; margin: 9pt 0 4pt; page-break-after: avoid; }}
p {{ margin: 5pt 0; text-align: justify; }}
table {{ border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 9pt; }}
th {{ background: {primary}; color: #fff; border: 0.75pt solid {primary}; padding: 4pt 6pt; text-align: center; font-weight: bold; }}
td {{ border: 0.5pt solid #aaa; padding: 3pt 6pt; }}
tr:nth-child(even) td {{ background: {headbg}; }}
tr {{ page-break-inside: avoid; }}
.item {{ border: 1pt solid #bbb; border-left: 3.5pt solid {primary}; background: {bg}; padding: 6pt 9pt; margin: 6pt 0; page-break-inside: avoid; }}
.item b {{ font-weight: bold; color: {text_primary}; }}
.item .tag {{ display: inline-block; background: {headbg}; border: 0.75pt solid {primary}; font-size: 8pt; font-weight: bold; color: {text_primary}; padding: 1pt 5pt; margin-right: 6pt; }}
blockquote {{ border-left: 3pt solid {primary}; background: {bg2}; padding: 6pt 9pt; color: #333; margin: 6pt 0; }}
.big {{ text-align: center; margin: 6pt 0; }}
.big .cell {{ display: inline-block; width: 23%; vertical-align: top; border: 1pt solid #bbb; border-top: 3pt solid {primary}; background: {bg}; margin: 0 1%; padding: 6pt 2pt; page-break-inside: avoid; }}
.big .n {{ font-size: 15pt; font-weight: bold; color: {text_primary}; }}
.big .l {{ font-size: 8.5pt; color: #444; }}
.big .d {{ font-size: 7.5pt; color: #666; }}
.price {{ border: 1.5pt solid {primary}; background: {bg2}; padding: 8pt 10pt; margin: 7pt 0; }}
.price b {{ font-weight: bold; color: {text_primary}; }}
.contact {{ text-align: center; border-top: 2pt solid {primary}; color: {text_primary}; margin-top: 12pt; padding-top: 7pt; font-weight: bold; font-size: 11pt; }}
"""

# ══════════════════════════════════════════════════════════════
# 四、渲染（内容唯一，样式不同）
# ══════════════════════════════════════════════════════════════
def 页面(主体, css):
    return f"<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><style>{css}</style></head><body>{主体}</body></html>"

def 渲染(厂商, C, 主题):
    P = []
    P.append(f"""<div class="cover">
<div class="bar-t"></div>
<div class="inner">
<div class="brand">{厂商} × SEMANTIC ECHO</div>
<div class="rule"></div>
<h1>语义回响商业合作提案</h1>
<div class="sub">{C['副标题']} ｜ 配色：{主题['名称']}</div>
<div class="meta">091635Aa 商业化推进系统 ｜ {date.today().isoformat()} ｜ 洽谈：DYPUBG2025@QQ.COM</div>
</div>
<div class="bar-b"></div>
</div>""")
    # ① 行业问题
    P.append('<h2><span class="no">1</span>行业问题：为什么「人味」是当前胜负手</h2>')
    P.append("".join(f'<div class="item"><span class="tag">{i+1}</span><b>{t}</b>：{d}</div>' for i, (t, d) in enumerate(行业问题)))
    # ② 项目介绍
    P.append('<h2><span class="no">2</span>项目介绍：语义回响（Semantic Echo）</h2>')
    P.append("".join(f'<div class="item"><span class="tag">{i+1}</span><b>{t}</b>：{d}</div>' for i, (t, d) in enumerate(项目介绍)))
    P.append('<h3>技术家族（P1~P6 五代推理期增强）</h3>')
    P.append('<table><tr><th>代号</th><th>操作空间</th><th>作用</th></tr>' + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in 技术家族) + "</table>")
    P.append('<h3>全流程 7 模式评测（2026 最终版，同种子对照）</h3>')
    P.append('<table><tr><th>模式</th><th>HeartBench</th><th>HEART-BENCH</th><th>LLM-Judge</th><th>TuringBench</th><th>EmoCharacter</th><th>综合</th></tr>' + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td><td>{f}</td><td><b>{g}</b></td></tr>" for a, b, c, d, e, f, g in 图灵数据) + "</table>")
    # ③ 对症下药
    P.append('<h2><span class="no">3</span>对症下药：面向贵司的方案</h2>')
    P.append('<h3>贵司现状（公开信息可溯源）</h3>')
    P.append('<table><tr><th>维度</th><th>公开事实</th><th>来源</th></tr>' + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in C["现状"]) + "</table>")
    P.append(f'<h3>诊断：痛点定位</h3><blockquote>{C["诊断"]}</blockquote>')
    P.append('<h3>现在就能用到的商业项目</h3>' + "".join(f'<div class="item"><b>{t}</b>：{d}</div>' for t, d in C["现在能用"]))
    P.append('<h3>未来可以接入的商业项目</h3>' + "".join(f'<div class="item"><b>{t}</b>：{d}</div>' for t, d in C["未来能用"]))
    P.append(f'<h3>怎么接入</h3><p>{C["怎么接入"]}</p>')
    P.append(f'<h3>怎么产生收益</h3><p>{C["怎么收益"]}</p>')
    P.append('<h3>关键实测数据</h3>')
    P.append('<div class="big">' + "".join(f'<div class="cell"><div class="n">{n}</div><div class="l">{l}</div><div class="d">{d}</div></div>' for l, n, d in C["数据行"]) + "</div>")
    # ④ 复现
    P.append('<h2><span class="no">4</span>怎么复现验证（实验逻辑，简单版）</h2>')
    P.append("".join(f'<div class="item"><span class="tag">{i+1}</span><b>{t}</b>：{d}</div>' for i, (t, d) in enumerate(复现逻辑)))
    P.append(f'<h3>技术原理详述（供技术评审）</h3><p>{技术原理详述}</p>')
    P.append('<h3>多模型泛化（LLM-Judge 对真人胜率）</h3>')
    P.append('<table><tr><th>模型</th><th>裸 → P6</th><th>提升</th></tr>' + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in 泛化数据) + "</table>")
    P.append('<h3>性能开销（1.5B 实测）</h3>')
    P.append('<table><tr><th>模式</th><th>吞吐</th><th>峰值显存</th><th>额外开销</th></tr>' + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in 性能数据) + "</table>")
    P.append('<h3>开源仓库清单（可溯源）</h3>' + "".join(f'<p>· {s}</p>' for s in 开源清单))
    P.append('<h3>合作流程</h3>' + "".join(f'<div class="item"><b>第{i+1}步</b>：{s}</div>' for i, s in enumerate(合作流程)))
    P.append('<h3>项目背景</h3>' + "".join(f'<p>· {s}</p>' for s in 项目背景))
    # ⑤ 价格
    P.append('<h2><span class="no">5</span>合作方式与参考价格</h2>')
    P.append('<div class="price">' + "".join(f'<p><b>{t}</b>：{d}</p>' for t, d in 价格表) + '<p style="margin-top:6pt;font-size:8.5pt;color:#666">* 参考价格，最终以双方签订合同为准；厂商专属优惠与实物条款面议。</p></div>')
    P.append(f'<div class="contact">洽谈：DYPUBG2025@QQ.COM ｜ 语义回响家族全部开源，欢迎先验证再合作</div>')
    css = CSS_BASE.format(footer=f"{厂商} × Semantic Echo", 主题名=主题["名称"], bg=主题["bg"], bg2=主题["bg2"], primary=主题["primary"], headbg=主题["headbg"], text_primary=主题["text_primary"])
    return 页面("".join(P), css)

# ══════════════════════════════════════════════════════════════
# 五、生成 HTML + PDF
# ══════════════════════════════════════════════════════════════
def 生成PDF(html文本, pdf路径):
    with TemporaryDirectory(prefix="prop_pdf_", ignore_cleanup_errors=True) as 临时:
        临时 = Path(临时)
        html文件 = 临时 / "prop.html"
        html文件.write_text(html文本, encoding="utf-8")
        pdf路径.parent.mkdir(parents=True, exist_ok=True)
        命令 = [EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--user-data-dir={临时 / 'profile'}", f"--print-to-pdf={pdf路径}", html文件.as_uri()]
        proc = subprocess.Popen(命令, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            if pdf路径.exists() and pdf路径.stat().st_size > 0:
                break
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        else:
            proc.kill(); return False
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        time.sleep(0.8)
        return pdf路径.exists() and pdf路径.stat().st_size > 0

def 页数(pdf路径):
    try:
        m = re.search(rb"/Count (\d+)", pdf路径.read_bytes())
        return int(m.group(1)) if m else 0
    except Exception:
        return 0

def 主():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    parser.add_argument("--style", default=None, choices=["黑白", "商务蓝", "墨绿"], help="用通用配色覆盖每家专属配色")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    html_dir = BASE / "提案" / "HTML"
    pdf_dir = BASE / "提案" / "PDF"
    html_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    厂商们 = list(CONTENT) if not args.only else [s.strip() for s in args.only.split(",")]

    print("=== 091635Aa 提案生成器 V4（每家专属柔和配色 · 浅色背景打印友好）===")
    for 厂商 in 厂商们:
        C = CONTENT[厂商]
        主题 = GENERIC_STYLES[args.style] if args.style else THEMES[厂商]
        后缀 = f"_{args.style}" if args.style else ""
        print(f"\n[GEN] {厂商}（{C['副标题']} · 配色：{主题['名称']}）")
        html文本 = 渲染(厂商, C, 主题)
        html路径 = html_dir / f"{厂商}_提案{后缀}.html"
        html路径.write_text(html文本, encoding="utf-8")
        print(f"  HTML: {html路径.name}")
        if not args.no_pdf:
            pdf路径 = pdf_dir / f"{厂商}_提案{后缀}.pdf"
            if 生成PDF(html文本, pdf路径):
                print(f"  PDF OK: {pdf路径.name} ({页数(pdf路径)} 页)")
            else:
                print(f"  PDF FAIL: {pdf路径.name}")
    print("\n=== 完成 ===")

if __name__ == "__main__":
    主()
