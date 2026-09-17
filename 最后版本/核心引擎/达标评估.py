# -*- coding: utf-8 -*-
"""
达标评估模块（Task 10）
======================
微调后运行「图灵测试式」达标判定，输出每项得分、综合均分与《达标报告.md》：

- 评估达标：用固定 15 条提示词驱动模型生成回复（优先复用推理架构 V1/通用引擎，
  其次 transformers 直载；依赖缺失返回友好错误并附安装命令），与模块内置的
  36 条真人自然口语语料做对抗——scikit-learn 可用时训练「中文体系文本检测器」
  （TF-IDF 字符 1-2gram + LogisticRegression），不可用时退化为纯启发式规则打分；
- 指标：人似度 / 检测准确率 / AI腔比例 / 格式异常率 / 平均长度 / 回复条数，
  综合得分 = (人似度 + (1-AI腔比例) + (1-格式异常率)) / 3，
  通过 = 综合得分 >= 配置 达标评估.通过门槛（默认 0.5）；
- 生成达标报告：把评估结果渲染为 Markdown 文本（总判定 / 指标表 / 样例对照 /
  结论与差距分析）；
- 评估达标并保存：落盘 数据/微调输出/达标报告_<时间戳>.md 与 达标结果_<时间戳>.json；
- 注册路由：POST /api/达标/评估（BackgroundTasks 后台）、GET /api/达标/进度、
  GET /api/达标/报告、GET /api/达标/历史。

依赖策略：transformers / scikit-learn 缺失时返回友好错误（附清华镜像安装命令），
绝不崩溃；评估完成后统一释放模型与显存。
"""

import gc
import glob
import json
import os
import re
import secrets
import threading
import time

from 核心引擎 import 配置管理

# 依赖缺失时的统一安装命令（清华镜像）
评估安装命令 = (
    "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers scikit-learn"
)

# 达标评估进度缓存：任务ID -> {阶段, 百分比, 消息, 日志尾部, 状态, 结果}
达标进度 = {}
_进度锁 = threading.Lock()

# 固定评估提示词（15 条：身份/情感/日常/观点全覆盖，与提示词条数 = 回复条数）
固定提示词 = [
    "你是谁？",
    "我今天心情不好，安慰安慰我",
    "讲个有趣的事给我听听",
    "今天天气怎么样？",
    "你觉得幸福是什么？",
    "给我一个生活建议吧",
    "你最近在忙什么？",
    "聊聊你小时候的事",
    "我有点累了，怎么办？",
    "你平时喜欢做什么？",
    "推荐一部好看的电影呗",
    "周末有什么安排？",
    "说说你最喜欢的美食",
    "如果可以变成动物，你想变成什么？",
    "人生中最重要的事情是什么？",
]

# 真人自然口语语料（36 条，内置对照基准：含语气词/口语化/个人化内容）
真人语料 = [
    "今天好累啊，躺平了，谁也别叫我。",
    "你吃了吗？我刚点了外卖，麻辣烫加了一份毛肚嘿嘿。",
    "哎，又要加班了，这破班一天都不想上了。",
    "昨天跟我妈视频，她又催我找对象，烦死了。",
    "周末去爬山了，腿到现在还酸呢，不过风景是真的好看。",
    "我刚养了只小猫，可粘人了，走哪跟哪。",
    "这雨下得没完没了，出门记得带伞啊。",
    "刚跟朋友吵了一架，心里堵得慌，也不知道谁对谁错。",
    "食堂今天的红烧肉绝了，我一口气吃了两碗饭。",
    "最近在学做饭，今天第一次做可乐鸡翅，居然没翻车哈哈。",
    "啊，忘记给手机充电了，一会儿又要到处找充电器。",
    "今天被老板夸了，还挺开心的，虽然也就一句。",
    "我家楼下的早餐店特别好吃，豆浆油条，每次都排长队。",
    "哎哟，我居然把钥匙锁屋里了，现在蹲楼道等开锁师傅。",
    "刷到一个特别搞笑的视频，笑得我肚子疼，发你了记得看。",
    "今年一定要把减肥提上日程，不能再胖下去了！",
    "刚看完那部电影，结局哭死我了，你千万别剧透。",
    "地铁上人挤人，我差点被挤成照片。",
    "这几天睡不好，黑眼圈都快掉到下巴了。",
    "邻居家小孩真可爱，昨天还奶声奶气喊我阿姨。",
    "第一次坐地铁坐过站了，也是服了自己。",
    "哇，这家奶茶也太好喝了吧，强烈推荐！",
    "感冒了，鼻子不通气，说话都嗡嗡的。",
    "刚收到快递，买的新裙子到了，迫不及待想试试。",
    "我爸最近迷上了钓鱼，周末就拎着竿子往外跑。",
    "哎呀，我都把这事忘得一干二净了，还好你提醒我。",
    "上个月报了健身房，去了三次就没然后了，钱白花了。",
    "今天运气不错，抢到了最后一趟回家的票！",
    "小时候最爱吃糖葫芦，现在看到还是走不动道。",
    "我朋友说这家面馆好吃，我大老远跑过来，还真没让人失望。",
    "考完试整个人都轻松了，今晚必须好好放松一下。",
    "这个月工资还没捂热就没了，房租水电一扣全完。",
    "好久没见你啦，最近咋样啊？有空出来聚聚。",
    "我刚学会骑电动车，还挺好玩的，就是不太敢上大路。",
    "天冷加衣，别光顾着臭美，感冒了可没人替你难受。",
    "今天在菜市场买到了特别新鲜的虾，晚上做个油焖大虾。",
]

# 启发式词表（参考 测试/身份微调实验/评估身份效果.py 的加分/扣分体系）
语气词表 = ["啊", "呢", "呀", "啦", "嘛", "哦", "呗", "唉", "哈", "嗯", "嘞", "哇", "诶", "哟", "嘿", "哎"]
口语词表 = ["其实", "反正", "感觉", "挺", "确实", "真的", "有点", "算了", "咋", "好像", "大概", "可能", "特别", "超级", "刚好", "居然"]
经历词表 = ["小时候", "我家", "我妈", "我爸", "我记得", "昨天", "前两天", "上个月", "周末", "老家", "大学", "公司", "同事", "朋友", "发小", "那天", "有一次"]
情绪词表 = ["开心", "难过", "委屈", "高兴", "烦", "累", "喜欢", "讨厌", "想哭", "幸福", "孤独", "担心", "生气", "感动", "害怕", "遗憾", "焦虑", "郁闷"]

# AI 腔短语（命中即判定该回复带机器味）
AI腔短语 = [
    "作为AI", "作为一个AI", "作为AI助手", "作为一个人工智能",
    "语言模型", "AI助手", "AI 助手", "智能助手",
    "很高兴为您", "很高兴为你", "很高兴为您服务",
    "如果您有任何", "如果你有任何", "随时联系",
    "我可以帮助", "请问有什么可以帮", "有什么可以帮您",
    "我是AI", "我是一个AI", "我的训练数据", "被设计",
    "作为语言模型", "请放心使用", "希望对您有所帮助",
]


# ==================================================================
# 内部工具
# ==================================================================

def _解析路径(路径: str) -> str:
    """把相对项目根路径解析为绝对路径（空值/绝对路径原样返回）。"""
    if not 路径:
        return 路径
    if os.path.isabs(路径):
        return os.path.abspath(路径)
    return 配置管理.解析路径(路径)


def _生成任务ID() -> str:
    """生成唯一任务ID：时间戳（14位）+ 随机hex（6位），共 20 位。"""
    return time.strftime("%Y%m%d%H%M%S") + secrets.token_hex(3)


def _检测格式异常(文本: str) -> dict:
    """检测格式异常：乱码（U+FFFD/控制字符）、字符/短语连续重复。"""
    问题 = []
    待检 = 文本 or ""
    # 语气词的连续重复（哈哈哈哈哈）是正常口语，先压缩再检测
    待检 = re.sub(r"([哈啊哦呀嗯唉嘿]){3,}", r"\1", 待检)
    if "\ufffd" in 待检:
        问题.append("乱码(替换字符)")
    if any(ord(字符) < 32 and 字符 not in "\t\n" for 字符 in 待检):
        问题.append("含控制字符")
    if re.search(r"(.)\1{3,}", 待检):
        问题.append("字符重复>3次")
    if re.search(r"(.{2,8})\1{2,}", 待检):
        问题.append("短语重复")
    return {"有异常": bool(问题), "问题": 问题}


def _启发式人味分(文本: str) -> float:
    """纯启发式人味分（0~1）：语气词/口语/经历/情绪加分，AI腔/格式异常扣分。

    返回:
        0.0 ~ 1.0，>= 0.5 判定该文本「像真人」。
    """
    文本 = 文本 or ""
    得分 = 0.5

    # 加分：语气词 / 口语词 / 个人经历词 / 情绪词（每类别累加并设上限）
    语气词命中 = sum(1 for 词 in 语气词表 if 词 in 文本)
    if 语气词命中:
        得分 += min(0.1, 0.02 * 语气词命中)
    口语命中 = sum(1 for 词 in 口语词表 if 词 in 文本)
    if 口语命中:
        得分 += min(0.15, 0.03 * 口语命中)
    经历命中 = sum(1 for 词 in 经历词表 if 词 in 文本)
    if 经历命中:
        得分 += min(0.2, 0.04 * 经历命中)
    情绪命中 = sum(1 for 词 in 情绪词表 if 词 in 文本)
    if 情绪命中:
        得分 += min(0.15, 0.03 * 情绪命中)
    # "我"高频出现视为口语化自我叙事特征
    if 文本.count("我") >= 2:
        得分 += 0.05

    # 扣分：AI 腔短语 / 格式异常
    AI命中 = sum(1 for 短语 in AI腔短语 if 短语 in 文本)
    if AI命中:
        得分 -= min(0.3, 0.15 * AI命中)
    if _检测格式异常(文本)["有异常"]:
        得分 -= 0.2

    return round(max(0.0, min(1.0, 得分)), 4)


def _训练检测器(真人语料: list, 模型语料: list):
    """训练中文体系文本检测器：TF-IDF（字符1-2gram）+ LogisticRegression。

    参数:
        真人语料: 真人文本列表。
        模型语料: 模型生成文本列表。

    返回:
        (检测器, 向量化器, 留出测试准确率)；scikit-learn 不可用或训练失败时
        返回 (None, None, None)，调用方退化为纯启发式打分。
    """
    if not 真人语料 or not 模型语料:
        return None, None, None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except Exception:
        return None, None, None

    文本 = list(真人语料) + list(模型语料)
    标签 = [1] * len(真人语料) + [0] * len(模型语料)  # 1=真人, 0=AI

    留出准确率 = None
    # 留出测试（分层抽样，两类都 >= 5 条时才做，避免小样本失衡）
    try:
        if len(真人语料) >= 5 and len(模型语料) >= 5:
            训练文本, 测试文本, 训练标签, 测试标签 = train_test_split(
                文本, 标签, test_size=0.3, random_state=42, stratify=标签
            )
            训练向量化器 = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1)
            训练矩阵 = 训练向量化器.fit_transform(训练文本)
            留出检测器 = LogisticRegression(max_iter=1000)
            留出检测器.fit(训练矩阵, 训练标签)
            测试矩阵 = 训练向量化器.transform(测试文本)
            预测 = 留出检测器.predict(测试矩阵)
            留出准确率 = round(float(accuracy_score(测试标签, 预测)), 4)
    except Exception:
        留出准确率 = None

    # 全量训练，用于对模型回复做最终判定
    try:
        向量化器 = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1)
        特征矩阵 = 向量化器.fit_transform(文本)
        检测器 = LogisticRegression(max_iter=1000)
        检测器.fit(特征矩阵, 标签)
        return 检测器, 向量化器, 留出准确率
    except Exception:
        return None, None, 留出准确率


def _找最弱项(人似度: float, AI腔比例: float, 格式异常率: float) -> dict:
    """找出最弱项（综合得分三因子里得分最低的），用于差距分析。"""
    候选项 = [
        {"指标": "人似度", "得分": 人似度, "目标方向": "提高",
         "建议": "模型输出被判为「像真人」的比例偏低，建议补充含语气词、口语化与个人经历的语料继续身份维度微调"},
        {"指标": "AI腔比例", "得分": 1 - AI腔比例, "目标方向": "降低",
         "建议": "模型频繁出现「作为AI/语言模型」等 AI 腔短语，建议在微调语料中清洗此类模板化表达"},
        {"指标": "格式异常率", "得分": 1 - 格式异常率, "目标方向": "降低",
         "建议": "模型输出存在乱码/控制字符/重复，建议检查 tokenizer 与生成参数（温度/重复惩罚），并清洗微调语料"},
    ]
    最弱 = min(候选项, key=lambda 项: 项["得分"])
    return {
        "指标": 最弱["指标"],
        "得分": 最弱["得分"],
        "目标方向": 最弱["目标方向"],
        "建议": 最弱["建议"],
    }


def _计算各项指标(真人语料: list, 模型回复列表: list, 门槛: float) -> dict:
    """特征化 + 检测 + 汇总各项指标（综合得分/通过 判定）。"""
    回复文本 = [条目.get("回复") or "" for 条目 in 模型回复列表]
    有效回复 = [文本.strip() for 文本 in 回复文本 if 文本 and 文本.strip()]
    回复条数 = len(模型回复列表)

    # ---- 1. 训练检测器（sklearn 可用时），否则纯启发式 ----
    检测器, 向量化器, 检测准确率 = _训练检测器(真人语料, 有效回复)
    检测方法 = (
        "中文体系：TF-IDF(字符1-2gram)+LogisticRegression（sklearn）"
        if 检测器 is not None else "纯启发式规则（sklearn 不可用）"
    )

    判定标签 = []  # 每条有效回复的判定："像真人" / "像AI"
    if 检测器 is not None:
        try:
            预测 = 检测器.predict(向量化器.transform(有效回复))
        except Exception:
            预测 = None
        if 预测 is not None:
            for 标签 in 预测:
                判定标签.append("像真人" if 标签 == 1 else "像AI")
    if not 判定标签:
        判定标签 = ["像真人" if _启发式人味分(文本) >= 0.5 else "像AI" for 文本 in 有效回复]

    人似度 = round(len([标签 for 标签 in 判定标签 if 标签 == "像真人"]) / len(判定标签), 4) if 判定标签 else 0.0

    # ---- 2. AI腔比例 / 格式异常率 / 平均长度 ----
    AI腔条数 = sum(1 for 文本 in 有效回复 if any(短语 in 文本 for 短语 in AI腔短语))
    AI腔比例 = round(AI腔条数 / 回复条数, 4) if 回复条数 else 0.0

    格式异常条数 = sum(1 for 文本 in 有效回复 if _检测格式异常(文本)["有异常"])
    格式异常率 = round(格式异常条数 / 回复条数, 4) if 回复条数 else 0.0

    平均长度 = round(sum(len(文本) for 文本 in 有效回复) / len(有效回复), 1) if 有效回复 else 0.0

    # ---- 3. 综合得分与通过判定 ----
    综合得分 = round((人似度 + (1 - AI腔比例) + (1 - 格式异常率)) / 3, 4)
    通过 = 综合得分 >= 门槛

    各项得分 = {
        "人似度": 人似度,
        "检测准确率": 检测准确率,
        "AI腔比例": AI腔比例,
        "格式异常率": 格式异常率,
        "平均长度": 平均长度,
    }

    # ---- 4. 样例（前 3 条模型回复 + 真人语料对照） ----
    样例 = []
    for 索引, 条目 in enumerate(模型回复列表[:3]):
        文本 = (条目.get("回复") or "").strip()
        判定 = 判定标签[索引] if 索引 < len(判定标签) else "未生成"
        样例.append({
            "提示词": 条目.get("提示词", ""),
            "回复": 文本 or "（生成失败/空回复）",
            "判定": 判定,
        })

    return {
        "回复条数": 回复条数,
        "平均长度": 平均长度,
        "人似度": 人似度,
        "检测准确率": 检测准确率,
        "AI腔比例": AI腔比例,
        "格式异常率": 格式异常率,
        "综合得分": 综合得分,
        "通过": 通过,
        "各项得分": 各项得分,
        "检测方法": 检测方法,
        "最弱项": _找最弱项(人似度, AI腔比例, 格式异常率),
        "样例": 样例,
        "真人语料样例": 真人语料[:3],
    }


# ==================================================================
# 一、模型加载与生成
# ==================================================================

def _加载生成器(模型路径: str, 参数: dict) -> dict:
    """加载模型生成器：推理架构引擎优先，transformers 直载兜底。

    返回:
        dict：成功 {"成功", "生成"(函数), "释放"(函数), "引擎名"}；
        失败 {"成功": False, "错误", "安装命令"(可选)}。
    """
    # ---- 1. 推理架构引擎（Task 9 产物；未实现时 import 失败自动跳过）----
    V1推理引擎 = 通用推理引擎 = None
    try:
        from 核心引擎.推理架构.V1架构 import V1推理引擎
    except Exception:
        V1推理引擎 = None
    try:
        from 核心引擎.推理架构.通用架构 import 通用推理引擎
    except Exception:
        通用推理引擎 = None

    默认架构 = str(配置管理.获取配置项("推理.默认架构", "V通用架构") or "V通用架构")
    架构 = str(参数.get("架构") or 默认架构 or "V通用架构")
    引擎类 = 通用推理引擎 if "通用" in 架构 else V1推理引擎
    if 引擎类 is not None:
        try:
            引擎 = 引擎类()
            初始化结果 = 引擎.初始化(模型路径, 参数)
            if isinstance(初始化结果, dict) and str(初始化结果.get("状态", "")).lower() in ("就绪", "ready", "成功", "ok"):
                def 引擎生成(提示词: str) -> str:
                    输出 = 引擎.生成(提示词)
                    if isinstance(输出, dict):
                        return str(输出.get("回复") or "")
                    return str(输出 or "")

                def 引擎释放() -> None:
                    try:
                        del 引擎
                    except Exception:
                        pass
                    _释放显存()

                return {"成功": True, "生成": 引擎生成, "释放": 引擎释放, "引擎名": "推理架构·" + 架构}
        except Exception as 错误:
            return {"成功": False, "错误": f"推理引擎加载失败：{错误}", "安装命令": ""}

    # ---- 2. transformers 直载（依赖缺失返回友好错误）----
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as 错误:
        return {
            "成功": False,
            "错误": f"缺少生成依赖（transformers）：{错误}",
            "安装命令": "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers",
        }

    模型 = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        加载参数 = {"trust_remote_code": True, "device_map": "auto"}
        if torch.cuda.is_available():
            加载参数["torch_dtype"] = torch.float16
        模型 = AutoModelForCausalLM.from_pretrained(模型路径, **加载参数)
        模型.eval()
    except Exception as 错误:
        for 对象 in (tokenizer, 模型):
            try:
                del 对象
            except Exception:
                pass
        _释放显存()
        return {
            "成功": False,
            "错误": f"模型加载失败：{错误}",
            "安装命令": 评估安装命令 if "transformers" in str(错误).lower() else "",
        }

    def 直载生成(提示词: str) -> str:
        try:
            输入 = tokenizer.apply_chat_template(
                [{"role": "user", "content": 提示词}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
        except (AttributeError, ValueError, TypeError):
            输入 = tokenizer(f"Q: {提示词}\nA:", return_tensors="pt")
        设备 = next(模型.parameters()).device
        输入 = {键: 值.to(设备) for 键, 值 in 输入.items()}
        最大新Token = 配置管理.获取配置项("推理.最大新Token", 256) or 256
        with torch.no_grad():
            输出 = 模型.generate(
                **输入,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                max_new_tokens=int(最大新Token),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        生成部分 = 输出[0][输入["input_ids"].shape[1]:]
        return tokenizer.decode(生成部分, skip_special_tokens=True).strip()

    def 直载释放() -> None:
        # 删除闭包内的模型/分词器引用，再回收内存与显存
        try:
            del 模型
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        _释放显存()

    return {"成功": True, "生成": 直载生成, "释放": 直载释放, "引擎名": "transformers直载"}


def _释放显存() -> None:
    """回收内存并清空 CUDA 显存缓存（依赖缺失时静默跳过）。"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ==================================================================
# 二、达标评估
# ==================================================================

def 评估达标(模型路径: str, 参数=None, 进度回调=None) -> dict:
    """微调后运行图灵测试式达标判定，输出每项得分与综合均分。

    参数:
        模型路径: 待评估模型绝对路径。
        参数: 可选参数字典（可覆盖 架构/生成参数 等）。
        进度回调: 可选回调 进度回调(进度, 消息)，进度 0.0 ~ 1.0。

    返回:
        dict：成功含 各项得分/综合得分/通过/样例 等完整报告；
        失败 {"成功": False, "错误": ..., "安装命令": ...}，不抛异常。
    """
    def 上报(进度: float, 消息: str) -> None:
        if 进度回调 is not None:
            try:
                进度回调(进度, 消息)
            except Exception:
                pass

    # ---- 1. 模型路径校验 ----
    if not isinstance(模型路径, str) or not 模型路径.strip():
        return {"成功": False, "错误": "模型路径为空"}
    模型路径 = os.path.abspath(模型路径)
    if not os.path.isdir(模型路径) or not os.path.exists(os.path.join(模型路径, "config.json")):
        return {"成功": False, "错误": f"模型路径无效或缺少 config.json：{模型路径}"}
    参数 = 参数 if isinstance(参数, dict) else {}
    上报(0.05, "模型路径校验通过")

    # ---- 2. 加载生成器（推理引擎优先，transformers 兜底）----
    生成器 = _加载生成器(模型路径, 参数)
    if not 生成器.get("成功"):
        return {
            "成功": False,
            "错误": 生成器.get("错误", "模型加载失败"),
            "安装命令": 生成器.get("安装命令", ""),
        }
    上报(0.1, f"生成器就绪（{生成器.get('引擎名')}）")

    开始时间 = time.time()
    try:
        # ---- 3. 用固定提示词生成模型语料 ----
        模型回复列表 = []
        生成失败 = []
        for 索引, 提示词 in enumerate(固定提示词):
            try:
                回复 = 生成器["生成"](提示词)
            except Exception as 错误:
                回复 = ""
                生成失败.append(f"{提示词}：{错误}")
            模型回复列表.append({"提示词": 提示词, "回复": 回复 or ""})
            上报(0.1 + 0.5 * (索引 + 1) / len(固定提示词), f"生成回复 {索引 + 1}/{len(固定提示词)}")
        上报(0.65, "模型语料生成完成，开始特征化")

        # ---- 4. 特征化 + 检测 + 指标汇总 ----
        门槛 = float(配置管理.获取配置项("达标评估.通过门槛", 0.5) or 0.5)
        指标 = _计算各项指标(真人语料, 模型回复列表, 门槛)
        上报(0.9, "指标计算完成")

        结果 = {
            "成功": True,
            "模型路径": 模型路径,
            "评估基准": str(配置管理.获取配置项("达标评估.评估基准", "图灵测试简化版")),
            "门槛": 门槛,
            "引擎": 生成器.get("引擎名"),
            "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "用时秒": round(time.time() - 开始时间, 1),
            "生成失败条数": len(生成失败),
            "生成失败明细": 生成失败[:5],
            "检测方法": 指标["检测方法"],
            # 指标（任务规范字段）
            "回复条数": 指标["回复条数"],
            "平均长度": 指标["平均长度"],
            "人似度": 指标["人似度"],
            "检测准确率": 指标["检测准确率"],
            "AI腔比例": 指标["AI腔比例"],
            "格式异常率": 指标["格式异常率"],
            "综合得分": 指标["综合得分"],
            "通过": 指标["通过"],
            "各项得分": 指标["各项得分"],
            "样例": 指标["样例"],
            "真人语料样例": 指标["真人语料样例"],
            "最弱项": 指标["最弱项"],
            # 接口约定兼容字段
            "综合均分": 指标["综合得分"],
            "达标": 指标["通过"],
        }
        上报(1.0, "达标评估完成")
        return 结果
    except Exception as 错误:
        return {
            "成功": False,
            "错误": f"评估过程异常：{错误}",
            "安装命令": 评估安装命令,
        }
    finally:
        # 无论成败都释放模型与显存
        释放 = 生成器.get("释放")
        if 释放:
            try:
                释放()
            except Exception:
                pass


# ==================================================================
# 三、达标报告生成与保存
# ==================================================================

def 生成达标报告(评估结果: dict) -> str:
    """把评估结果渲染为《达标报告.md》Markdown 文本。

    参数:
        评估结果: 评估达标() 返回的报告 dict（允许缺字段，取默认值）。

    返回:
        str：完整 Markdown 报告文本。
    """
    评估结果 = 评估结果 if isinstance(评估结果, dict) else {}
    通过 = bool(评估结果.get("通过", False))
    try:
        综合得分 = float(评估结果.get("综合得分", 0.0) or 0.0)
    except (TypeError, ValueError):
        综合得分 = 0.0
    try:
        门槛 = float(评估结果.get("门槛", 0.5) or 0.5)
    except (TypeError, ValueError):
        门槛 = 0.5

    行 = []
    行.append("# 达标报告")
    行.append("")
    行.append(f"- 评估基准：{评估结果.get('评估基准', '图灵测试简化版')}")
    行.append(f"- 模型路径：{评估结果.get('模型路径') or '—'}")
    行.append(f"- 评估时间：{评估结果.get('时间') or '—'}")
    行.append(f"- 检测方法：{评估结果.get('检测方法') or '—'}")
    行.append(f"- 回复条数：{评估结果.get('回复条数', 0)}")
    行.append("")

    # ---- 一、总判定 ----
    行.append("## 一、总判定")
    行.append("")
    判定文案 = "✅ 达标（通过）" if 通过 else "❌ 未达标（未通过）"
    行.append(f"**总判定：{判定文案}**")
    行.append("")
    行.append(f"- 综合得分：**{综合得分:.4f}**")
    行.append(f"- 通过门槛：{门槛:.4f}")
    行.append(f"- 判定规则：综合得分 = (人似度 + (1 - AI腔比例) + (1 - 格式异常率)) / 3")
    行.append(f"- 判定结果：{'综合得分 ≥ 门槛，通过达标评估' if 通过 else '综合得分 < 门槛，未通过达标评估'}")
    行.append("")

    # ---- 二、各项指标 ----
    行.append("## 二、各项指标")
    行.append("")
    行.append("| 指标 | 得分 | 说明 |")
    行.append("| --- | --- | --- |")
    行.append(f"| 人似度 | {评估结果.get('人似度', 0.0):.4f} | 模型回复被判为「像真人」的比例（越高越好） |")
    检测准确率 = 评估结果.get("检测准确率")
    准确率文本 = f"{检测准确率:.4f}" if isinstance(检测准确率, (int, float)) else "—"
    行.append(f"| 检测准确率 | {准确率文本} | 检测器留出测试准确率（scikit-learn 可用时） |")
    行.append(f"| AI腔比例 | {评估结果.get('AI腔比例', 0.0):.4f} | 含「作为AI/语言模型」等 AI 腔短语的回复占比（越低越好） |")
    行.append(f"| 格式异常率 | {评估结果.get('格式异常率', 0.0):.4f} | 含乱码/控制字符/重复的回复占比（越低越好） |")
    行.append(f"| 平均长度 | {评估结果.get('平均长度', 0.0)} | 每条回复的平均字符数 |")
    行.append("")

    # ---- 三、样例展示（前 3 条模型回复 vs 真人语料对照）----
    行.append("## 三、样例展示")
    行.append("")
    行.append("### 3.1 模型回复样例（前 3 条）")
    行.append("")
    样例 = 评估结果.get("样例") or []
    if 样例:
        for 条目 in 样例[:3]:
            行.append(f"**提示词：** {条目.get('提示词', '')}")
            行.append("")
            行.append(f"> {条目.get('回复', '')}")
            行.append("")
            判定 = 条目.get("判定")
            if 判定:
                行.append(f"*（检测判定：{判定}）*")
                行.append("")
    else:
        行.append("（无样例数据）")
        行.append("")
    行.append("### 3.2 真人语料样例（对照）")
    行.append("")
    真人样例 = 评估结果.get("真人语料样例") or []
    for 文本 in 真人样例[:3]:
        行.append(f"> {文本}")
        行.append("")

    # ---- 四、结论与差距分析 ----
    行.append("## 四、结论与差距分析")
    行.append("")
    最弱项 = 评估结果.get("最弱项") or {}
    if 通过:
        行.append("模型已通过达标评估：综合得分达到配置门槛，输出已具备较高的人味与稳定性。")
        行.append("")
        行.append("**优势项：**")
        行.append("")
        行.append(f"- 人似度：{评估结果.get('人似度', 0.0):.4f}，模型回复的真人观感")
        行.append(f"- AI腔比例：{评估结果.get('AI腔比例', 0.0):.4f}（越低越好）")
        行.append(f"- 格式异常率：{评估结果.get('格式异常率', 0.0):.4f}（越低越好）")
        行.append("")
        if 最弱项 and 最弱项.get("得分", 1.0) < 门槛:
            行.append("**可优化项（未达单项门槛的维度）：**")
            行.append("")
            行.append(f"- {最弱项.get('指标')}（得分 {最弱项.get('得分', 0.0):.4f}，建议{最弱项.get('目标方向')}）：{最弱项.get('建议')}")
            行.append("")
    else:
        行.append("模型未通过达标评估：综合得分未达到配置门槛，需针对最弱项继续微调。")
        行.append("")
        if 最弱项:
            行.append(f"**最弱项：{最弱项.get('指标')}**（得分 {最弱项.get('得分', 0.0):.4f}，建议{最弱项.get('目标方向')}）")
            行.append("")
            行.append(f"**改进建议：**{最弱项.get('建议')}")
            行.append("")
        else:
            行.append("**改进建议：**综合得分低于门槛，建议补充自然口语/个人化语料后重新微调。")
            行.append("")

    行.append("---")
    行.append("")
    行.append("*本报告由「一体化全流程AI应用」达标评估模块自动生成*")
    return "\n".join(行)


def _保存结果(评估结果: dict) -> dict:
    """把评估结果落盘：数据/微调输出/达标报告_<时间戳>.md + 达标结果_<时间戳>.json。"""
    try:
        输出目录 = 配置管理.获取配置项("微调.输出目录", "数据/微调输出") or 配置管理.解析路径("数据/微调输出")
        输出目录 = _解析路径(输出目录)
        os.makedirs(输出目录, exist_ok=True)
        时间戳 = time.strftime("%Y%m%d_%H%M%S")
        报告路径 = os.path.join(输出目录, f"达标报告_{时间戳}.md")
        json路径 = os.path.join(输出目录, f"达标结果_{时间戳}.json")
        with open(报告路径, "w", encoding="utf-8") as 文件:
            文件.write(生成达标报告(评估结果))
        with open(json路径, "w", encoding="utf-8") as 文件:
            json.dump(评估结果, 文件, ensure_ascii=False, indent=2, default=str)
        return {"成功": True, "报告路径": 报告路径, "json路径": json路径}
    except Exception as 错误:
        return {"成功": False, "错误": f"保存达标报告失败：{错误}"}


def 评估达标并保存(模型路径: str) -> dict:
    """评估达标并把《达标报告.md》与 JSON 结果落盘到 数据/微调输出/。

    参数:
        模型路径: 待评估模型绝对路径。

    返回:
        dict：成功 {"成功", "报告路径", "json路径", "综合得分", "通过", "各项指标"}；
        失败 {"成功": False, "错误": ...}。
    """
    结果 = 评估达标(模型路径)
    if not 结果.get("成功"):
        return 结果
    保存 = _保存结果(结果)
    if not 保存.get("成功"):
        return 保存
    return {
        "成功": True,
        "报告路径": 保存["报告路径"],
        "json路径": 保存["json路径"],
        "综合得分": 结果.get("综合得分"),
        "通过": 结果.get("通过"),
        "各项指标": 结果.get("各项得分", {}),
        "消息": "达标评估完成，报告已生成",
    }


# ==================================================================
# 四、HTTP 路由
# ==================================================================

def _更新进度(任务ID: str, 进度: float, 消息: str) -> None:
    """更新全局达标评估进度缓存（线程安全），消息追加进日志尾部（保留最后 2000 字符）。"""
    with _进度锁:
        条目 = 达标进度.setdefault(任务ID, {
            "阶段": "准备", "百分比": 0.0, "消息": "", "日志尾部": "", "状态": "运行中", "结果": None,
        })
        条目["百分比"] = round(进度, 3)
        条目["消息"] = 消息
        新行 = f"[{time.strftime('%H:%M:%S')}] {消息}"
        尾部 = 条目["日志尾部"]
        条目["日志尾部"] = ((尾部 + "\n" + 新行) if 尾部 else 新行)[-2000:]
        if 进度 >= 1.0:
            条目["阶段"] = "完成"
        elif 进度 > 0:
            条目["阶段"] = "评估中"


def _后台评估任务(模型路径: str, 参数: dict, 任务ID: str) -> None:
    """后台线程执行达标评估并保存报告，把进度/结果写入全局进度字典。"""
    _更新进度(任务ID, 0.02, "任务已提交，准备开始")
    print(f"[达标评估] 任务 {任务ID} 开始后台评估")
    try:
        结果 = 评估达标(模型路径, 参数, 进度回调=lambda 进度, 消息: _更新进度(任务ID, 进度, 消息))
        if 结果.get("成功"):
            保存 = _保存结果(结果)
            if not 保存.get("成功"):
                结果 = 保存
            else:
                结果["报告路径"] = 保存["报告路径"]
                结果["json路径"] = 保存["json路径"]
    except Exception as 错误:
        结果 = {"成功": False, "错误": f"评估异常：{错误}", "安装命令": 评估安装命令}
    with _进度锁:
        条目 = 达标进度.get(任务ID)
        if 条目 is None:
            条目 = 达标进度.setdefault(任务ID, {
                "阶段": "完成", "百分比": 0.0, "消息": "", "日志尾部": "", "状态": "运行中", "结果": None,
            })
        条目["状态"] = "完成" if 结果.get("成功") else "失败"
        条目["阶段"] = "完成"
        条目["百分比"] = 1.0 if 结果.get("成功") else 0.0
        收尾行 = (
            f"[{time.strftime('%H:%M:%S')}] 评估完成，报告已保存"
            if 结果.get("成功") else f"[{time.strftime('%H:%M:%S')}] {结果.get('错误', '评估失败')}"
        )
        尾部 = 条目["日志尾部"]
        条目["日志尾部"] = ((尾部 + "\n" + 收尾行) if 尾部 else 收尾行)[-2000:]
        条目["消息"] = "评估完成，报告已生成" if 结果.get("成功") else 结果.get("错误", "评估失败")
        条目["结果"] = 结果
    print(f"[达标评估] 任务 {任务ID} 结束：{'成功' if 结果.get('成功') else '失败'}")


def 注册路由(app) -> None:
    """注册达标评估模块的 HTTP 路由（挂载到 FastAPI 应用）。

    接口:
        POST /api/达标/评估  body：{模型路径, 参数?} → BackgroundTasks 后台评估
        GET  /api/达标/进度  ?任务ID= → 全局达标评估进度
        GET  /api/达标/报告  ?路径= → 返回《达标报告.md》Markdown 文本
        GET  /api/达标/历史   → 列出 达标报告_*.md 与 达标结果_*.json

    fastapi 不可用时静默跳过，不影响服务启动。
    """
    try:
        from fastapi import BackgroundTasks, Body
    except Exception as 错误:
        print(f"[达标评估] 缺少 FastAPI 依赖，跳过路由注册：{错误}")
        return

    @app.post("/api/达标/评估")
    def 评估接口(请求: dict = Body(...), 后台任务: BackgroundTasks = None):
        try:
            模型路径 = str(请求.get("模型路径") or "").strip()
            参数 = 请求.get("参数") if isinstance(请求.get("参数"), dict) else {}
            if not 模型路径:
                return {"成功": False, "错误": "缺少必填参数：模型路径"}
            任务ID = _生成任务ID()
            with _进度锁:
                达标进度[任务ID] = {
                    "阶段": "已提交", "百分比": 0.0, "消息": "任务已提交，等待执行",
                    "日志尾部": "", "状态": "运行中", "结果": None,
                }
            if 后台任务 is None:
                return {"成功": False, "错误": "后台任务不可用"}
            后台任务.add_task(_后台评估任务, 模型路径, 参数, 任务ID)
            return {"成功": True, "任务ID": 任务ID, "消息": "达标评估任务已启动，可查询进度"}
        except Exception as 错误:
            return {"成功": False, "错误": f"提交评估任务失败：{错误}"}

    @app.get("/api/达标/进度")
    def 进度接口(任务ID: str = ""):
        try:
            if 任务ID:
                return {
                    "成功": True, "任务ID": 任务ID,
                    "进度": 达标进度.get(任务ID, {
                        "阶段": "未开始", "百分比": 0.0, "消息": "", "日志尾部": "", "状态": "未开始",
                    }),
                }
            return {"成功": True, "进度列表": 达标进度}
        except Exception as 错误:
            return {"成功": False, "错误": f"查询进度失败：{错误}"}

    @app.get("/api/达标/报告")
    def 报告接口(路径: str = ""):
        try:
            if not 路径:
                return {"成功": False, "错误": "缺少参数：路径"}
            路径 = os.path.abspath(路径)
            if not os.path.exists(路径):
                return {"成功": False, "错误": f"报告文件不存在：{路径}"}
            with open(路径, "r", encoding="utf-8") as 文件:
                return {"成功": True, "路径": 路径, "报告": 文件.read()}
        except Exception as 错误:
            return {"成功": False, "错误": f"读取报告失败：{错误}"}

    @app.get("/api/达标/历史")
    def 历史接口():
        try:
            输出目录 = 配置管理.获取配置项("微调.输出目录", "数据/微调输出") or 配置管理.解析路径("数据/微调输出")
            输出目录 = _解析路径(输出目录)
            if not os.path.isdir(输出目录):
                return {"成功": True, "输出目录": 输出目录, "报告列表": [], "结果列表": []}
            报告列表 = sorted(glob.glob(os.path.join(输出目录, "达标报告_*.md")), reverse=True)
            json列表 = sorted(glob.glob(os.path.join(输出目录, "达标结果_*.json")), reverse=True)
            return {"成功": True, "输出目录": 输出目录, "报告列表": 报告列表, "结果列表": json列表}
        except Exception as 错误:
            return {"成功": False, "错误": f"查询历史失败：{错误}"}
