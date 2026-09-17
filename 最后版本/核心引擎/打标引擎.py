# -*- coding: utf-8 -*-
"""
打标引擎模块
============
负责对分割片段自动打标（情感维度/情感标签/内容标签/风格标签），
支持人工复核、批量打标与微调数据包导出（Task 5）。

- 自动打标：双层策略 —— cnsenti 库可用时以其 pos/neg 统计判定情感维度
  （正向率>0.5 判"积极"，负向率>0.5 判"消极"，否则"中性"），同时用内置
  中文情感词表（模块内硬编码，约 40 个情感词）匹配文本提取具体情感标签；
  cnsenti 缺失时仅用内置情感词表 + 简单正负向词统计降级判定；
- 情感强度：(pos+neg)/(pos+neg+0.001)，取值 0~1；
- 内容标签：jieba 分词（缺失时降级为连续中文片段双字滑窗），取 TF 最高
  的 2~5 个词作为主题关键词；
- 风格标签：规则检测（口语化/真诚/温柔/活泼/严肃），无命中给"中性"兜底；
- 全部标签去重，并按配置 打标.打标维度 只保留配置启用的维度；
- 人工复核：按 片段ID 搜索 数据/打标结果/*_打标结果.json，用传入标签
  覆盖并标记 人工修正 后写回；
- 批量打标：扫描 数据/分割片段/*_话题分割.json，逐文件自动打标并汇总，
  结果落盘 数据/打标结果/<任务ID>_打标结果.json；
- 导出数据包：把打标片段转成 instruction/response 微调格式，输出到
  数据/微调数据包/，并同步生成 汇总.json（条数、各情感分布统计）；
- 注册路由：挂载 FastAPI HTTP 接口（含 BackgroundTasks 后台批量打标与
  进度查询）。

本模块所有外部依赖（cnsenti / jieba / fastapi / pydantic）均按需
try/except 容错降级，缺失时保证核心流程仍可用。
"""

import json
import os
import re
import threading
import time
from collections import Counter

try:
    from 核心引擎.配置管理 import 获取配置项, 解析路径, 项目根
except Exception:
    import sys

    当前目录 = os.path.dirname(os.path.abspath(__file__))
    项目根 = os.path.dirname(当前目录)
    if 项目根 not in sys.path:
        sys.path.insert(0, 项目根)
    from 核心引擎.配置管理 import 获取配置项, 解析路径, 项目根

# ==================================================================
# 常量与全局状态
# ==================================================================

# cnsenti 情感分析库（可用时启用，缺失时降级为内置词表）
try:
    from cnsenti import Sentiment

    _情感对象 = Sentiment()
    _CNSENTI可用 = True
except Exception:
    _情感对象 = None
    _CNSENTI可用 = False

# 全局打标进度缓存：{任务ID: {"百分比": float, "消息": str, "完成": bool}}
打标进度 = {}
_进度锁 = threading.Lock()

# 内置中文情感词表：{情感词: (情感方向, 权重)}，用于提取 情感标签 与降级判定。
# 情感方向：积极 / 消极；权重 0~1 表示情感强烈程度（同时作为降级统计计数）。
_情感词表 = {
    # 积极
    "开心": ("积极", 1.0), "高兴": ("积极", 1.0), "快乐": ("积极", 0.9),
    "愉快": ("积极", 0.9), "兴奋": ("积极", 0.9), "喜悦": ("积极", 1.0),
    "幸福": ("积极", 0.9), "满足": ("积极", 0.7), "满意": ("积极", 0.7),
    "轻松": ("积极", 0.7), "期待": ("积极", 0.8), "惊喜": ("积极", 0.9),
    "欣慰": ("积极", 0.7), "温暖": ("积极", 0.7), "感动": ("积极", 0.8),
    "自豪": ("积极", 0.7), "自信": ("积极", 0.6), "喜欢": ("积极", 0.7),
    "安心": ("积极", 0.6), "爱": ("积极", 0.7),
    # 消极
    "难过": ("消极", 1.0), "伤心": ("消极", 1.0), "悲伤": ("消极", 1.0),
    "痛苦": ("消极", 1.0), "沮丧": ("消极", 0.9), "失望": ("消极", 0.8),
    "愤怒": ("消极", 1.0), "生气": ("消极", 1.0), "恼火": ("消极", 0.9),
    "委屈": ("消极", 0.8), "焦虑": ("消极", 0.8), "担心": ("消极", 0.7),
    "害怕": ("消极", 0.8), "恐惧": ("消极", 1.0), "讨厌": ("消极", 0.8),
    "烦躁": ("消极", 0.7), "疲惫": ("消极", 0.7), "累": ("消极", 0.5),
    "无奈": ("消极", 0.7), "失落": ("消极", 0.8), "郁闷": ("消极", 0.8),
    "压抑": ("消极", 0.8), "后悔": ("消极", 0.8), "嫉妒": ("消极", 0.7),
}

# 风格标签规则：(风格名, 触发词列表)
_风格规则 = (
    ("口语化", ("啊", "呢", "呀", "嘛", "啦")),
    ("真诚", ("其实", "真的", "说实话")),
    ("温柔", ("别怕", "没事", "慢慢")),
    ("活泼", ("哈哈", "耶", "哇")),
    ("严肃", ("必须", "注意", "要记住")),
)

# 内容标签提取时过滤的轻度停用词表
_停用词 = {
    "的", "了", "是", "在", "和", "与", "及", "就", "都", "也", "还", "又", "很",
    "把", "被", "让", "给", "对", "从", "向", "到", "我", "你", "他", "她", "它",
    "我们", "你们", "他们", "她们", "它们", "这", "那", "个", "中", "上", "下",
    "有", "说", "道", "着", "过", "呢", "吗", "吧", "啊", "呀", "哦", "嗯",
    "然后", "就是", "这个", "那个", "一个", "什么", "怎么", "这样", "那样",
    "自己", "起来", "出来", "下去", "因为", "所以", "但是", "可是", "不过",
    "如果", "虽然", "而且", "并且", "或者", "还是", "已经", "正在", "可以",
    "可能", "应该", "觉得", "知道", "没有", "不是", "非常", "特别", "真的",
    "其实", "当然", "今天", "现在", "时候", "一下", "一点", "一样",
}


# ==================================================================
# 内部辅助函数
# ==================================================================


def _生成任务ID() -> str:
    """生成唯一任务ID：时间戳（14位）+ 随机hex（6位），共 20 位。"""
    return time.strftime("%Y%m%d%H%M%S") + __import__("secrets").token_hex(3)


def _取目录(配置路径: str, 默认相对路径: str) -> str:
    """取配置中的目录（绝对路径），缺失/相对时用 解析路径 修正，不存在则创建。"""
    目录 = 获取配置项(配置路径, "")
    if not 目录:
        目录 = 解析路径(默认相对路径)
    elif not os.path.isabs(目录):
        目录 = 解析路径(目录)
    os.makedirs(目录, exist_ok=True)
    return 目录


def _更新进度(任务ID: str, 进度: float, 消息: str) -> None:
    """更新模块级打标进度缓存（线程安全）。"""
    with _进度锁:
        打标进度[任务ID] = {
            "百分比": round(进度, 2),
            "消息": 消息,
            "完成": bool(进度 >= 1.0),
        }


def _清洗文本(文本) -> str:
    """文本清洗：去控制字符、去乱码（U+FFFD）、压缩空白，确保合法 UTF-8。"""
    if 文本 is None:
        return ""
    文本 = str(文本)
    保留 = []
    for 字符 in 文本:
        码点 = ord(字符)
        if 字符 in ("\n", "\t", "\r", "\u3000"):
            保留.append(" ")
        elif 码点 == 0xFFFD or 码点 == 127 or (码点 < 32 and 码点 != 9):
            continue
        else:
            保留.append(字符)
    文本 = "".join(保留)
    文本 = re.sub(r" {2,}", " ", 文本).strip()
    return 文本


def _取首句(文本: str) -> str:
    """取文本首句（按 句号/问号/感叹号/省略号 切分）。"""
    for 句 in re.split(r"[。！？!?…]+", 文本 or ""):
        if 句.strip():
            return 句.strip()
    return (文本 or "").strip()[:20]


def _分词(文本: str) -> list:
    """jieba 分词（过滤停用词）；jieba 缺失时降级为连续中文片段双字滑窗。"""
    try:
        import jieba
        jieba.setLogLevel(20)
        词列表 = [词.strip() for 词 in jieba.cut(文本 or "")]
    except Exception:
        中文段 = re.findall(r"[\u4e00-\u9fff]+", 文本 or "")
        词列表 = []
        for 段 in 中文段:
            if len(段) <= 2:
                词列表.append(段)
            else:
                for i in range(len(段) - 1):
                    词列表.append(段[i:i + 2])
    return [
        词 for 词 in 词列表
        if 词 and 词 not in _停用词
        and len(词) >= 2 and re.search(r"[\u4e00-\u9fff]", 词)
    ]


# ==================================================================
# 一、自动打标
# ==================================================================


def _情感分析(文本: str) -> tuple:
    """对单段文本做情感分析（双层策略）。

    a. cnsenti 可用：sentiment_count 统计 pos/neg，正向率>0.5 判"积极"，
       负向率>0.5 判"消极"，否则"中性"；情感强度=(pos+neg)/(pos+neg+0.001)；
       同时用内置情感词表匹配提取具体情感标签；
    b. cnsenti 缺失：仅用内置情感词表匹配 + 正负向词权重统计降级判定。

    返回:
        (情感维度, 情感标签, 情感强度, 置信度)
    """
    正向数 = 0.0
    负向数 = 0.0
    # 1) 内置情感词表匹配：提取具体情感标签 + 降级判定计数
    匹配标签 = []
    for 词, (方向, 权重) in _情感词表.items():
        if 词 in 文本:
            匹配标签.append(词)
            if 方向 == "积极":
                正向数 += 权重
            else:
                负向数 += 权重
    # 2) cnsenti（可用时）以 pos/neg 统计覆盖计数，用于维度判定
    使用cnsenti = _CNSENTI可用
    if 使用cnsenti:
        try:
            统计 = _情感对象.sentiment_count(文本)
            cnsenti正向 = float(统计.get("pos", 0) or 0)
            cnsenti负向 = float(统计.get("neg", 0) or 0)
        except Exception:
            cnsenti正向 = cnsenti负向 = 0.0
        if cnsenti正向 or cnsenti负向:
            正向数 = cnsenti正向
            负向数 = cnsenti负向
        else:
            # 老版本 cnsenti 可能只有 sentiment_calculate（加权分数）
            try:
                统计 = _情感对象.sentiment_calculate(文本)
                计算正向 = float(统计.get("pos", 0) or 0)
                计算负向 = float(统计.get("neg", 0) or 0)
                if 计算正向 or 计算负向:
                    正向数 = 计算正向
                    负向数 = 计算负向
            except Exception:
                pass
    总情感 = 正向数 + 负向数
    情感强度 = round(总情感 / (总情感 + 0.001), 3)
    正向率 = 正向数 / (总情感 + 0.001)
    负向率 = 负向数 / (总情感 + 0.001)
    if 正向率 > 0.5 and 正向率 >= 负向率:
        情感维度 = "积极"
    elif 负向率 > 0.5 and 负向率 > 正向率:
        情感维度 = "消极"
    else:
        情感维度 = "中性"
    # 情感标签去重（按词表遍历顺序，结果稳定）
    情感标签 = []
    for 词 in 匹配标签:
        if 词 not in 情感标签:
            情感标签.append(词)
    # 置信度：cnsenti 可用时略高，降级时略低
    if 使用cnsenti:
        置信度 = round(min(0.98, 0.5 + 0.5 * 情感强度), 2)
    else:
        置信度 = round(min(0.9, 0.4 + 0.5 * 情感强度), 2)
    return 情感维度, 情感标签, 情感强度, 置信度


def _内容标签(文本: str) -> list:
    """分词后取 TF 最高的 2~5 个词作为主题关键词（内容标签）。"""
    词列表 = _分词(文本)
    if not 词列表:
        return []
    频率 = Counter(词列表)
    候选 = [词 for 词, _ in 频率.most_common()]
    if not 候选:
        return []
    if len(候选) <= 1:
        return 候选
    return 候选[:min(5, len(候选))]


def _风格标签(文本: str) -> list:
    """规则检测风格标签；无任何命中时给"中性"兜底。"""
    风格 = []
    for 风格名, 触发词 in _风格规则:
        if any(词 in 文本 for 词 in 触发词):
            if 风格名 not in 风格:
                风格.append(风格名)
    if not 风格:
        风格.append("中性")
    return 风格


def 自动打标(片段列表: list) -> list:
    """对片段列表自动打标（情感维度/情感标签/内容标签/风格标签/情感强度/置信度）。

    参数:
        片段列表: 话题分割 返回的片段列表（每个 dict 含 文本/话题ID/话题摘要 等）。

    返回:
        list，元素为 dict：原片段字段 + 新增打标字段；
        配置 打标.自动打标 为 False 时原样返回未打标片段。
    """
    if not 获取配置项("打标.自动打标", True):
        return [dict(片段) for 片段 in (片段列表 or [])]
    配置维度 = 获取配置项("打标.打标维度", ["情感维度", "内容标签", "风格标签"])
    结果 = []
    for 片段 in 片段列表 or []:
        if not isinstance(片段, dict):
            continue
        文本 = str(片段.get("文本", "") or "")
        情感维度, 情感标签, 情感强度, 置信度 = _情感分析(文本)
        内容标签 = _内容标签(文本)
        风格标签 = _风格标签(文本)
        副本 = dict(片段)
        # 按配置 打标.打标维度 只保留配置启用的维度（情感强度/置信度始终输出）
        if "情感维度" in 配置维度:
            副本["情感维度"] = 情感维度
            副本["情感标签"] = 情感标签
        else:
            副本["情感维度"] = ""
            副本["情感标签"] = []
        if "内容标签" in 配置维度:
            副本["内容标签"] = 内容标签
        else:
            副本["内容标签"] = []
        if "风格标签" in 配置维度:
            副本["风格标签"] = 风格标签
        else:
            副本["风格标签"] = []
        副本["情感强度"] = 情感强度
        副本["置信度"] = 置信度
        结果.append(副本)
    return 结果


# ==================================================================
# 二、人工复核
# ==================================================================


def 人工复核(片段ID: str, 标签: dict, 打标结果列表=None) -> bool:
    """按 片段ID 找到打标记录，用传入标签覆盖并标记 人工修正 后写回。

    参数:
        片段ID: 片段唯一标识，如 "seg_0001"（兼容 话题ID）。
        标签: 人工确认后的标签字典（结构同 自动打标 的元素）。
        打标结果列表: 可选。内存中的打标结果列表（如刚自动打标返回的结果）。
            提供时直接在该列表上修改并自动落盘为新的 打标结果 json 文件；
            不提供时按 片段ID 在 数据/打标结果/*_打标结果.json 中查找。

    返回:
        bool：复核保存成功返回 True，否则 False。
    """
    if not 片段ID or not isinstance(标签, dict):
        return False
    if not 获取配置项("打标.人工复核", True):
        return False

    # ── 方式一：基于内存打标结果列表（未落盘也能复核） ──
    if isinstance(打标结果列表, list):
        for 索引, 片段 in enumerate(打标结果列表):
            if not isinstance(片段, dict):
                continue
            if 片段.get("片段ID") != 片段ID and 片段.get("话题ID") != 片段ID:
                continue
            for 键, 值 in 标签.items():
                片段[键] = 值
            片段["人工修正"] = True
            打标结果列表[索引] = 片段
            # 落盘：<任务ID>_打标结果.json（兼容 批量打标任务 的落盘格式）
            try:
                任务ID = 片段.get("任务ID") or _生成任务ID()
                打标结果目录 = _取目录("打标.打标结果目录", "数据/打标结果")
                os.makedirs(打标结果目录, exist_ok=True)
                路径 = os.path.join(打标结果目录, f"{任务ID}_打标结果.json")
                数据 = {"任务ID": 任务ID, "片段数": len(打标结果列表), "片段列表": 打标结果列表}
                with open(路径, "w", encoding="utf-8") as 文件句柄:
                    json.dump(数据, 文件句柄, ensure_ascii=False, indent=2)
                return True
            except OSError:
                return False
        return False

    # ── 方式二：按落盘文件查找 ──
    打标结果目录 = _取目录("打标.打标结果目录", "数据/打标结果")
    if not os.path.isdir(打标结果目录):
        return False
    for 文件名 in sorted(os.listdir(打标结果目录)):
        if not 文件名.endswith("_打标结果.json"):
            continue
        路径 = os.path.join(打标结果目录, 文件名)
        try:
            with open(路径, "r", encoding="utf-8") as 文件句柄:
                数据 = json.load(文件句柄)
        except Exception:
            continue
        片段列表 = 数据.get("片段列表") if isinstance(数据, dict) else None
        if not isinstance(片段列表, list):
            continue
        for 索引, 片段 in enumerate(片段列表):
            if not isinstance(片段, dict):
                continue
            if 片段.get("片段ID") != 片段ID and 片段.get("话题ID") != 片段ID:
                continue
            # 用传入标签覆盖打标字段
            for 键, 值 in 标签.items():
                片段[键] = 值
            片段["人工修正"] = True
            片段列表[索引] = 片段
            try:
                with open(路径, "w", encoding="utf-8") as 文件句柄:
                    json.dump(数据, 文件句柄, ensure_ascii=False, indent=2)
                return True
            except OSError:
                return False
    return False


# ==================================================================
# 三、批量打标任务
# ==================================================================


def 批量打标任务(片段目录: str = "", 进度回调=None, 任务ID: str = "") -> dict:
    """扫描 片段目录 下所有 *_话题分割.json，逐文件自动打标并汇总落盘。

    参数:
        片段目录: 分割片段目录（绝对路径或相对项目根的路径；空则取配置默认值）。
        进度回调: 可选回调函数 `进度回调(进度: float, 消息: str)`。
        任务ID: 可选，指定任务ID（后台接口用于关联进度）；空则自动生成。

    返回:
        dict：{"任务ID", "片段数", "打标结果路径", "成功"}；失败时含 "错误"。
    """
    任务ID = 任务ID or _生成任务ID()
    if 片段目录:
        if not os.path.isabs(片段目录):
            片段目录 = 解析路径(片段目录)
    else:
        片段目录 = _取目录("数据预处理.分割片段目录", "数据/分割片段")
    if not os.path.isdir(片段目录):
        return {
            "任务ID": 任务ID, "片段数": 0, "打标结果路径": "",
            "成功": False, "错误": f"分割片段目录不存在：{片段目录}",
        }
    文件列表 = sorted(
        文件名 for 文件名 in os.listdir(片段目录)
        if 文件名.endswith("_话题分割.json")
    )
    if not 文件列表:
        return {
            "任务ID": 任务ID, "片段数": 0, "打标结果路径": "",
            "成功": False, "错误": f"未在 {片段目录} 找到任何 *_话题分割.json 文件",
        }
    全部片段 = []
    来源统计 = {}
    for 序号, 文件名 in enumerate(文件列表, 1):
        进度 = (序号 - 1) / len(文件列表)
        消息 = f"正在打标 {文件名}（{序号}/{len(文件列表)}）"
        _更新进度(任务ID, 进度, 消息)
        if 进度回调:
            try:
                进度回调(进度, 消息)
            except Exception:
                pass
        来源任务ID = 文件名.split("_", 1)[0] or 任务ID
        try:
            with open(os.path.join(片段目录, 文件名), "r", encoding="utf-8") as 文件句柄:
                片段列表 = json.load(文件句柄)
        except Exception as 错误:
            print(f"[打标] 读取分割文件失败：{文件名}（{错误}）")
            continue
        if not isinstance(片段列表, list):
            continue
        打标后 = 自动打标(片段列表)
        for 片段 in 打标后:
            片段["来源任务ID"] = 来源任务ID
        全部片段.extend(打标后)
        来源统计[来源任务ID] = 来源统计.get(来源任务ID, 0) + len(打标后)
    if not 全部片段:
        return {
            "任务ID": 任务ID, "片段数": 0, "打标结果路径": "",
            "成功": False, "错误": "所有分割文件均为空或解析失败",
        }
    结果 = {
        "任务ID": 任务ID,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "片段数": len(全部片段),
        "来源文件数": len(文件列表),
        "来源统计": 来源统计,
        "算法": "cnsenti(可用时)+内置情感词表+jieba(可用时)",
        "片段列表": 全部片段,
    }
    打标结果目录 = _取目录("打标.打标结果目录", "数据/打标结果")
    结果路径 = os.path.join(打标结果目录, f"{任务ID}_打标结果.json")
    try:
        with open(结果路径, "w", encoding="utf-8") as 文件句柄:
            json.dump(结果, 文件句柄, ensure_ascii=False, indent=2)
    except OSError as 错误:
        return {
            "任务ID": 任务ID, "片段数": len(全部片段), "打标结果路径": "",
            "成功": False, "错误": f"打标结果写入失败：{错误}",
        }
    _更新进度(任务ID, 1.0, "打标完成")
    if 进度回调:
        try:
            进度回调(1.0, "打标完成")
        except Exception:
            pass
    return {
        "任务ID": 任务ID,
        "片段数": len(全部片段),
        "打标结果路径": 结果路径,
        "成功": True,
    }


# ==================================================================
# 四、导出数据包
# ==================================================================


def _写汇总(数据包目录: str, 输出路径: str, 数据包类型: str,
            本次条目: list, 本次分布: Counter) -> None:
    """汇总全部 情感微调_* 数据包（含本次）的条数与情感分布，写入 汇总.json。"""
    累计条数 = len(本次条目)
    累计分布 = Counter(本次分布)
    try:
        已有文件 = sorted(
            文件名 for 文件名 in os.listdir(数据包目录)
            if 文件名.startswith("情感微调_")
            and 文件名.endswith((".jsonl", ".json"))
        )
        for 文件名 in 已有文件:
            路径 = os.path.join(数据包目录, 文件名)
            if os.path.abspath(路径) == os.path.abspath(输出路径):
                continue
            try:
                with open(路径, "r", encoding="utf-8") as 文件句柄:
                    if 文件名.endswith(".jsonl"):
                        for 行 in 文件句柄:
                            try:
                                数据 = json.loads(行)
                            except Exception:
                                continue
                            if not isinstance(数据, dict):
                                continue
                            情感 = (数据.get("标签") or {}).get("情感维度") or "未知"
                            累计分布[情感] += 1
                            累计条数 += 1
                    else:
                        数据 = json.load(文件句柄)
                        数据列表 = 数据.get("数据") if isinstance(数据, dict) else None
                        for 条目 in (数据列表 or []):
                            if not isinstance(条目, dict):
                                continue
                            情感 = (条目.get("标签") or {}).get("情感维度") or "未知"
                            累计分布[情感] += 1
                            累计条数 += 1
            except Exception:
                continue
    except OSError:
        pass
    汇总 = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "数据包类型": 数据包类型,
        "最近数据包": os.path.basename(输出路径),
        "最近条数": len(本次条目),
        "累计条数": 累计条数,
        "累计情感分布": dict(累计分布),
    }
    try:
        with open(os.path.join(数据包目录, "汇总.json"), "w", encoding="utf-8") as 文件句柄:
            json.dump(汇总, 文件句柄, ensure_ascii=False, indent=2)
    except OSError:
        pass


def 导出数据包(打标结果: list, 格式: str = "jsonl", 数据包类型: str = "情感") -> dict:
    """把打标片段转成微调数据包（instruction/response + 标签），可一键送入微调模块。

    情感数据包每行：{"instruction": "请以{情感维度}的语气回复：{摘要/首句}",
                     "response": "{完整片段文本}", "标签": {...}}。

    参数:
        打标结果: 打标后的片段列表。
        格式: 导出格式，取值 "json" / "jsonl"。
        数据包类型: 数据包类型，默认 "情感"。

    返回:
        dict：{"成功", "路径", "条数", "格式", "数据包类型"}；失败时含 "错误"。
    """
    if 格式 not in ("json", "jsonl"):
        格式 = "jsonl"
    打标结果 = 打标结果 or []
    数据包目录 = _取目录("微调.数据包目录", "数据/微调数据包")
    时间戳 = time.strftime("%Y%m%d_%H%M%S")
    扩展名 = "jsonl" if 格式 == "jsonl" else "json"
    输出路径 = os.path.join(数据包目录, f"情感微调_{时间戳}.{扩展名}")
    条目 = []
    分布 = Counter()
    for 片段 in 打标结果:
        if not isinstance(片段, dict):
            continue
        文本 = _清洗文本(片段.get("文本", ""))
        if not 文本:
            continue
        情感维度 = 片段.get("情感维度") or "自然"
        摘要 = 片段.get("话题摘要") or _取首句(文本)
        instruction = f"请以{情感维度}的语气回复：{摘要}"
        标签 = {}
        for 键 in ("情感维度", "情感标签", "内容标签", "风格标签", "情感强度", "置信度"):
            if 键 in 片段:
                标签[键] = 片段.get(键)
        条目.append({"instruction": instruction, "response": 文本, "标签": 标签})
        分布[情感维度] += 1
    if not 条目:
        return {
            "成功": False, "错误": "打标结果为空或清洗后无有效文本",
            "路径": "", "条数": 0, "格式": 格式, "数据包类型": 数据包类型,
        }
    try:
        with open(输出路径, "w", encoding="utf-8") as 文件句柄:
            if 格式 == "jsonl":
                for 数据 in 条目:
                    文件句柄.write(json.dumps(数据, ensure_ascii=False) + "\n")
            else:
                json.dump({
                    "数据包类型": 数据包类型,
                    "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "条目数": len(条目),
                    "数据": 条目,
                }, 文件句柄, ensure_ascii=False, indent=2)
    except OSError as 错误:
        return {
            "成功": False, "错误": f"数据包写入失败：{错误}",
            "路径": "", "条数": 0, "格式": 格式, "数据包类型": 数据包类型,
        }
    _写汇总(数据包目录, 输出路径, 数据包类型, 条目, 分布)
    return {
        "成功": True,
        "路径": 输出路径,
        "条数": len(条目),
        "格式": 格式,
        "数据包类型": 数据包类型,
    }


# ==================================================================
# 五、HTTP 路由注册
# ==================================================================


def _读取片段目录(片段目录: str) -> list:
    """读取目录下所有 *_话题分割.json 的片段并合并（供 自动打标 接口使用）。"""
    if 片段目录:
        if not os.path.isabs(片段目录):
            片段目录 = 解析路径(片段目录)
    else:
        片段目录 = _取目录("数据预处理.分割片段目录", "数据/分割片段")
    if not os.path.isdir(片段目录):
        return []
    片段列表 = []
    for 文件名 in sorted(os.listdir(片段目录)):
        if not 文件名.endswith("_话题分割.json"):
            continue
        try:
            with open(os.path.join(片段目录, 文件名), "r", encoding="utf-8") as 文件句柄:
                内容 = json.load(文件句柄)
        except Exception:
            continue
        if isinstance(内容, list):
            片段列表.extend(内容)
    return 片段列表


def _后台批量打标(任务ID: str, 片段目录: str) -> None:
    """后台任务：执行批量打标并同步全局 打标进度。"""

    def 进度回调(进度: float, 消息: str) -> None:
        _更新进度(任务ID, 进度, 消息)

    try:
        结果 = 批量打标任务(片段目录, 进度回调=进度回调, 任务ID=任务ID)
        with _进度锁:
            打标进度[任务ID] = {
                "百分比": 1.0 if 结果.get("成功") else 0.0,
                "消息": "打标完成" if 结果.get("成功") else 结果.get("错误", "打标失败"),
                "完成": True,
                "结果": 结果,
            }
    except Exception as 错误:
        with _进度锁:
            打标进度[任务ID] = {"百分比": 0.0, "消息": f"打标异常：{错误}", "完成": True}


def 注册路由(app) -> None:
    """注册打标引擎模块的 HTTP 路由（挂载到 FastAPI 应用）。

    接口:
        POST /api/打标/自动   自动打标（body：片段列表 或 片段目录）
        POST /api/打标/批量   批量打标任务（body：片段目录，后台异步）
        GET  /api/打标/进度   查询批量打标进度（?任务ID=）
        POST /api/打标/复核   人工复核（body：片段ID + 标签）
        GET  /api/打标/结果   读取打标结果（?任务ID=）
        POST /api/打标/导出   导出微调数据包（body：任务ID + 格式 + 数据包类型）
    """
    try:
        from fastapi import BackgroundTasks, Query
        from pydantic import BaseModel
    except Exception as 错误:
        print(f"[打标引擎] 缺少 FastAPI 依赖，跳过路由注册：{错误}")
        return

    class 自动打标请求(BaseModel):
        片段列表: list = []
        片段目录: str = ""

    class 批量打标请求(BaseModel):
        片段目录: str = ""

    class 复核请求(BaseModel):
        片段ID: str
        标签: dict

    class 导出请求(BaseModel):
        任务ID: str
        格式: str = "jsonl"
        数据包类型: str = "情感"

    @app.post("/api/打标/自动")
    def 自动打标接口(请求: 自动打标请求) -> dict:
        try:
            if 请求.片段目录:
                片段列表 = _读取片段目录(请求.片段目录)
            else:
                片段列表 = 请求.片段列表 or []
            打标后 = 自动打标(片段列表)
            return {"成功": True, "片段数": len(打标后), "片段列表": 打标后}
        except Exception as 错误:
            return {"成功": False, "错误": f"自动打标失败：{错误}"}

    @app.post("/api/打标/批量")
    def 批量打标接口(请求: 批量打标请求, 后台任务: BackgroundTasks) -> dict:
        try:
            任务ID = _生成任务ID()
            _更新进度(任务ID, 0.0, "任务已提交，等待后台执行")
            后台任务.add_task(_后台批量打标, 任务ID, 请求.片段目录)
            return {
                "任务ID": 任务ID,
                "状态": "打标中",
                "消息": "批量打标已提交后台执行，可查询进度",
                "成功": True,
            }
        except Exception as 错误:
            return {"成功": False, "错误": f"提交批量打标失败：{错误}"}

    @app.get("/api/打标/进度")
    def 打标进度接口(任务ID: str = Query(...)) -> dict:
        try:
            进度 = 打标进度.get(任务ID)
            if not 进度:
                return {
                    "任务ID": 任务ID, "成功": True,
                    "进度": {"百分比": 0.0, "消息": "未找到该任务的打标记录", "完成": False},
                }
            return {"任务ID": 任务ID, "成功": True, "进度": 进度}
        except Exception as 错误:
            return {"成功": False, "错误": f"查询打标进度失败：{错误}"}

    @app.post("/api/打标/复核")
    def 复核接口(请求: 复核请求) -> dict:
        try:
            成功 = 人工复核(请求.片段ID, 请求.标签)
            return {
                "片段ID": 请求.片段ID,
                "成功": 成功,
                "状态": "已复核" if 成功 else "未找到该片段",
            }
        except Exception as 错误:
            return {"成功": False, "错误": f"人工复核失败：{错误}"}

    @app.get("/api/打标/结果")
    def 打标结果接口(任务ID: str = Query(...)) -> dict:
        try:
            打标结果目录 = _取目录("打标.打标结果目录", "数据/打标结果")
            路径 = os.path.join(打标结果目录, f"{任务ID}_打标结果.json")
            if not os.path.exists(路径):
                return {
                    "任务ID": 任务ID, "成功": False,
                    "错误": f"未找到任务 {任务ID} 的打标结果",
                }
            with open(路径, "r", encoding="utf-8") as 文件句柄:
                数据 = json.load(文件句柄)
            return {"任务ID": 任务ID, "成功": True, "结果": 数据}
        except Exception as 错误:
            return {"成功": False, "错误": f"读取打标结果失败：{错误}"}

    @app.post("/api/打标/导出")
    def 导出接口(请求: 导出请求) -> dict:
        try:
            打标结果目录 = _取目录("打标.打标结果目录", "数据/打标结果")
            路径 = os.path.join(打标结果目录, f"{请求.任务ID}_打标结果.json")
            if not os.path.exists(路径):
                return {"成功": False, "错误": f"未找到任务 {请求.任务ID} 的打标结果"}
            with open(路径, "r", encoding="utf-8") as 文件句柄:
                数据 = json.load(文件句柄)
            片段列表 = 数据.get("片段列表") if isinstance(数据, dict) else []
            return 导出数据包(片段列表, 请求.格式, 请求.数据包类型)
        except Exception as 错误:
            return {"成功": False, "错误": f"导出数据包失败：{错误}"}
