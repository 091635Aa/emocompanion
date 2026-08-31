# -*- coding: utf-8 -*-
"""
日记生成模块（Task 6 · 全新板块）
================================
伪造角色超长期日记（从起始年龄起逐年累积，每篇数百~数千字），
让 AI 模型"觉得自己有真实人生记忆"，日记作为身份/记忆微调的核心文本。

核心能力：
- 教师对话：连接配置的本地大模型（配置→模型→教师模型，如 30B 千问），
  模块级单例缓存，重复调用不重复加载；未配置/依赖缺失返回友好错误不崩溃；
- 规划时间线：教师模型规划人生大事 → 失败降级为内置启发式（逐年分段主题）；
- 生成单篇日记：教师模型按人设+素材写某年龄日记 → 失败降级为内置模板；
- 生成日记：时间线 → 逐年生成 → 汇总落盘 数据/日记/<角色名>_日记.json +
  每篇独立 .md，幂等增量（参数签名未变时只补缺失年份）；
- 审阅日记：把人工修改写回 json 与 md；
- 导出微调文本：全部日记转为身份/记忆微调 jsonl 数据包；
- 注册路由：挂载 FastAPI HTTP 接口（后台生成 + 进度查询）。

约束：
- 所有外部依赖（transformers / torch / fastapi / pydantic）按需
  try/except 容错降级，缺失时保证核心流程仍可用；
- 进度回调约定：`进度回调(进度: float, 消息: str)`，进度取值 0.0 ~ 1.0。
"""

import json
import os
import re
import threading
import time

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

# 教师模型模块级缓存（全局单例，重复调用不重复加载）
教师模型实例 = None          # AutoModelForCausalLM 实例
教师tokenizer = None         # AutoTokenizer 实例
教师模型设备 = "cpu"          # 当前设备（cuda:0 / cpu）
教师模型路径缓存 = ""         # 已加载模型路径（防止重复加载）

# 全局日记生成进度：任务ID -> {"进度": float, "消息": str, "状态": str, "结果": list|None}
日记进度 = {}
_进度锁 = threading.Lock()

# 从打标/分割数据中提取"上下文素材"时最多拼接的片段条数
素材默认条数 = 20

# 教师模型对话特殊 token（模型输出残留时清洗掉）
_特殊标记 = (
    "<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>", "<s>",
    "<pad>", "[UNK]", "[PAD]", "<|end|>", "<|assistant|>",
)

# Windows 文件名非法字符（汇总/单篇落盘时替换）
_非法文件名字符 = re.compile(r'[\\/:*?"<>|\r\n]')

_教师模型未配置提示 = "未配置教师模型（配置→模型→教师模型）"
_教师模型缺依赖提示 = "缺少 transformers，请运行: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers"


# ==================================================================
# 内部辅助函数
# ==================================================================


def _取整数值(值, 默认值=None):
    """把任意输入稳健地转为整数；失败返回默认值。"""
    try:
        if 值 is None or 值 == "":
            return 默认值
        return int(float(值))
    except (TypeError, ValueError):
        return 默认值


def _计字数(文本: str) -> int:
    """统计有效字数（不计空白字符）。"""
    return len([字符 for 字符 in str(文本 or "") if not 字符.isspace()])


def _安全文件名(名称) -> str:
    """把角色名等转为安全的 Windows 文件名（替换非法字符）。"""
    名称 = str(名称 or "未命名角色").strip()
    return _非法文件名字符.sub("_", 名称) or "未命名角色"


def _阶段主题(年龄: int) -> str:
    """按年龄返回人生阶段主题（时间线/标题/降级模板共用）。"""
    if 年龄 <= 6:
        return "幼年时光"
    if 年龄 <= 12:
        return "童年小学"
    if 年龄 <= 15:
        return "中学岁月"
    return "青春成长"


def _规范化人设(人设: dict) -> dict:
    """补齐人设缺失字段并规范化类型，保证后续逻辑不因缺键崩溃。"""
    人设 = dict(人设 or {})
    出生年份 = _取整数值(人设.get("出生年份"), 2008)
    if 出生年份 is None:
        出生年份 = 2008
    return {
        "姓名": str(人设.get("姓名") or "未命名角色").strip(),
        "性别": str(人设.get("性别") or "未知"),
        "出生年份": 出生年份,
        "出生地": str(人设.get("出生地") or "家乡的老巷子"),
        "家庭": str(人设.get("家庭") or "家人"),
        "性格": str(人设.get("性格") or "开朗"),
        "人设描述": str(人设.get("人设描述") or ""),
        "关键经历": str(人设.get("关键经历") or ""),
    }


def _合并参数(参数=None) -> dict:
    """合并配置默认值 + 用户覆盖参数，返回完整生成参数字典（类型已归一）。"""
    默认 = {
        "起始年龄": _取整数值(获取配置项("日记.起始年龄", 7), 7),
        "当前年龄": _取整数值(获取配置项("日记.当前年龄", 18), 18),
        "每篇字数最小": _取整数值(获取配置项("日记.每篇字数最小", 300), 300),
        "每篇字数最大": _取整数值(获取配置项("日记.每篇字数最大", 2000), 2000),
        "日记风格": str(获取配置项("日记.日记风格", "口语化温暖") or "口语化温暖"),
        "生成数量上限": _取整数值(获取配置项("日记.生成数量上限", 100), 100),
    }
    合并 = dict(默认)
    for 键, 值 in (参数 or {}).items():
        if 键 in 合并 and 值 is not None:
            合并[键] = 值
    for 键 in ("起始年龄", "当前年龄", "每篇字数最小", "每篇字数最大", "生成数量上限"):
        合并[键] = _取整数值(合并[键], 默认[键]) or 默认[键]
    # 字数下限/上限互斥修正
    if 合并["每篇字数最小"] > 合并["每篇字数最大"]:
        合并["每篇字数最大"] = 合并["每篇字数最小"]
    if 合并["当前年龄"] < 合并["起始年龄"]:
        合并["当前年龄"] = 合并["起始年龄"]
    return 合并


def _参数签名(参数=None) -> str:
    """把合并后的生成参数序列化为稳定签名，用于幂等增量判断。"""
    return json.dumps(_合并参数(参数), ensure_ascii=False, sort_keys=True)


def _日记目录() -> str:
    """读取配置项 日记.日记目录 并解析为绝对路径（不存在则创建）。"""
    目录 = 获取配置项("日记.日记目录", "数据/日记")
    if not 目录:
        目录 = "数据/日记"
    if not os.path.isabs(目录):
        目录 = 解析路径(目录)
    os.makedirs(目录, exist_ok=True)
    return 目录


def _日记汇总路径(角色名: str) -> str:
    """汇总 JSON 路径：数据/日记/<角色名>_日记.json。"""
    return os.path.join(_日记目录(), f"{_安全文件名(角色名)}_日记.json")


def _单篇md路径(角色名: str, 年龄: int, 年份: int) -> str:
    """单篇 Markdown 路径：数据/日记/<角色名>_<年龄>岁_<年份>.md。"""
    return os.path.join(_日记目录(), f"{_安全文件名(角色名)}_{年龄}岁_{年份}.md")


def _读取已有日记(汇总路径: str) -> tuple:
    """读取已有汇总 JSON，返回 (日记列表, 参数签名)；文件缺失/损坏返回 ([], "")。"""
    if not os.path.exists(汇总路径):
        return [], ""
    try:
        with open(汇总路径, "r", encoding="utf-8") as 文件:
            数据 = json.load(文件)
        return (
            list(数据.get("日记列表") or []),
            str(数据.get("元数据", {}).get("参数签名") or ""),
        )
    except Exception:
        return [], ""


def _写单篇md(角色名: str, 单篇: dict) -> bool:
    """把单篇日记写成独立 .md 文件（供人工审阅）。"""
    路径 = _单篇md路径(角色名, 单篇.get("年龄"), 单篇.get("年份"))
    内容 = (
        f"# {单篇.get('标题')}\n\n"
        f"- 年龄：{单篇.get('年龄')} 岁\n"
        f"- 年份：{单篇.get('年份')} 年\n"
        f"- 字数：{单篇.get('字数')}\n"
        f"- 来源：{单篇.get('来源')}\n"
        f"- 状态：{单篇.get('状态')}\n\n"
        f"---\n\n{单篇.get('正文')}\n"
    )
    try:
        with open(路径, "w", encoding="utf-8") as 文件:
            文件.write(内容)
        return True
    except OSError:
        return False


# ------------------------------------------------------------------
# 文本后处理（清洗控制字符 / 去乱码 / 字数控制）
# ------------------------------------------------------------------


def _后处理正文(文本) -> str:
    """清洗生成文本：去特殊 token、乱码字符、控制字符，统一换行与标点。"""
    文本 = str(文本 or "")
    for 标记 in _特殊标记:
        文本 = 文本.replace(标记, "")
    # 去乱码替换符与空字符
    文本 = 文本.replace("\ufffd", "").replace("\u0000", "")
    保留 = []
    for 字符 in 文本:
        if 字符 in ("\n", "\t", "\r"):
            保留.append(字符)
        elif ord(字符) < 32 or ord(字符) == 127:
            continue  # 丢弃控制字符
        else:
            保留.append(字符)
    文本 = "".join(保留)
    文本 = 文本.replace("\r\n", "\n").replace("\r", "\n")
    文本 = re.sub(r"\n{3,}", "\n\n", 文本)
    文本 = re.sub(r"[ \t]{2,}", " ", 文本)
    # 去重复标点（如 "！！！！！"、"。。。"）
    文本 = re.sub(r"([。！？!?…])\1{2,}", r"\1", 文本)
    return 文本.strip()


def _截断到字数(文本: str, 最大: int) -> str:
    """把文本截断到不超过 最大 个有效字符；优先在句号/感叹号等标点处截断。"""
    文本 = str(文本 or "")
    有效数 = 0
    最后标点位置 = -1
    for 索引, 字符 in enumerate(文本):
        if 字符.isspace():
            continue
        有效数 += 1
        if 字符 in "。！？!?…":
            最后标点位置 = 索引
        if 有效数 >= 最大:
            if 最后标点位置 > 0:
                return 文本[:最后标点位置 + 1]
            return 文本[:索引 + 1]
    return 文本


def _控制字数(正文: str, 最小: int, 最大: int) -> str:
    """把正文控制在 [最小, 最大] 有效字数内：超过截断到句号。"""
    最小 = max(1, _取整数值(最小, 300))
    最大 = max(最小, _取整数值(最大, 2000))
    字数 = _计字数(正文)
    if 字数 > 最大:
        正文 = _截断到字数(正文, 最大)
    return 正文.strip()


def _提取标题(正文: str) -> str:
    """从生成文本中提取标题：优先"标题：xxx"，其次首行《xxx》。"""
    正文 = str(正文 or "")
    匹配 = re.search(r"标题[：:]\s*(.+)", 正文)
    if 匹配:
        return 匹配.group(1).strip().strip("《》").strip()[:30]
    首行 = (正文.splitlines() or [""])[0].strip()
    匹配 = re.search(r"《(.+?)》", 首行)
    if 匹配:
        return 匹配.group(1).strip()[:30]
    return ""


def _去标题行(正文: str) -> str:
    """若首行是标题行（标题：/《》）则从正文中移除。"""
    行 = 正文.splitlines()
    if 行 and (行[0].startswith("标题") or 行[0].startswith("《") or 行[0].startswith("【标题")):
        行 = 行[1:]
    return "\n".join(行).strip()


# ------------------------------------------------------------------
# 教师模型（本地大模型）对话
# ------------------------------------------------------------------


def _加载教师模型(模型路径: str) -> tuple:
    """加载教师模型（AutoModelForCausalLM + AutoTokenizer，trust_remote_code）。

    fp16 → cuda:0 加载，失败降级 CPU。返回 (是否成功, 错误消息)。
    模型与 tokenizer 存入模块级单例，重复调用不重复加载。
    """
    global 教师模型实例, 教师tokenizer, 教师模型设备, 教师模型路径缓存
    if 教师模型实例 is not None and 教师模型路径缓存 == 模型路径:
        return True, ""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as 错误:
        缺失模块 = getattr(错误, "name", "") or str(错误)
        if "transformers" in 缺失模块:
            return False, _教师模型缺依赖提示
        return False, (
            "缺少 torch/transformers 依赖，请运行: "
            "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch transformers"
        )
    if not os.path.isdir(模型路径):
        return False, f"教师模型路径不存在：{模型路径}"
    try:
        print(f"[日记生成] 加载教师模型：{模型路径}")
        加载tokenizer = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
        try:
            模型 = AutoModelForCausalLM.from_pretrained(
                模型路径, trust_remote_code=True, torch_dtype=torch.float16
            )
            模型 = 模型.to("cuda:0")
            设备 = "cuda:0"
            print("[日记生成] 教师模型已加载到 cuda:0（fp16）")
        except Exception as 错误:
            print(f"[日记生成] GPU 加载失败（{错误}），降级到 CPU")
            模型 = AutoModelForCausalLM.from_pretrained(模型路径, trust_remote_code=True)
            设备 = "cpu"
        模型.eval()
        教师模型实例, 教师tokenizer, 教师模型设备, 教师模型路径缓存 = (
            模型, 加载tokenizer, 设备, 模型路径
        )
        return True, ""
    except Exception as 错误:
        return False, f"教师模型加载失败：{错误}"


def _渲染对话模板(消息列表: list) -> str:
    """把消息列表渲染为模型提示词：优先 apply_chat_template，缺失时手工拼接。"""
    消息 = []
    for 条目 in 消息列表 or []:
        if isinstance(条目, str):
            消息.append({"role": "user", "content": 条目})
        elif isinstance(条目, dict):
            role = 条目.get("role") or 条目.get("角色") or "user"
            content = 条目.get("content") or 条目.get("内容") or ""
            消息.append({"role": str(role), "content": str(content)})
    if not 消息:
        消息 = [{"role": "user", "content": "你好"}]
    try:
        if 教师tokenizer is not None and hasattr(教师tokenizer, "apply_chat_template"):
            return 教师tokenizer.apply_chat_template(
                消息, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    片段 = []
    for 条目 in 消息:
        if 条目["role"] == "system":
            片段.append(f"<|im_start|>system\n{条目['content']}<|im_end|>")
        elif 条目["role"] == "assistant":
            片段.append(f"<|im_start|>assistant\n{条目['content']}<|im_end|>")
        else:
            片段.append(f"<|im_start|>user\n{条目['content']}<|im_end|>")
    片段.append("<|im_start|>assistant\n")
    return "\n".join(片段)


def 教师对话(消息列表: list, 参数=None) -> dict:
    """调用配置的本地教师模型（配置→模型→教师模型）做一次对话生成。

    参数:
        消息列表: 对话消息列表，元素可为字符串或
            {"role": "user"/"system"/"assistant", "content": "..."}。
        参数: 可选覆盖生成参数（temperature / top_p / max_new_tokens）。

    返回:
        {"成功": True, "回复": "..."} 或 {"成功": False, "错误": "..."}。
        未配置模型 / 依赖缺失 / 加载失败 / 生成失败均返回友好错误，不抛异常。
    """
    模型路径 = 获取配置项("模型.教师模型", "")
    if not 模型路径:
        return {"成功": False, "错误": _教师模型未配置提示}
    模型路径 = 解析路径(模型路径)

    加载成功, 加载消息 = _加载教师模型(模型路径)
    if not 加载成功:
        return {"成功": False, "错误": 加载消息}

    # 默认生成参数（可被 参数 覆盖）：temperature=0.85, top_p=0.9,
    # max_new_tokens = 配置字数上限 × 1.2
    字数最大 = _取整数值(获取配置项("日记.每篇字数最大", 2000), 2000) or 2000
    生成参数 = {
        "temperature": 0.85,
        "top_p": 0.9,
        "max_new_tokens": int(字数最大 * 1.2),
        "do_sample": True,
    }
    for 键 in ("temperature", "top_p", "do_sample"):
        if 参数 is not None and 键 in 参数 and 参数[键] is not None:
            生成参数[键] = 参数[键]
    if 参数 is not None and "max_new_tokens" in 参数 and 参数["max_new_tokens"]:
        生成参数["max_new_tokens"] = _取整数值(参数["max_new_tokens"], 生成参数["max_new_tokens"])

    try:
        import torch
    except ImportError as 错误:
        return {"成功": False, "错误": f"缺少 torch 依赖：{错误}"}

    try:
        提示 = _渲染对话模板(消息列表)
        输入 = 教师tokenizer(提示, return_tensors="pt")
        设备 = 教师模型设备
        输入 = {键: 值.to(设备) for 键, 值 in 输入.items() if hasattr(值, "to")}
        输入长度 = int(输入["input_ids"].shape[1])
        with torch.no_grad():
            输出 = 教师模型实例.generate(
                **输入,
                max_new_tokens=生成参数["max_new_tokens"],
                temperature=生成参数["temperature"],
                top_p=生成参数["top_p"],
                do_sample=生成参数["do_sample"],
                pad_token_id=教师tokenizer.eos_token_id,
            )
        回复 = 教师tokenizer.decode(输出[0][输入长度:], skip_special_tokens=True)
        回复 = _后处理正文(回复)
        if not 回复:
            return {"成功": False, "错误": "教师模型生成了空回复"}
        return {"成功": True, "回复": 回复}
    except Exception as 错误:
        return {"成功": False, "错误": f"教师模型生成失败：{错误}"}


# ==================================================================
# 二、规划时间线
# ==================================================================


def _内置时间线条目(年龄: int, 出生年份: int) -> dict:
    """内置启发式：按年龄阶段生成一条时间线条目（含主题）。"""
    年份 = 出生年份 + 年龄
    阶段 = _阶段主题(年龄)
    主题库 = {
        "幼年时光": ["学说话", "第一次上幼儿园", "跟着爷爷奶奶生活", "过年放鞭炮"],
        "童年小学": ["上小学的第一天", "和小伙伴们放学回家", "第一次参加运动会", "期末考试", "过生日"],
        "中学岁月": ["升入中学", "认识新同学", "第一次住校", "中考前的日子", "喜欢上写作"],
        "青春成长": ["第一次打工", "学会独自生活", "决定去做喜欢的事", "开始直播", "青春期的迷茫"],
    }
    候选 = 主题库.get(阶段, ["平凡又真实的一天"])
    主题 = 候选[年龄 % len(候选)]
    return {"年龄": 年龄, "年份": 年份, "主题": f"{阶段}·{主题}"}


def _解析时间线(文本: str, 起始年龄: int, 当前年龄: int, 出生年份: int):
    """解析教师模型的规划输出（JSON 数组或行级文本），失败返回 None。"""
    if not isinstance(文本, str) or not 文本.strip():
        return None
    条目列表 = []
    # 方式1：提取 JSON 数组
    开始 = 文本.find("[")
    结束 = 文本.rfind("]")
    if 开始 != -1 and 结束 > 开始:
        try:
            数据 = json.loads(文本[开始:结束 + 1])
        except Exception:
            数据 = None
        if isinstance(数据, list):
            for 项 in 数据:
                if not isinstance(项, dict):
                    continue
                年龄 = _取整数值(项.get("年龄") or 项.get("age") or 项.get("岁"), None)
                if 年龄 is None:
                    continue
                年份 = _取整数值(
                    项.get("年份") or 项.get("year") or 项.get("年"), 出生年份 + 年龄
                )
                主题 = str(项.get("主题") or 项.get("topic") or 项.get("事件") or "")
                条目列表.append({"年龄": 年龄, "年份": 年份, "主题": 主题})
    # 方式2：行级解析，如 "7岁（2017年）：上小学的第一天"
    if not 条目列表:
        for 行 in 文本.splitlines():
            匹配 = re.search(r"(\d+)\s*岁[（(]?(\d{4})?[)）]?[：:，,]\s*(.*)", 行)
            if not 匹配:
                continue
            年龄 = int(匹配.group(1))
            年份 = _取整数值(匹配.group(2), 出生年份 + 年龄)
            主题 = 匹配.group(3).strip()
            条目列表.append({"年龄": 年龄, "年份": 年份, "主题": 主题})
    if not 条目列表:
        return None
    条目列表 = [条目 for 条目 in 条目列表 if 起始年龄 <= 条目["年龄"] <= 当前年龄]
    if not 条目列表:
        return None
    条目列表.sort(key=lambda 条目: 条目["年龄"])
    唯一 = {}
    for 条目 in 条目列表:
        唯一.setdefault(条目["年龄"], 条目)
    # 补全缺失年份
    结果 = []
    for 年龄 in range(起始年龄, 当前年龄 + 1):
        if 年龄 in 唯一:
            结果.append(唯一[年龄])
        else:
            结果.append(_内置时间线条目(年龄, 出生年份))
    return 结果


def 规划时间线(人设: dict, 参数=None) -> list:
    """规划角色人生大事时间线（逐年）。

    参数:
        人设: {"姓名","性别","出生年份","人设描述","关键经历"}，缺失字段给默认。
        参数: 可选覆盖 起始年龄 / 当前年龄 等日记参数。

    返回:
        list，元素为 {"年龄": n, "年份": 出生年份+n, "主题": "..."}，
        从 起始年龄 到 当前年龄 逐年一条。
    """
    人设 = _规范化人设(人设)
    合并参数 = _合并参数(参数)
    起始年龄 = 合并参数["起始年龄"]
    当前年龄 = 合并参数["当前年龄"]
    出生年份 = 人设["出生年份"]

    提示 = (
        f"请为角色「{人设['姓名']}」（{人设['性别']}，{出生年份} 年出生，"
        f"{人设['人设描述'] or '普通又真实的人'}）规划人生大事时间线。\n"
        f"要求：从 {起始年龄} 岁到 {当前年龄} 岁，每一年一条，"
        f"覆盖上学、家庭、成长、打工、学习等真实人生事件，口语化接地气。\n"
        f"只输出 JSON 数组，格式严格为 "
        f'[{{"年龄": {起始年龄}, "年份": {出生年份 + 起始年龄}, "主题": "事件描述"}}, ...]，'
        f"不要输出其他任何文字。"
    )
    结果 = 教师对话(
        [{"role": "user", "content": 提示}],
        参数={"temperature": 0.7, "max_new_tokens": 1024},
    )
    if 结果.get("成功"):
        解析结果 = _解析时间线(结果.get("回复", ""), 起始年龄, 当前年龄, 出生年份)
        if 解析结果:
            return 解析结果
        print("[日记生成] 教师模型时间线解析失败，降级为内置启发式")
    else:
        print(f"[日记生成] 教师对话不可用（{结果.get('错误', '')}），降级为内置启发式")
    # 降级：内置启发式逐年
    return [_内置时间线条目(年龄, 出生年份) for 年龄 in range(起始年龄, 当前年龄 + 1)]


# ==================================================================
# 三、生成单篇日记
# ==================================================================


def _构造日记提示词(
    年龄: int, 年份: int, 人设: dict, 素材: str,
    字数最小: int, 字数最大: int, 风格: str,
) -> str:
    """构造教师对话提示词：要求以角色第一人称写该年龄日记。"""
    姓名 = 人设["姓名"]
    素材段 = f"当天相关素材：{素材}" if 素材 else "当天相关素材：无"
    return (
        f"你是 {姓名}（{人设['性别']}，{人设['出生年份']} 年出生，"
        f"{人设['人设描述'] or '一个普通又真实的人'}）。\n"
        f"现在请你以第一人称\"我\"（即 {姓名}）的口吻，写一篇你 {年龄} 岁"
        f"（{年份} 年）这一天的日记。\n"
        f"要求：\n"
        f"1. 风格：{风格}，像真人随手写的日记，不要像作文或 AI 输出；\n"
        f"2. 字数：约 {字数最小}～{字数最大} 字；\n"
        f"3. 内容真实细腻：写到具体的人、地点、对话和感受，呼应人设"
        f"（性格：{人设['性格']}；家庭：{人设['家庭']}；关键经历：{人设['关键经历'] or '无'}）；\n"
        f"4. 把下面这段素材自然地融进当天的故事里：{素材段}；\n"
        f"5. 先写一句标题（格式：标题：xxx），再另起一段写正文。"
    )


def _降级模板日记(年龄: int, 年份: int, 人设: dict, 素材: str, 最小: int) -> str:
    """内置模板 + 素材拼接生成一篇结构化占位日记（来源：降级模板）。

    保证字数不低于 最小；不足时用扩充段落池补齐。
    """
    姓名 = 人设["姓名"]
    性别 = 人设["性别"]
    他她 = "她" if 性别 == "女" else ("他" if 性别 == "男" else "我")
    阶段 = _阶段主题(年龄)
    描述 = 人设["人设描述"] or "一个喜欢把日子记下来的人"
    家庭 = 人设["家庭"]
    性格 = 人设["性格"]
    出生地 = 人设["出生地"]
    素材句 = f"今天{素材}。" if 素材 else "今天没什么特别的事，吃过饭就待在家里。"

    段落 = [
        f"{年份}年，我{年龄}岁了。这一年我还住在{出生地}，日子过得简简单单，"
        f"但每一件小事我都记得清清楚楚。",
        f"我从小就是个{性格}的人，{描述}。早上醒来，{家庭}已经在厨房里忙开了，"
        f"锅里冒着热气，那种暖洋洋的烟火气，我一辈子都忘不了。",
        f"白天发生了一件让我印象很深的事：{素材句} 那时候的我还不太会讲大道理，"
        f"只觉得心里热乎乎的，想着一定要把这一天好好写下来。",
        f"{年龄}岁的我，慢慢学会了观察身边的人。{家庭}总是念叨我，说我不小了，"
        f"该懂事了。我嘴上答应着，心里却偷偷盼着能快点长大。",
        f"晚上躺在床上，我把今天的事又在脑子里过了一遍：早上的阳光、午饭的香味、"
        f"{素材句} 这些细碎的小事，拼成了我{年龄}岁的这一年。",
        f"这就是我，{姓名}，{年份}年，{年龄}岁。日子很普通，但每一件小事我都记得，"
        f"因为这些都是我真真实实的生活。",
    ]
    扩充池 = [
        f"我悄悄在日记本上写道：{年份}年，{年龄}岁，我要记住今天的全部细节——"
        f"风往哪个方向吹，{家庭}说了什么话，我又笑了多少次。",
        f"后来我才慢慢明白，人会长大，很多事会被忘记，但{年龄}岁这一年的感受，"
        f"会一直留在身体里，就像今天这样的普通日子。",
        f"妈妈说，日子要一天一天过，开心也要一点一点攒。{年龄}岁的我，"
        f"把今天这件小事小心翼翼地攒进了心里。",
        f"隔壁的小伙伴喊我出去玩，我在{出生地}的巷子里跑来跑去，"
        f"笑声把屋顶上的鸽子都惊飞了，那天的天空特别蓝。",
        f"晚上我认认真真想了想{年龄}岁这一年，发现自己好像又长高了一点点，"
        f"也变得更勇敢了一点点。",
    ]
    正文 = "\n\n".join(段落)
    while _计字数(正文) < 最小 and 扩充池:
        正文 += "\n\n" + 扩充池.pop(0)
    return _后处理正文(正文)


def 生成单篇日记(年龄: int, 年份: int, 人设: dict, 上下文素材: str = "", 参数=None) -> dict:
    """生成一篇指定年龄的日记（教师模型优先，失败降级为内置模板）。

    参数:
        年龄: 角色年龄（岁）。
        年份: 日记所属年份。
        人设: 角色人设字典（缺失字段给默认）。
        上下文素材: 打标数据里的转写片段摘要/画面描述，注入相关主题。
        参数: 可选覆盖 每篇字数最小/最大、日记风格 等。

    返回:
        {"年龄", "年份", "标题", "正文", "字数", "来源"}，
        "来源" 为 "教师模型" 或 "降级模板"。
    """
    年龄 = _取整数值(年龄, 7) or 7
    年份 = _取整数值(年份, 2017) or 2017
    人设 = _规范化人设(人设)
    合并参数 = _合并参数(参数)
    字数最小 = 合并参数["每篇字数最小"]
    字数最大 = 合并参数["每篇字数最大"]
    风格 = 合并参数["日记风格"]

    提示词 = _构造日记提示词(
        年龄, 年份, 人设, str(上下文素材 or ""), 字数最小, 字数最大, 风格
    )
    结果 = 教师对话(
        [
            {"role": "system", "content": "你是一位温暖真诚的写作助手，"
             "擅长以角色的第一人称写真实感的生活日记。"},
            {"role": "user", "content": 提示词},
        ],
        参数={"temperature": 0.85, "top_p": 0.9},
    )

    if 结果.get("成功"):
        正文 = _后处理正文(结果.get("回复", ""))
        标题 = _提取标题(正文) or f"{年龄}岁·{_阶段主题(年龄)}"
        正文 = _去标题行(正文)
        来源 = "教师模型"
    else:
        print(f"[日记生成] {年龄} 岁日记生成失败（{结果.get('错误', '')}），降级为内置模板")
        正文 = _降级模板日记(年龄, 年份, 人设, str(上下文素材 or ""), 字数最小)
        标题 = f"{年龄}岁·{_阶段主题(年龄)}"
        来源 = "降级模板"

    正文 = _控制字数(正文, 字数最小, 字数最大)
    标题 = str(标题 or f"{年龄}岁·{_阶段主题(年龄)}").strip()[:40]
    return {
        "年龄": 年龄,
        "年份": 年份,
        "标题": 标题,
        "正文": 正文,
        "字数": _计字数(正文),
        "来源": 来源,
    }


# ==================================================================
# 四、上下文素材收集（打标结果 / 分割片段）
# ==================================================================


def _提取素材片段(数据) -> list:
    """从打标/分割 JSON 数据中提取片段 dict 列表（含 文本/话题摘要）。"""
    if isinstance(数据, list):
        return [条目 for 条目 in 数据 if isinstance(条目, dict)]
    if isinstance(数据, dict):
        分段 = (
            数据.get("分段") or 数据.get("片段列表") or 数据.get("时间戳列表")
            or 数据.get("时间戳") or []
        )
        if isinstance(分段, list):
            片段 = [条目 for 条目 in 分段 if isinstance(条目, dict)]
            if 片段:
                return 片段
        文本 = 数据.get("转写文本") or 数据.get("文本") or ""
        if 文本:
            return [{"话题摘要": "转写", "文本": str(文本)[:500]}]
    return []


def _收集上下文素材(数据目录: str, 上限: int = 素材默认条数) -> str:
    """读取 数据/打标结果 或 数据/分割片段 目录下的转写/片段文本，
    拼接摘要作为"上下文素材"（最多取前 上限 条）。

    参数:
        数据目录: 用户选择的目录（绝对路径或相对项目根路径）。
        上限: 最多拼接的片段条数。

    返回:
        str：以"；"拼接的素材摘要；目录无效或无数据时返回空串。
    """
    if not 数据目录:
        return ""
    目录 = 数据目录 if os.path.isabs(数据目录) else 解析路径(数据目录)
    if not os.path.isdir(目录):
        return ""
    片段列表 = []
    try:
        文件名列表 = sorted(os.listdir(目录))
    except OSError:
        文件名列表 = []
    for 文件名 in 文件名列表:
        if not 文件名.lower().endswith(".json"):
            continue
        try:
            with open(os.path.join(目录, 文件名), "r", encoding="utf-8") as 文件:
                数据 = json.load(文件)
        except Exception:
            continue
        片段列表.extend(_提取素材片段(数据))
        if len(片段列表) >= 上限:
            break
    摘要行 = []
    for 片段 in 片段列表[:上限]:
        摘要 = str(片段.get("话题摘要") or "").strip()
        文本 = str(片段.get("文本") or "").strip()
        if 摘要 and 文本:
            if 摘要 not in 文本:
                摘要行.append(f"[{摘要}] {文本[:80]}")
            else:
                摘要行.append(摘要)
        elif 摘要:
            摘要行.append(摘要)
        elif 文本:
            摘要行.append(文本[:80])
    return "；".join(摘要行)


def _取本年度素材(素材: str, 年龄: int) -> str:
    """从拼接素材中按年龄轮转取一段，保证每年注入的主题不同。"""
    if not 素材:
        return ""
    段列表 = [段.strip() for 段 in 素材.split("；") if 段.strip()]
    if not 段列表:
        return ""
    return 段列表[年龄 % len(段列表)]


# ==================================================================
# 五、生成日记（主流程）
# ==================================================================


def 生成日记(人设: dict, 数据目录: str, 参数=None, 进度回调=None) -> list:
    """生成角色的超长期日记（时间线 → 逐年生成 → 汇总落盘）。

    参数:
        人设: {"姓名","性别","出生年份","人设描述","关键经历"}，缺失字段给默认。
        数据目录: 打标结果或分割片段目录（供注入上下文素材），可为空。
        参数: 可选覆盖配置项 日记.* 的参数字典。
        进度回调: 可选回调函数 进度回调(进度: float, 消息: str)。

    落盘:
        - 数据/日记/<角色名>_日记.json（元数据 + 人设 + 日记列表）
        - 数据/日记/<角色名>_<年龄>岁_<年份>.md（每篇单独，供人工审阅）

    幂等: 已有 <角色名>_日记.json 且 参数签名 未变时，只增量生成缺失年份；
    参数签名变化时全量重新生成。

    返回:
        list，元素为 {"年龄","年份","标题","正文","字数","来源","状态"}。
    """
    人设 = _规范化人设(人设)
    角色名 = 人设["姓名"]
    合并参数 = _合并参数(参数)
    签名 = _参数签名(参数)
    上限 = 合并参数["生成数量上限"]

    汇总路径 = _日记汇总路径(角色名)
    已有列表, 已有签名 = _读取已有日记(汇总路径)
    增量模式 = bool(已有列表) and (已有签名 == 签名)

    时间线 = 规划时间线(人设, 参数=参数)[:上限]
    素材 = _收集上下文素材(数据目录)

    日记列表 = list(已有列表) if 增量模式 else []
    已有年龄 = {条目.get("年龄") for 条目 in 日记列表}
    待生成 = (
        [条目 for 条目 in 时间线 if 条目.get("年龄") not in 已有年龄]
        if 增量模式 else 时间线
    )

    总数 = len(待生成)
    for 序号, 条目 in enumerate(待生成, 1):
        年龄 = 条目.get("年龄")
        年份 = 条目.get("年份")
        进度 = 序号 / 总数 if 总数 else 1.0
        if 进度回调 is not None:
            try:
                进度回调(进度, f"正在生成 {年龄} 岁日记...")
            except Exception:
                pass
        单篇 = 生成单篇日记(
            年龄, 年份, 人设, _取本年度素材(素材, 年龄 or 0), 参数=参数
        )
        单篇["主题"] = str(条目.get("主题") or "")
        单篇["状态"] = "待审阅"
        日记列表.append(单篇)
        _写单篇md(角色名, 单篇)

    日记列表.sort(key=lambda 篇: _取整数值(篇.get("年龄"), 0) or 0)
    元数据 = {
        "角色名": 角色名,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "数据目录": os.path.abspath(数据目录) if 数据目录 else "",
        "参数签名": 签名,
        "参数": {键: 值 for 键, 值 in 合并参数.items()},
        "起始年龄": 合并参数["起始年龄"],
        "当前年龄": 合并参数["当前年龄"],
        "总篇数": len(日记列表),
        "本次新增": len(待生成),
        "模式": "增量" if 增量模式 else "全新生成",
    }
    try:
        with open(汇总路径, "w", encoding="utf-8") as 文件:
            json.dump(
                {"元数据": 元数据, "人设": 人设, "日记列表": 日记列表},
                文件, ensure_ascii=False, indent=2,
            )
    except OSError as 错误:
        print(f"[日记生成] 汇总文件写入失败：{错误}")
    if 进度回调 is not None:
        try:
            进度回调(1.0, "全部日记生成完成")
        except Exception:
            pass
    return 日记列表


# ==================================================================
# 六、审阅日记
# ==================================================================


def 审阅日记(角色名: str, 年龄: int, 修改后正文: str) -> bool:
    """把人工修改后的正文写回 json 与 md。

    参数:
        角色名: 角色姓名。
        年龄: 日记对应年龄。
        修改后正文: 人工修改/补充后的正文。

    返回:
        bool：写回成功返回 True，否则 False。
    """
    角色名 = (角色名 or "").strip()
    修改后正文 = (修改后正文 or "").strip()
    if not 角色名 or not 修改后正文:
        return False
    汇总路径 = _日记汇总路径(角色名)
    已有列表, _ = _读取已有日记(汇总路径)
    if not 已有列表:
        return False
    年龄 = _取整数值(年龄, None)
    if 年龄 is None:
        return False
    目标条目 = None
    for 条目 in 已有列表:
        if _取整数值(条目.get("年龄"), None) == 年龄:
            目标条目 = 条目
            break
    if 目标条目 is None:
        return False
    目标条目["正文"] = _后处理正文(修改后正文)
    目标条目["字数"] = _计字数(目标条目["正文"])
    目标条目["状态"] = "已审阅"
    try:
        with open(汇总路径, "r", encoding="utf-8") as 文件:
            完整数据 = json.load(文件)
    except Exception:
        完整数据 = None
    try:
        with open(汇总路径, "w", encoding="utf-8") as 文件:
            if 完整数据 is not None and isinstance(完整数据, dict):
                完整数据["日记列表"] = 已有列表
                json.dump(完整数据, 文件, ensure_ascii=False, indent=2)
            else:
                json.dump(
                    {"元数据": {}, "人设": {}, "日记列表": 已有列表},
                    文件, ensure_ascii=False, indent=2,
                )
    except OSError:
        return False
    return _写单篇md(角色名, 目标条目)


# ==================================================================
# 七、导出微调文本
# ==================================================================


def _微调数据包目录() -> str:
    """读取配置项 微调.数据包目录 并解析为绝对路径（不存在则创建）。"""
    目录 = 获取配置项("微调.数据包目录", "数据/微调数据包")
    if not 目录:
        目录 = "数据/微调数据包"
    if not os.path.isabs(目录):
        目录 = 解析路径(目录)
    os.makedirs(目录, exist_ok=True)
    return 目录


def 导出微调文本(角色名: str) -> dict:
    """把角色全部日记转为身份/记忆微调文本，落盘为 jsonl 数据包。

    每篇日记 → {"instruction": "你过去的一天（{年龄}岁，{年份}年）",
                 "response": "{日记正文}"}（指令模板可由配置覆盖）。

    落盘: 数据/微调数据包/日记微调_<角色名>_<时间戳>.jsonl

    返回:
        {"成功": True, "路径": "...", "条数": n}；无日记时返回友好错误。
    """
    角色名 = (角色名 or "").strip()
    if not 角色名:
        return {"成功": False, "错误": "角色名不能为空", "路径": "", "条数": 0}
    日记列表, _ = _读取已有日记(_日记汇总路径(角色名))
    if not 日记列表:
        return {
            "成功": False,
            "错误": f"角色「{角色名}」暂无日记，请先在日记页生成日记",
            "路径": "",
            "条数": 0,
        }
    指令模板 = str(
        获取配置项("微调.日记指令模板", "你过去的一天（{年龄}岁，{年份}年）")
        or "你过去的一天（{年龄}岁，{年份}年）"
    )
    时间戳 = time.strftime("%Y%m%d_%H%M%S")
    路径 = os.path.join(_微调数据包目录(), f"日记微调_{_安全文件名(角色名)}_{时间戳}.jsonl")
    条数 = 0
    try:
        with open(路径, "w", encoding="utf-8") as 文件:
            for 条目 in 日记列表:
                正文 = _后处理正文(条目.get("正文") or "")
                if not 正文:
                    continue
                指令 = 指令模板.format(
                    年龄=条目.get("年龄"), 年份=条目.get("年份"), 标题=条目.get("标题") or ""
                )
                记录 = {
                    "instruction": 指令,
                    "response": 正文,
                    "年龄": 条目.get("年龄"),
                    "年份": 条目.get("年份"),
                    "标题": 条目.get("标题") or "",
                    "角色名": 角色名,
                }
                文件.write(json.dumps(记录, ensure_ascii=False) + "\n")
                条数 += 1
    except OSError as 错误:
        return {"成功": False, "错误": f"导出微调文本失败：{错误}", "路径": "", "条数": 0}
    return {"成功": True, "路径": 路径, "条数": 条数}


# ==================================================================
# 八、HTTP 路由注册
# ==================================================================


def _后台生成(任务ID: str, 人设: dict, 数据目录: str, 参数=None) -> None:
    """后台任务：执行 生成日记 并同步全局 日记进度。"""
    with _进度锁:
        日记进度[任务ID] = {"进度": 0.0, "消息": "准备生成", "状态": "生成中", "结果": None}

    def 回调(进度: float, 消息: str) -> None:
        with _进度锁:
            条目 = 日记进度.get(任务ID)
            if 条目 is not None:
                条目["进度"] = 进度
                条目["消息"] = 消息

    try:
        结果 = 生成日记(人设, 数据目录, 参数=参数, 进度回调=回调)
        with _进度锁:
            条目 = 日记进度.get(任务ID)
            if 条目 is not None:
                条目["结果"] = 结果
                条目["进度"] = 1.0
                条目["消息"] = "生成完成"
                条目["状态"] = "完成"
    except Exception as 错误:
        with _进度锁:
            条目 = 日记进度.get(任务ID)
            if 条目 is not None:
                条目["状态"] = "失败"
                条目["消息"] = f"生成异常：{错误}"


def 注册路由(app) -> None:
    """注册日记生成模块的 HTTP 路由（挂载到 FastAPI 应用）。

    接口:
        POST /api/日记/规划     body：人设 → 返回时间线
        POST /api/日记/生成     body：人设 + 数据目录（后台异步）
        GET  /api/日记/进度     ?任务ID= → 后台生成进度
        GET  /api/日记/列表     ?角色名= → 已生成日记列表
        GET  /api/日记/单篇     ?角色名=&年龄= → 单篇内容
        POST /api/日记/审阅     body：角色名+年龄+修改后正文
        POST /api/日记/导出     body：角色名 → 导出微调文本
        POST /api/日记/对话     body：消息列表 → 教师对话
    """
    try:
        from fastapi import BackgroundTasks, Query
        from pydantic import BaseModel
    except Exception as 错误:
        print(f"[日记生成] 缺少 FastAPI 依赖，跳过路由注册：{错误}")
        return

    class 规划请求(BaseModel):
        人设: dict
        参数: dict = None

    class 生成请求(BaseModel):
        人设: dict
        数据目录: str = ""
        参数: dict = None

    class 审阅请求(BaseModel):
        角色名: str
        年龄: int
        修改后正文: str

    class 角色请求(BaseModel):
        角色名: str

    class 对话请求(BaseModel):
        消息列表: list = []
        参数: dict = None

    @app.post("/api/日记/规划")
    def 规划接口(请求: 规划请求) -> dict:
        """根据人设规划人生大事时间线。"""
        try:
            时间线 = 规划时间线(请求.人设, 参数=请求.参数)
            return {"成功": True, "时间线": 时间线, "条数": len(时间线)}
        except Exception as 错误:
            return {"成功": False, "错误": f"规划时间线失败：{错误}"}

    @app.post("/api/日记/生成")
    def 生成接口(请求: 生成请求, 后台任务: BackgroundTasks) -> dict:
        """提交后台生成任务，返回任务ID供轮询进度。"""
        人设 = 请求.人设 or {}
        角色名 = str(人设.get("姓名") or "未命名角色").strip() or "未命名角色"
        任务ID = f"{_安全文件名(角色名)}_{time.strftime('%Y%m%d%H%M%S')}"
        with _进度锁:
            日记进度[任务ID] = {
                "进度": 0.0, "消息": "任务已提交，等待后台执行", "状态": "生成中", "结果": None,
            }
        后台任务.add_task(_后台生成, 任务ID, 人设, (请求.数据目录 or "").strip(), 请求.参数)
        return {
            "成功": True,
            "任务ID": 任务ID,
            "角色名": 角色名,
            "消息": "日记生成已提交后台执行",
        }

    @app.get("/api/日记/进度")
    def 进度接口(任务ID: str = Query("")) -> dict:
        """查询后台生成进度；不传任务ID时返回全部任务状态。"""
        if 任务ID:
            条目 = 日记进度.get(任务ID)
            if not 条目:
                return {"任务ID": 任务ID, "进度": 0.0, "状态": "未开始", "消息": ""}
            return {
                "任务ID": 任务ID,
                "进度": 条目["进度"],
                "消息": 条目["消息"],
                "状态": 条目["状态"],
            }
        return {"成功": True, "任务列表": 日记进度}

    @app.get("/api/日记/列表")
    def 列表接口(角色名: str = Query(...)) -> dict:
        """返回已生成的角色日记列表。"""
        角色名 = (角色名 or "").strip()
        if not 角色名:
            return {"成功": False, "错误": "角色名不能为空", "日记列表": []}
        日记列表, _ = _读取已有日记(_日记汇总路径(角色名))
        if not 日记列表:
            return {
                "成功": False, "角色名": 角色名,
                "错误": f"角色「{角色名}」暂无日记", "日记列表": [],
            }
        return {"成功": True, "角色名": 角色名, "日记列表": 日记列表, "条数": len(日记列表)}

    @app.get("/api/日记/单篇")
    def 单篇接口(角色名: str = Query(...), 年龄: int = Query(...)) -> dict:
        """返回指定年龄的单篇日记。"""
        角色名 = (角色名 or "").strip()
        日记列表, _ = _读取已有日记(_日记汇总路径(角色名))
        for 条目 in 日记列表:
            if _取整数值(条目.get("年龄"), None) == 年龄:
                return {"成功": True, "角色名": 角色名, "篇目": 条目}
        return {"成功": False, "错误": f"未找到 {角色名} {年龄} 岁的日记"}

    @app.post("/api/日记/审阅")
    def 审阅接口(请求: 审阅请求) -> dict:
        """人工审阅：把修改后的正文写回 json 与 md。"""
        成功 = 审阅日记(请求.角色名.strip(), 请求.年龄, 请求.修改后正文)
        return {"成功": 成功, "状态": "已保存" if 成功 else "保存失败"}

    @app.post("/api/日记/导出")
    def 导出接口(请求: 角色请求) -> dict:
        """把角色全部日记导出为微调文本数据包。"""
        return 导出微调文本(请求.角色名.strip())

    @app.post("/api/日记/对话")
    def 对话接口(请求: 对话请求) -> dict:
        """教师对话（供前端和 30B 聊怎么写日记/人设）。"""
        return 教师对话(请求.消息列表, 参数=请求.参数)
