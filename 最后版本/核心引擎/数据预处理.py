# -*- coding: utf-8 -*-
"""
数据预处理模块
==============
负责全流程数据入口：上传清洗 → 音频转文本 → 话题分割 → 边界微调。

- 上传并清洗：把文件复制到 数据/上传/，生成唯一任务ID，做文本清洗或音视频基础校验；
- 音频转文本：调用配置的全模态 ASR 模型（Qwen2.5-Omni / Whisper 等）分段转写，
  支持断点续转与进度回调；
- 话题分割：按话题语义（jieba 词集重叠相似度）而非按时长切分转写文本；
- 分割预览 / 调整片段边界：供前端展示与人工微调片段边界；
- 注册路由：挂载 FastAPI HTTP 接口（含 BackgroundTasks 异步转写与进度查询）。

本模块所有外部依赖（transformers / torch / jieba / 音频解码库）均按需
try/except 容错降级，缺失时保证核心流程仍可用。
"""

import json
import math
import os
import re
import shutil
import subprocess
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

视频扩展名 = {".mp4", ".mkv", ".avi", ".mov", ".flv"}
音频扩展名 = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}

# 话题分割用到的轻度停用词表（jieba 缺失或摘要提取时过滤）
_停用词 = {
    "的", "了", "是", "在", "和", "与", "及", "就", "都", "也", "还", "又", "很",
    "把", "被", "让", "给", "对", "从", "向", "到", "我", "你", "他", "她", "它",
    "我们", "你们", "他们", "她们", "它们", "这", "那", "个", "中", "上", "下",
    "有", "说", "道", "着", "过", "呢", "吗", "吧", "啊", "呀", "哦", "嗯", "哈",
    "然后", "就是", "这个", "那个", "一个", "什么", "怎么", "这样", "那样",
    "自己", "起来", "出来", "下去", "因为", "所以", "但是", "可是", "不过",
    "如果", "虽然", "而且", "并且", "或者", "还是", "已经", "正在", "可以",
    "可能", "应该", "觉得", "知道", "没有", "不是", "非常", "特别", "真的",
    "其实", "当然", "今天", "现在", "时候", "一下", "一点", "一样",
}

# 话题切换提示词（jieba 缺失降级算法使用）
_话题转换词 = (
    "然后", "接着", "接下来", "最后", "说到", "谈到", "关于", "对了",
    "聊到", "再说", "另外", "还有", "顺便", "回到", "换个", "接下来我们",
)

# ASR 模型加载缓存（模块级，避免重复加载）
_ASR模型 = None
_ASR处理器 = None
_ASR设备 = "cpu"
_ASR模型ID缓存 = ""

# 转写进度缓存：{任务ID: {"进度": float, "消息": str, "状态": str, "结果": dict|None}}
_转写进度 = {}
_进度锁 = threading.Lock()


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


def _从路径取任务ID(路径: str) -> str:
    """从文件名前缀推断任务ID（形如 <任务ID>_<原名>），否则新建。"""
    文件名 = os.path.splitext(os.path.basename(路径 or ""))[0]
    前缀 = 文件名.split("_", 1)[0]
    if len(前缀) >= 16 and 前缀[:8].isdigit():
        return 前缀
    return _生成任务ID()


def _读文本文件(路径: str) -> str:
    """按多种编码尝试读取文本文件，避免乱码。"""
    for 编码 in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            with open(路径, "r", encoding=编码) as 文件:
                return 文件.read()
        except (UnicodeDecodeError, OSError):
            continue
    with open(路径, "r", encoding="utf-8", errors="replace") as 文件:
        return 文件.read()


def _清洗文本(文本: str) -> tuple:
    """文本清洗：去控制字符 → 统一换行 → 压缩连续空行 → 可选相同行去重。

    返回:
        (清洗后文本, 统计dict)
    """
    原字符数 = len(文本)
    控制字符数 = 0
    保留 = []
    for 字符 in 文本:
        if 字符 in ("\n", "\t", "\r"):
            保留.append(字符)
        elif ord(字符) < 32 or ord(字符) == 127:
            控制字符数 += 1
        else:
            保留.append(字符)
    文本 = "".join(保留)
    # 统一换行：\r\n → \n，\r → \n
    文本 = 文本.replace("\r\n", "\n").replace("\r", "\n")
    # 压缩连续空行
    空行数 = 文本.count("\n\n")
    文本 = re.sub(r"\n{2,}", "\n", 文本)
    # 可选相同行去重（配置 数据预处理.清洗去重）
    去重条数 = 0
    if 获取配置项("数据预处理.清洗去重", True):
        已见 = set()
        行列表 = 文本.split("\n")
        保留行 = []
        for 行 in 行列表:
            行内容 = 行.strip()
            if 行内容 and 行内容 in 已见:
                去重条数 += 1
                continue
            已见.add(行内容)
            保留行.append(行)
        文本 = "\n".join(保留行)
    return 文本, {
        "原字符数": 原字符数,
        "清洗后字符数": len(文本),
        "去除控制字符数": 控制字符数,
        "去除空行数": 空行数,
        "去重条数": 去重条数,
        "清洗算法": "去控制字符+统一换行+压缩空行+相同行去重",
    }


def _探测时长(路径: str) -> float:
    """尝试用 ffprobe / soundfile 探测媒体时长，失败返回 0。"""
    try:
        结果 = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", 路径],
            capture_output=True, text=True, timeout=30,
        )
        if 结果.returncode == 0 and 结果.stdout.strip():
            return round(float(结果.stdout.strip()), 2)
    except Exception:
        pass
    try:
        import soundfile as sf
        return round(sf.info(路径).duration, 2)
    except Exception:
        pass
    return 0.0


# ------------------------------------------------------------------
# 音频解码与切段
# ------------------------------------------------------------------


def _重采样(音频, 原采样率: int, 新采样率: int):
    """numpy 线性插值重采样（无 scipy 依赖）。"""
    import numpy as np
    目标长度 = int(len(音频) * 新采样率 / 原采样率)
    原索引 = np.arange(len(音频))
    新索引 = np.linspace(0, len(音频) - 1, 目标长度)
    return np.interp(新索引, 原索引, 音频).astype("float32")


def _读取音频(路径: str) -> tuple:
    """解码音频/视频为 (float32 单声道数组, 采样率)。

    依次尝试 librosa / soundfile / torchaudio / ffmpeg 子进程，
    全部失败时抛出 RuntimeError。
    """
    try:
        import librosa
        音频, 采样率 = librosa.load(路径, sr=16000, mono=True)
        return 音频.astype("float32"), 16000
    except ImportError:
        pass
    except Exception:
        pass
    try:
        import soundfile as sf
        音频, 采样率 = sf.read(路径, dtype="float32", always_2d=False)
        if len(音频.shape) > 1:
            音频 = 音频.mean(axis=1)
        if 采样率 != 16000:
            音频 = _重采样(音频, 采样率, 16000)
            采样率 = 16000
        return 音频.astype("float32"), 16000
    except ImportError:
        pass
    except Exception:
        pass
    try:
        import torchaudio
        波形, 采样率 = torchaudio.load(路径)
        if 波形.shape[0] > 1:
            波形 = 波形.mean(dim=0, keepdim=True)
        if 采样率 != 16000:
            波形 = torchaudio.functional.resample(波形, 采样率, 16000)
            采样率 = 16000
        return 波形[0].numpy().astype("float32"), 16000
    except ImportError:
        pass
    except Exception:
        pass
    try:
        结果 = subprocess.run(
            ["ffmpeg", "-i", 路径, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
            capture_output=True, timeout=600,
        )
        if 结果.returncode == 0 and len(结果.stdout) > 0:
            import numpy as np
            音频 = np.frombuffer(结果.stdout, dtype=np.int16).astype("float32") / 32768.0
            return 音频, 16000
    except Exception:
        pass
    raise RuntimeError(
        f"无法解码音频：{路径}（请安装 librosa / soundfile / torchaudio 或 ffmpeg）"
    )


def _音频切段(音频, 采样率: int, 段长秒: float):
    """把音频数组按固定秒数切段，产出 (段数组, 起始秒, 结束秒)。"""
    段采样数 = max(1, int(段长秒 * 采样率))
    总采样数 = len(音频)
    for 起始 in range(0, 总采样数, 段采样数):
        结束 = min(起始 + 段采样数, 总采样数)
        if 结束 - 起始 <= 0:
            break
        yield 音频[起始:结束], 起始 / 采样率, 结束 / 采样率


# ------------------------------------------------------------------
# ASR 模型加载与多方式调用
# ------------------------------------------------------------------


def _调用ASRWorker(模型ID: str, 音频路径: str, 段边界: list, 任务ID: str) -> dict:
    """通过独立 worker 子进程调用全模态 ASR 模型转写。

    Qwen2.5-Omni-7B 等全模态模型需要 transformers 4.5x，而主环境为 5.x，
    因此转写在 .venv_asr 独立环境中执行，避免版本冲突。

    返回:
        dict：worker 输出的 {"成功", "文本", "分段", "总时长秒", "耗时秒"} 或
              {"成功": False, "错误": ...}
    """
    import subprocess
    项目根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(项目根, ".venv_asr", "Scripts", "python.exe")
    worker脚本 = os.path.join(项目根, "核心引擎", "asr转写worker.py")
    if not os.path.isfile(venv_python):
        return {
            "成功": False,
            "错误": "未找到 ASR 环境 .venv_asr\\Scripts\\python.exe，请先执行："
                    "python -m venv --system-site-packages .venv_asr 并运行 "
                    ".venv_asr\\Scripts\\pip install -i https://mirrors.aliyun.com/pypi/simple/ transformers==4.54.1",
        }
    if not os.path.isfile(worker脚本):
        return {"成功": False, "错误": f"ASR worker 脚本缺失：{worker脚本}"}

    打标目录 = _取目录("打标.打标结果目录", "数据/打标结果")
    os.makedirs(打标目录, exist_ok=True)
    输出路径 = os.path.join(打标目录, f"{任务ID}_asr结果.json")
    边界路径 = os.path.join(打标目录, f"{任务ID}_段边界.json")
    try:
        with open(边界路径, "w", encoding="utf-8") as f:
            json.dump(段边界, f, ensure_ascii=False)
    except OSError as 错误:
        return {"成功": False, "错误": f"段边界写入失败：{错误}"}

    try:
        # 干净子进程环境：移除 PYTHONPATH/PYTHONHOME，避免继承主环境 transformers 5.x 路径
        子环境 = dict(os.environ)
        子环境.pop("PYTHONPATH", None)
        子环境.pop("PYTHONHOME", None)
        子环境["PYTHONIOENCODING"] = "utf-8"
        # stdout/stderr 重定向到日志文件（避免管道交互导致原生库崩溃）
        日志路径 = os.path.join(打标目录, f"{任务ID}_asr日志.txt")

        # 全模态模型原生加载偶发崩溃（CUDA 驱动级 0xC0000005），自动重试兜底
        最大重试 = 2
        最后诊断 = ""
        for 尝试 in range(最大重试 + 1):
            if 尝试 > 0:
                print(f"[转写] ASR worker 第 {尝试} 次重试（共 {最大重试 + 1} 次）", flush=True)
                time.sleep(8)  # 等待 GPU 上下文释放
            with open(日志路径, "w", encoding="utf-8", errors="replace") as 日志句柄:
                进程 = subprocess.run(
                    [venv_python, worker脚本, 模型ID, 音频路径, 输出路径, 边界路径],
                    stdout=日志句柄, stderr=subprocess.STDOUT,
                    timeout=7200, env=子环境,
                )
            if os.path.isfile(输出路径):
                try:
                    with open(输出路径, "r", encoding="utf-8") as f:
                        结果 = json.load(f)
                    if isinstance(结果, dict) and 结果.get("成功"):
                        return 结果
                except Exception:
                    pass
            # 读取日志尾部作为诊断（仅保留最后一次）
            try:
                with open(日志路径, "r", encoding="utf-8", errors="replace") as f:
                    最后诊断 = f.read()[-600:]
            except OSError:
                最后诊断 = ""
    except subprocess.TimeoutExpired:
        return {"成功": False, "错误": "ASR 转写超时（超过 2 小时）"}
    except Exception as 错误:
        return {"成功": False, "错误": f"ASR 进程启动失败：{错误}"}

    return {"成功": False, "错误": f"ASR 转写失败（退出码 {进程.returncode}，已重试 {最大重试} 次）：{最后诊断}"}


def _加载ASR模型(模型ID: str, 模型类型: str) -> tuple:
    """加载 ASR 模型（AutoModel + AutoProcessor，trust_remote_code）。

    fp16 加载到 cuda:0，失败降级 CPU。返回 (是否成功, 错误消息)。
    """
    global _ASR模型, _ASR处理器, _ASR设备, _ASR模型ID缓存
    if _ASR模型 is not None and _ASR模型ID缓存 == 模型ID:
        return True, ""
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as 错误:
        return False, f"缺少 transformers / torch 依赖，无法加载 ASR 模型：{错误}"
    try:
        print(f"[转写] 加载ASR模型：{模型ID}（类型：{模型类型}）")
        处理器 = AutoProcessor.from_pretrained(模型ID, trust_remote_code=True)
        try:
            模型 = AutoModel.from_pretrained(
                模型ID, trust_remote_code=True, torch_dtype=torch.float16
            )
            模型 = 模型.to("cuda:0")
            _ASR设备 = "cuda:0"
            print("[转写] 模型已加载到 cuda:0（fp16）")
        except Exception as 错误:
            print(f"[转写] GPU 加载失败（{错误}），降级到 CPU")
            模型 = AutoModel.from_pretrained(模型ID, trust_remote_code=True)
            _ASR设备 = "cpu"
        _ASR模型, _ASR处理器, _ASR模型ID缓存 = 模型, 处理器, 模型ID
        return True, ""
    except Exception as 错误:
        return False, f"ASR 模型加载失败：{错误}"


def _模型转写(模型, 处理器, 音频段, 采样率: int, 设备: str) -> str:
    """尝试多种调用方式完成单段音频转写，返回文本（失败返回空串）。"""
    # 方式1：Omni 类模型（如 Qwen2.5-Omni）
    音频占位 = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    for 变体, 文本参数 in (("Omni类(标准模板)", 音频占位), ("Omni类(无文本)", None)):
        try:
            print(f"[转写] 调用方式：{变体}")
            输入 = 处理器(
                text=文本参数, audios=音频段, sampling_rate=采样率, return_tensors="pt"
            )
            输入 = {k: v.to(设备) for k, v in 输入.items() if hasattr(v, "to")}
            生成结果 = 模型.generate(
                **输入, generate_kwargs={"language": "zh", "task": "transcribe"}
            )
            文本 = 处理器.batch_decode(生成结果, skip_special_tokens=True)
            拼接 = "".join(文本).strip()
            if 拼接:
                return 拼接
        except Exception as 错误:
            print(f"[转写] {变体} 失败：{错误}")
    # 方式2：Whisper 类模型（input_features）
    try:
        print("[转写] 调用方式：Whisper类(input_features)")
        输入 = 处理器(音频段, sampling_rate=采样率, return_tensors="pt")
        输入 = {k: v.to(设备) for k, v in 输入.items() if hasattr(v, "to")}
        生成结果 = 模型.generate(**输入)
        文本 = 处理器.batch_decode(生成结果, skip_special_tokens=True)
        拼接 = "".join(文本).strip()
        if 拼接:
            return 拼接
    except Exception as 错误:
        print(f"[转写] Whisper类 失败：{错误}")
    # 方式3：通用类（audios 参数 + generate 直传）
    try:
        print("[转写] 调用方式：通用类(audios参数)")
        输入 = 处理器(audios=音频段, sampling_rate=采样率, return_tensors="pt")
        输入 = {k: v.to(设备) for k, v in 输入.items() if hasattr(v, "to")}
        生成结果 = 模型.generate(**输入, max_new_tokens=512)
        文本 = 处理器.batch_decode(生成结果, skip_special_tokens=True)
        拼接 = "".join(文本).strip()
        if 拼接:
            return 拼接
    except Exception as 错误:
        print(f"[转写] 通用类 失败：{错误}")
    return ""


# ------------------------------------------------------------------
# 进度缓存
# ------------------------------------------------------------------


def _更新进度(任务ID: str, 进度: float, 消息: str) -> None:
    """更新模块级转写进度缓存（线程安全）。"""
    with _进度锁:
        条目 = _转写进度.setdefault(
            任务ID, {"进度": 0.0, "消息": "", "状态": "转写中", "结果": None}
        )
        条目["进度"] = 进度
        条目["消息"] = 消息
        if 进度 >= 1.0:
            条目["状态"] = "完成"


# ------------------------------------------------------------------
# 文本处理辅助（分句 / 分词 / 相似度 / 摘要）
# ------------------------------------------------------------------


def _分句(文本: str) -> list:
    """按 句号/问号/感叹号/省略号 分句，返回 [(文本, 起始字符, 结束字符)]。"""
    句子 = []
    for 匹配 in re.finditer(r"[^。！？!?…]+[。！？!?…]*", 文本 or ""):
        内容 = 匹配.group()
        if 内容.strip():
            句子.append((内容, 匹配.start(), 匹配.end()))
    if not 句子 and (文本 or "").strip():
        句子.append((文本.strip(), 0, len(文本)))
    return 句子


def _分词(文本: str) -> list:
    """jieba 分词（过滤停用词）；jieba 缺失时降级为连续中文双字滑窗。"""
    try:
        import jieba
        词列表 = [w.strip() for w in jieba.cut(文本 or "")]
    except ImportError:
        中文段 = re.findall(r"[\u4e00-\u9fff]+", 文本 or "")
        词列表 = []
        for 段 in 中文段:
            if len(段) <= 2:
                词列表.append(段)
            else:
                for i in range(len(段) - 1):
                    词列表.append(段[i:i + 2])
    # 过滤停用词、空串及纯标点/非中文 token（避免标点污染相似度计算）
    return [
        词 for 词 in 词列表
        if 词 and 词 not in _停用词 and re.search(r"[\u4e00-\u9fff]", 词)
    ]


def _词集相似度(集A: set, 集B: set) -> float:
    """Jaccard 相似度：交集 / 并集。"""
    if not 集A or not 集B:
        return 0.0
    并集 = 集A | 集B
    if not 并集:
        return 0.0
    return len(集A & 集B) / len(并集)


def _含话题转换词(句子: str) -> bool:
    """句子是否以话题转换提示词开头（jieba 缺失时的降级信号）。"""
    文本 = (句子 or "").strip()
    return any(文本.startswith(词) for 词 in _话题转换词)


def _生成话题摘要(文本: str, 算法: str) -> str:
    """用片段内词频最高的 2~3 个词拼成话题摘要。"""
    词列表 = _分词(文本)
    if not 词列表:
        return (文本 or "")[:10]
    频率 = Counter(词列表)
    候选 = [词 for 词, _ in 频率.most_common() if len(词) > 1]
    if not 候选:
        候选 = [词 for 词, _ in 频率.most_common()]
    return "、".join(候选[:3])


def _时间戳总时长(时间戳列表: list) -> float:
    """取时间戳列表中的最大结束秒作为总时长。"""
    if not 时间戳列表:
        return 0.0
    最大 = 0.0
    for 条目 in 时间戳列表:
        if isinstance(条目, dict):
            开始 = 条目.get("开始秒") or 条目.get("开始时间") or 0
            结束 = 条目.get("结束秒") or 条目.get("结束时间") or 开始
            最大 = max(最大, float(结束 or 开始 or 0))
        else:
            try:
                最大 = max(最大, float(条目[1]))
            except (IndexError, TypeError, ValueError):
                continue
    return 最大


def _字符位置到时间(字符位置: int, 总字符数: int, 总时长: float) -> float:
    """按字符位置在全文中占比映射到时间轴（无逐句时间戳时的近似）。"""
    if 总字符数 <= 0 or 总时长 <= 0:
        return 0.0
    return round(总时长 * 字符位置 / 总字符数, 2)


def _规范化时间戳(时间戳列表: list) -> list:
    """把 dict 列表或 [[起,止,文本]...] 统一为 [{"起始秒","结束秒","文本"}]。"""
    结果 = []
    for 条目 in 时间戳列表 or []:
        if isinstance(条目, dict):
            起始 = 条目.get("起始秒", 条目.get("开始秒", 条目.get("开始时间", 0)))
            结束 = 条目.get("结束秒", 条目.get("结束时间", 条目.get("停止秒", 起始)))
            文本 = 条目.get("文本", "") or ""
        else:
            try:
                起始, 结束, 文本 = 条目[0], 条目[1], 条目[2]
            except (IndexError, TypeError):
                try:
                    起始, 结束, 文本 = 条目[0], 条目[1], ""
                except (IndexError, TypeError):
                    continue
        try:
            起始 = float(起始)
            结束 = float(结束)
        except (TypeError, ValueError):
            continue
        if 结束 < 起始:
            结束, 起始 = 起始, 结束
        结果.append({"起始秒": 起始, "结束秒": 结束, "文本": str(文本)})
    return 结果


def _按时间取文本(片段列表: list, 起始: float, 结束: float) -> str:
    """从片段列表中取时间落在 [起始, 结束] 范围内的文本并拼接。"""
    文本 = ""
    for 片段 in 片段列表 or []:
        片段起 = float(片段.get("起始秒") or 片段.get("开始秒") or 0)
        片段止 = float(片段.get("结束秒") or 片段起)
        if 片段止 <= 起始:
            continue
        if 片段起 >= 结束:
            break
        文本 += 片段.get("文本", "") or ""
    return 文本


def _读转写结果(任务ID: str) -> dict:
    """读取断点续转文件 数据/打标结果/<任务ID>_转写.json。"""
    打标结果目录 = _取目录("打标.打标结果目录", "数据/打标结果")
    文件 = os.path.join(打标结果目录, f"{任务ID}_转写.json")
    if not os.path.exists(文件):
        return None
    try:
        with open(文件, "r", encoding="utf-8") as 文件句柄:
            return json.load(文件句柄)
    except Exception:
        return None


def _保存话题分割(任务ID: str, 片段列表: list) -> bool:
    """保存话题分割结果到 数据/分割片段/<任务ID>_话题分割.json。"""
    分割片段目录 = _取目录("数据预处理.分割片段目录", "数据/分割片段")
    文件 = os.path.join(分割片段目录, f"{任务ID}_话题分割.json")
    try:
        with open(文件, "w", encoding="utf-8") as 文件句柄:
            json.dump(片段列表, 文件句柄, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


# ==================================================================
# 一、上传并清洗
# ==================================================================


def 上传并清洗(文件路径: str, 类型: str) -> dict:
    """接收上传文件并完成清洗（去噪/去重/编码统一/格式校验）。

    参数:
        文件路径: 上传文件的绝对路径。
        类型: 输入类型，取值 "视频" / "音频" / "文本"。

    返回:
        dict：{"任务ID", "状态", "路径", "文件路径", "类型", "文件大小MB",
               "清洗结果", "成功"}；失败时含 "错误" 字段。
    """
    if 类型 not in ("视频", "音频", "文本"):
        return {
            "任务ID": "", "状态": "失败", "路径": "", "类型": 类型,
            "文件大小MB": 0, "清洗结果": {},
            "成功": False, "错误": f"不支持的输入类型：{类型}（应为 视频/音频/文本）",
        }
    if not os.path.isfile(文件路径):
        return {
            "任务ID": "", "状态": "失败", "路径": 文件路径, "类型": 类型,
            "文件大小MB": 0, "清洗结果": {},
            "成功": False, "错误": f"上传文件不存在：{文件路径}",
        }
    大小 = os.path.getsize(文件路径)
    if 大小 <= 0:
        return {
            "任务ID": "", "状态": "失败", "路径": 文件路径, "类型": 类型,
            "文件大小MB": 0, "清洗结果": {},
            "成功": False, "错误": "文件大小为 0，无法处理",
        }

    任务ID = _生成任务ID()
    上传目录 = _取目录("数据预处理.输入目录", "数据/上传")
    原始名 = os.path.basename(文件路径)
    目标路径 = os.path.join(上传目录, f"{任务ID}_{原始名}")
    try:
        shutil.copy2(文件路径, 目标路径)
    except OSError as 错误:
        return {
            "任务ID": 任务ID, "状态": "失败", "路径": 文件路径, "类型": 类型,
            "文件大小MB": round(大小 / 1024 / 1024, 2), "清洗结果": {},
            "成功": False, "错误": f"复制文件失败：{错误}",
        }

    大小MB = round(大小 / 1024 / 1024, 2)
    if 类型 == "文本":
        try:
            原始文本 = _读文本文件(文件路径)
            清洗后文本, 清洗统计 = _清洗文本(原始文本)
            清洗统计["编码"] = "自动检测"
            清洗统计["原文件行数"] = 原始文本.count("\n") + 1
            清洗结果 = 清洗统计
        except OSError as 错误:
            清洗结果 = {"清洗失败": str(错误)}
    else:
        # 视频/音频基础校验：扩展名合法 + 时长探测
        扩展名 = os.path.splitext(文件路径)[1].lower()
        合法扩展名 = (扩展名 in 视频扩展名) if 类型 == "视频" else (扩展名 in 音频扩展名)
        时长秒 = _探测时长(目标路径)
        清洗结果 = {
            "扩展名合法": 合法扩展名,
            "文件扩展名": 扩展名 or "（无扩展名）",
            "时长秒": 时长秒,
            "格式校验": "通过" if 合法扩展名 else "失败",
        }

    return {
        "任务ID": 任务ID,
        "状态": "已上传",
        "文件路径": 目标路径,
        "路径": 目标路径,
        "类型": 类型,
        "文件大小MB": 大小MB,
        "清洗结果": 清洗结果,
        "成功": True,
    }


# ==================================================================
# 二、音频转文本
# ==================================================================


def 音频转文本(音频路径: str, 进度回调=None) -> dict:
    """调用配置的全模态 ASR 模型（配置项 模型.ASR模型）分段转写音频。

    参数:
        音频路径: 音频/视频文件绝对路径。
        进度回调: 可选回调函数 进度回调(进度: float, 消息: str)。

    返回:
        dict：{"成功", "任务ID", "音频路径", "文本", "分段",
               "时间戳列表", "总时长秒", "状态"}；失败时含 "错误" 字段。
    """
    if not os.path.isfile(音频路径):
        return {"成功": False, "错误": f"音频文件不存在：{音频路径}"}

    任务ID = _从路径取任务ID(音频路径)
    打标结果目录 = _取目录("打标.打标结果目录", "数据/打标结果")
    转写文件 = os.path.join(打标结果目录, f"{任务ID}_转写.json")

    # 断点续转：重跑同任务时若转写文件已存在直接返回
    if os.path.exists(转写文件):
        try:
            with open(转写文件, "r", encoding="utf-8") as 文件句柄:
                已有结果 = json.load(文件句柄)
            print(f"[转写] 命中断点续转：{转写文件}")
            return {**已有结果, "断点续转": True}
        except Exception:
            pass

    模型ID = 获取配置项("模型.ASR模型", "")
    if not 模型ID:
        return {"成功": False, "错误": "未配置ASR模型（需支持音频解码器的全模态模型）"}
    模型类型 = 获取配置项("模型.ASR模型类型", "")

    # 解码音频（获取时长与分段边界）
    try:
        音频, 采样率 = _读取音频(音频路径)
    except Exception as 错误:
        return {"成功": False, "错误": f"音频解码失败：{错误}"}
    总时长秒 = round(len(音频) / 采样率, 2)
    if 总时长秒 <= 0:
        return {"成功": False, "错误": "音频为空或时长为零"}

    # 段长：默认 30 秒，限制在 [最小片段秒, 最大片段秒] 内
    最小片段秒 = float(获取配置项("数据预处理.最小片段秒", 5))
    最大片段秒 = float(获取配置项("数据预处理.最大片段秒", 300))
    段长秒 = max(最小片段秒, min(30.0, 最大片段秒))
    总段数 = max(1, math.ceil(总时长秒 / 段长秒))
    段边界 = [
        {"起始秒": round(起始秒, 2), "结束秒": round(结束秒, 2)}
        for _, 起始秒, 结束秒 in _音频切段(音频, 采样率, 段长秒)
    ]

    # 调用独立 ASR worker（Qwen2.5-Omni 需 transformers 4.5x，与主环境 5.x 隔离）
    进度 = 0.05
    _更新进度(任务ID, 进度, "启动 ASR 转写进程（加载全模态模型）")
    if 进度回调:
        try:
            进度回调(进度, "启动 ASR 转写进程（加载全模态模型）")
        except Exception:
            pass
    结果 = _调用ASRWorker(模型ID, 音频路径, 段边界, 任务ID)
    if not 结果.get("成功"):
        return {"成功": False, "错误": 结果.get("错误", "ASR 转写失败")}

    分段 = 结果.get("分段", [])
    文本 = (结果.get("文本") or "").strip()
    时间戳列表 = [
        {"开始秒": 段["起始秒"], "结束秒": 段["结束秒"], "文本": 段["文本"]}
        for 段 in 分段
    ]
    结果 = {
        "成功": True,
        "任务ID": 任务ID,
        "音频路径": 音频路径,
        "文本": 文本,
        "转写文本": 文本,
        "分段": 分段,
        "时间戳列表": 时间戳列表,
        "总时长秒": 总时长秒,
        "状态": "完成",
        "算法": "全模态模型分段转写（独立进程）",
        "耗时秒": 结果.get("耗时秒"),
    }

    # 断点续转落盘
    try:
        with open(转写文件, "w", encoding="utf-8") as 文件句柄:
            json.dump(结果, 文件句柄, ensure_ascii=False, indent=2)
    except OSError as 错误:
        print(f"[转写] 转写结果写入失败：{错误}")
    _更新进度(任务ID, 1.0, "转写完成")
    if 进度回调:
        try:
            进度回调(1.0, "转写完成")
        except Exception:
            pass
    return 结果


# ==================================================================
# 三、话题分割（核心：按话题语义切分）
# ==================================================================


def _找切分点(分句时间戳: list, 算法: str, 阈值: float) -> list:
    """根据相邻句相似度（或降级规则）找话题切换点，返回句索引列表。"""
    切分点 = [0]
    if len(分句时间戳) <= 1:
        return 切分点
    最小片段秒 = float(获取配置项("数据预处理.最小片段秒", 5))
    if "jieba" in 算法:
        try:
            import jieba
            jieba.setLogLevel(20)
            词集列表 = [set(_分词(文本)) for _, _, 文本 in 分句时间戳]
            for i in range(1, len(分句时间戳)):
                if _词集相似度(词集列表[i - 1], 词集列表[i]) < 阈值:
                    切分点.append(i)
            return 切分点
        except Exception:
            pass
    # 降级：时间间隙 + 话题转换词
    for i in range(1, len(分句时间戳)):
        间隙 = 分句时间戳[i][0] - 分句时间戳[i - 1][1]
        if 间隙 >= max(3.0, 最小片段秒 * 0.5) or _含话题转换词(分句时间戳[i][2]):
            切分点.append(i)
    return 切分点


def 合并片段时间戳(分段: list) -> list:
    """内部辅助：规范化时间戳列表并应用 合并过短 / 切分过长 规则。

    参数:
        分段: 时间戳列表，元素可为 dict（起始秒/结束秒/文本）或
              [起始秒, 结束秒, 文本]。

    返回:
        list，元素为 dict：{"起始秒", "结束秒", "文本"}。
    """
    规范化 = _规范化时间戳(分段)
    if not 规范化:
        return []
    最小片段秒 = float(获取配置项("数据预处理.最小片段秒", 5))
    最大片段秒 = float(获取配置项("数据预处理.最大片段秒", 300))

    def _合并两段(段A: dict, 段B: dict) -> dict:
        return {
            "起始秒": min(段A["起始秒"], 段B["起始秒"]),
            "结束秒": max(段A["结束秒"], 段B["结束秒"]),
            "文本": 段A["文本"] + 段B["文本"],
        }

    # 第一步：合并过短片段（< 最小片段秒 的并入邻近片段）
    结果 = []
    待合并 = None
    for 片段 in 规范化:
        片段时长 = 片段["结束秒"] - 片段["起始秒"]
        if 片段时长 < 最小片段秒:
            if 结果:
                结果[-1] = _合并两段(结果[-1], 片段)
            else:
                # 首个片段过短：暂存并与后一片段合并
                待合并 = _合并两段(待合并, 片段) if 待合并 is not None else dict(片段)
            continue
        if 待合并 is not None:
            片段 = _合并两段(待合并, 片段)
            待合并 = None
        结果.append(dict(片段))
    if 待合并 is not None:
        if 结果:
            结果[-1] = _合并两段(结果[-1], 待合并)
        else:
            结果.append(待合并)

    # 第二步：切分过长片段（> 最大片段秒 的按句内切分）
    最终 = []
    for 片段 in 结果:
        if (片段["结束秒"] - 片段["起始秒"]) <= 最大片段秒:
            最终.append(片段)
            continue
        句子 = _分句(片段["文本"])
        if len(句子) <= 1:
            最终.append(片段)
            continue
        时长 = 片段["结束秒"] - 片段["起始秒"]
        每句时长 = 时长 / len(句子)
        子片段 = []
        当前文本 = ""
        当前起 = 片段["起始秒"]
        for 句号, (句文本, _, _) in enumerate(句子):
            句起 = 片段["起始秒"] + 句号 * 每句时长
            句止 = 片段["起始秒"] + (句号 + 1) * 每句时长
            if 当前文本 and (句止 - 当前起) > 最大片段秒:
                子片段.append({"起始秒": 当前起, "结束秒": 句起, "文本": 当前文本})
                当前文本 = ""
                当前起 = 句起
            当前文本 += 句文本
        if 当前文本:
            子片段.append({"起始秒": 当前起, "结束秒": 片段["结束秒"], "文本": 当前文本})
        最终.extend(子片段 or [片段])
    return 最终


def 话题分割(转写文本: str, 时间戳列表: list) -> list:
    """基于转写文本做语义分段（按话题而非按时长切分）。

    参数:
        转写文本: 完整转写文本。
        时间戳列表: 音频转文本 返回的时间戳列表（或 [[起,止,文本]...]）。

    返回:
        list，元素为 dict：{"片段ID", "话题ID", "话题摘要", "开始秒",
               "起始秒", "结束秒", "时长秒", "文本", "边界可调", "算法"}。
    """
    转写文本 = (转写文本 or "").strip()
    if not 转写文本:
        return []
    句子 = _分句(转写文本)
    if not 句子:
        return []

    总时长 = _时间戳总时长(时间戳列表)
    总字符数 = len(转写文本)
    # 句子 → 时间戳 [[起始秒, 结束秒, 文本]]
    分句时间戳 = [
        [_字符位置到时间(起, 总字符数, 总时长),
         _字符位置到时间(止, 总字符数, 总时长), 文本]
        for 文本, 起, 止 in 句子
    ]

    # 相似度算法（jieba 可用与否决定算法名与切分逻辑）
    阈值 = float(获取配置项("数据预处理.话题相似度阈值", 0.15))
    try:
        import jieba
        jieba.setLogLevel(20)
        算法 = "jieba+词集重叠相似度"
    except ImportError:
        算法 = "标点+时间间隙（jieba缺失降级）"

    切分点 = _找切分点(分句时间戳, 算法, 阈值)
    # 按切分点分组为初步话题组
    切分点 = sorted(set(切分点))
    if not 切分点:
        切分点 = [0]
    话题组 = []
    for i, 起点 in enumerate(切分点):
        终点 = 切分点[i + 1] if i + 1 < len(切分点) else len(分句时间戳)
        if 终点 > 起点:
            话题组.append(分句时间戳[起点:终点])

    # 合并过短 / 切分过长
    片段列表 = 合并片段时间戳(
        [
            {"起始秒": 组[0][0], "结束秒": 组[-1][1],
             "文本": "".join(句[2] for 句 in 组)}
            for 组 in 话题组
        ]
    )

    结果 = []
    for i, 片段 in enumerate(片段列表, 1):
        片段ID = f"seg_{i:04d}"
        结果.append({
            "片段ID": 片段ID,
            "话题ID": 片段ID,
            "话题摘要": _生成话题摘要(片段["文本"], 算法),
            "开始秒": 片段["起始秒"],
            "起始秒": 片段["起始秒"],
            "结束秒": 片段["结束秒"],
            "时长秒": round(片段["结束秒"] - 片段["起始秒"], 2),
            "文本": 片段["文本"],
            "边界可调": True,
            "算法": 算法,
        })
    return 结果


# ==================================================================
# 四、分割预览 / 调整片段边界
# ==================================================================


def 分割预览(任务ID: str) -> dict:
    """返回某任务的片段列表 + 时间线信息，供前端展示与微调边界。

    若该任务尚无话题分割结果，则从转写结果现场分割并保存。
    """
    分割片段目录 = _取目录("数据预处理.分割片段目录", "数据/分割片段")
    话题文件 = os.path.join(分割片段目录, f"{任务ID}_话题分割.json")
    片段列表 = []
    if os.path.exists(话题文件):
        try:
            with open(话题文件, "r", encoding="utf-8") as 文件句柄:
                片段列表 = json.load(文件句柄)
        except Exception:
            片段列表 = []
    if not 片段列表:
        转写结果 = _读转写结果(任务ID)
        if 转写结果 and 转写结果.get("文本"):
            片段列表 = 话题分割(
                转写结果.get("文本", ""), 转写结果.get("分段", [])
            )
            _保存话题分割(任务ID, 片段列表)
    时间线 = {
        "任务ID": 任务ID,
        "总时长秒": 片段列表[-1]["结束秒"] if 片段列表 else 0.0,
        "片段数": len(片段列表),
        "可调边界": True,
    }
    return {"任务ID": 任务ID, "片段列表": 片段列表, "时间线": 时间线, "状态": "完成"}


def 调整片段边界(任务ID: str, 边界列表: list) -> bool:
    """人工微调边界后重存（把边界列表应用到片段并重算摘要）。

    参数:
        任务ID: 任务唯一标识。
        边界列表: [{"话题ID", "起始秒", "结束秒"}, ...] 或兼容 "开始秒"。

    返回:
        bool：保存成功返回 True，否则 False。
    """
    if not 边界列表:
        return False
    分割片段目录 = _取目录("数据预处理.分割片段目录", "数据/分割片段")
    话题文件 = os.path.join(分割片段目录, f"{任务ID}_话题分割.json")
    try:
        with open(话题文件, "r", encoding="utf-8") as 文件句柄:
            原片段 = json.load(文件句柄)
    except Exception:
        return False
    try:
        新边界 = sorted(
            边界列表,
            key=lambda x: float(x.get("起始秒") or x.get("开始秒") or 0),
        )
    except Exception:
        return False

    新片段 = []
    for i, 边界 in enumerate(新边界, 1):
        起始 = float(边界.get("起始秒") or 边界.get("开始秒") or 0)
        结束 = float(边界.get("结束秒") or 0)
        片段ID = f"seg_{i:04d}"
        文本 = _按时间取文本(原片段, 起始, 结束)
        新片段.append({
            "片段ID": 片段ID,
            "话题ID": 片段ID,
            "话题摘要": _生成话题摘要(文本, "人工微调"),
            "开始秒": 起始,
            "起始秒": 起始,
            "结束秒": 结束,
            "时长秒": round(结束 - 起始, 2),
            "文本": 文本,
            "边界可调": True,
            "人工调整": True,
        })
    try:
        with open(话题文件, "w", encoding="utf-8") as 文件句柄:
            json.dump(新片段, 文件句柄, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


# ==================================================================
# 五、HTTP 路由注册
# ==================================================================


def _后台转写(音频路径: str, 任务ID: str) -> None:
    """后台任务：执行转写并更新进度缓存。"""
    _更新进度(任务ID, 0.0, "准备开始转写")
    try:
        结果 = 音频转文本(
            音频路径,
            进度回调=lambda 进度, 消息: _更新进度(任务ID, 进度, 消息),
        )
        with _进度锁:
            _转写进度[任务ID]["结果"] = 结果
            _转写进度[任务ID]["状态"] = "完成" if 结果.get("成功") else "失败"
            _转写进度[任务ID]["消息"] = (
                "转写完成" if 结果.get("成功") else 结果.get("错误", "转写失败")
            )
    except Exception as 错误:
        with _进度锁:
            _转写进度[任务ID]["状态"] = "失败"
            _转写进度[任务ID]["消息"] = f"转写异常：{错误}"


def 注册路由(app) -> None:
    """注册数据预处理模块的 HTTP 路由（挂载到 FastAPI 应用）。

    - POST /api/预处理/上传        multipart 文件 + 类型字段
    - POST /api/预处理/转写        body：任务ID 或 音频路径（后台异步）
    - GET  /api/预处理/转写/进度   ?任务ID=
    - POST /api/预处理/话题分割    body：任务ID
    - GET  /api/预处理/预览        ?任务ID=
    - POST /api/预处理/调整边界    body：任务ID + 边界列表
    """
    from fastapi import BackgroundTasks, File, Form, Query, UploadFile
    from pydantic import BaseModel

    class 转写请求(BaseModel):
        任务ID: str = ""
        音频路径: str = ""

    class 话题分割请求(BaseModel):
        任务ID: str

    class 调整边界请求(BaseModel):
        任务ID: str
        边界列表: list

    @app.post("/api/预处理/上传")
    async def 上传接口(文件: UploadFile = File(...), 类型: str = Form(...)):
        上传目录 = _取目录("数据预处理.输入目录", "数据/上传")
        原始名 = 文件.filename or "上传文件"
        临时路径 = os.path.join(上传目录, 原始名)
        if os.path.exists(临时路径):
            临时路径 = os.path.join(上传目录, f"{_生成任务ID()}_{原始名}")
        try:
            内容 = await 文件.read()
            with open(临时路径, "wb") as 文件句柄:
                文件句柄.write(内容)
            return 上传并清洗(临时路径, 类型)
        except Exception as 错误:
            return {"成功": False, "状态": "失败", "错误": f"上传失败：{错误}"}
        finally:
            try:
                if os.path.exists(临时路径):
                    os.remove(临时路径)
            except OSError:
                pass

    @app.post("/api/预处理/转写")
    def 转写接口(请求: 转写请求, 后台任务: BackgroundTasks):
        任务ID = 请求.任务ID.strip()
        音频路径 = 请求.音频路径.strip()
        if 音频路径:
            if not os.path.isabs(音频路径):
                音频路径 = 解析路径(音频路径)
        elif 任务ID:
            上传目录 = _取目录("数据预处理.输入目录", "数据/上传")
            候选 = [
                os.path.join(上传目录, 文件名)
                for 文件名 in os.listdir(上传目录)
                if 文件名.startswith(任务ID + "_")
            ]
            if not 候选:
                return {
                    "成功": False, "状态": "失败",
                    "错误": f"未找到任务 {任务ID} 的上传文件",
                }
            音频路径 = 候选[0]
        else:
            return {
                "成功": False, "状态": "失败",
                "错误": "请提供 任务ID 或 音频路径",
            }
        后台任务.add_task(_后台转写, 音频路径, 任务ID or _从路径取任务ID(音频路径))
        实际任务ID = 任务ID or _从路径取任务ID(音频路径)
        return {
            "任务ID": 实际任务ID,
            "状态": "转写中",
            "消息": "转写已提交后台执行，可查询进度",
        }

    @app.get("/api/预处理/转写/进度")
    def 转写进度接口(任务ID: str = Query(...)):
        进度 = _转写进度.get(任务ID)
        if not 进度:
            return {"任务ID": 任务ID, "进度": 0.0, "状态": "未开始", "消息": ""}
        return {
            "任务ID": 任务ID,
            "进度": 进度["进度"],
            "消息": 进度["消息"],
            "状态": 进度["状态"],
        }

    @app.post("/api/预处理/话题分割")
    def 话题分割接口(请求: 话题分割请求):
        转写结果 = _读转写结果(请求.任务ID)
        if not 转写结果 or not 转写结果.get("文本"):
            return {
                "成功": False, "状态": "失败",
                "错误": f"任务 {请求.任务ID} 无转写结果，请先执行转写",
            }
        片段列表 = 话题分割(转写结果.get("文本", ""), 转写结果.get("分段", []))
        _保存话题分割(请求.任务ID, 片段列表)
        return {
            "任务ID": 请求.任务ID,
            "片段数": len(片段列表),
            "片段列表": 片段列表,
            "状态": "完成",
            "成功": True,
        }

    @app.get("/api/预处理/预览")
    def 预览接口(任务ID: str = Query(...)):
        return 分割预览(任务ID)

    @app.post("/api/预处理/调整边界")
    def 调整边界接口(请求: 调整边界请求):
        成功 = 调整片段边界(请求.任务ID, 请求.边界列表)
        return {
            "任务ID": 请求.任务ID,
            "状态": "已保存" if 成功 else "保存失败",
            "成功": 成功,
        }
