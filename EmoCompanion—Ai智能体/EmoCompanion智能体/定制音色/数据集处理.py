# -*- coding: utf-8 -*-
"""数据集处理：把输入音频（单文件或目录）统一重采样为 24kHz 单声道 float32，
拼接为全长音频，并支持切分 30 秒分段。

输出约定（声音库目录下）：
  - 全长24k单声道.wav   拼接后的全长音频（PCM_16 / 24kHz / 单声道）
  - 数据集分段/partNNN.wav  若全长超过 10 分钟自动切分的 30 秒分段（可选）
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

采样率 = 24000                # 统一目标采样率
直接支持扩展名 = {".wav", ".flac"}   # soundfile 原生支持
需转码扩展名 = {".mp3"}              # 需 ffmpeg 转码
支持扩展名 = 直接支持扩展名 | 需转码扩展名
超过分钟提示阈值 = 10.0              # 超过该时长提示需截取


def _读取音频(路径, 目标采样率=采样率):
    """读取任意支持的音频，返回 (float32 单声道 24k 数组, 采样率)。

    - wav/flac 用 soundfile 直接读取；非 24k 用 scipy.signal.resample_poly 重采样。
    - mp3 用 ffmpeg 转成 24k 单声道 wav 后再读取；无 ffmpeg 时报清晰错误。
    """
    路径 = Path(路径)
    后缀 = 路径.suffix.lower()
    if 后缀 in 直接支持扩展名:
        数据, 原始采样率 = sf.read(路径, dtype="float32")
        if 数据.ndim > 1:
            数据 = 数据.mean(axis=1)   # 多声道取均值变单声道
        if 原始采样率 != 目标采样率:
            数据 = signal.resample_poly(数据, 目标采样率, 原始采样率).astype("float32")
        return 数据, 目标采样率
    if 后缀 in 需转码扩展名:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                f"检测到 mp3 输入（{路径.name}），但系统未安装 ffmpeg。"
                "请提供 wav/flac 格式，或安装 ffmpeg（https://ffmpeg.org）后重试。")
        with tempfile.TemporaryDirectory() as 临时目录:
            临时wav = Path(临时目录) / "转换.wav"
            命令 = [ffmpeg, "-y", "-v", "error", "-i", str(路径),
                    "-ar", str(目标采样率), "-ac", "1", "-f", "wav", str(临时wav)]
            结果 = subprocess.run(命令, capture_output=True, text=True)
            if 结果.returncode != 0 or not 临时wav.exists():
                raise RuntimeError(f"ffmpeg 转换 {路径.name} 失败：{结果.stderr.strip()[:300]}")
            数据, _采样率 = sf.read(临时wav, dtype="float32")
            if 数据.ndim > 1:
                数据 = 数据.mean(axis=1)
            return 数据, 目标采样率
    raise RuntimeError(f"不支持的音频格式：{后缀 or '（无扩展名）'}（支持 "
                       f"{'、'.join(sorted(直接支持扩展名))}，mp3 需安装 ffmpeg）")


def 分段(全长路径, 输出目录, 段时长秒=30):
    """把全长音频切成 段时长秒 左右的分段 part001.wav...

    最后一段不足 段时长秒*25% 时并入前一段，避免出现过短的尾巴。
    返回分段文件路径列表。
    """
    输出目录 = Path(输出目录)
    输出目录.mkdir(parents=True, exist_ok=True)
    数据, 采样率 = sf.read(全长路径, dtype="float32")
    if 数据.ndim > 1:
        数据 = 数据.mean(axis=1)
    每段样本 = int(段时长秒 * 采样率)
    总样本 = len(数据)
    段路径列表 = []
    起点 = 0
    序号 = 1
    while 起点 < 总样本:
        终点 = min(起点 + 每段样本, 总样本)
        剩余 = 总样本 - 终点
        # 剩余不足 25% 段长且不是第一段：并入当前段
        if 序号 > 1 and 剩余 > 0 and 剩余 < 每段样本 * 0.25:
            终点 = 总样本
        段 = 数据[起点:终点]
        路径 = 输出目录 / f"part{序号:03d}.wav"
        sf.write(路径, 段, 采样率, subtype="PCM_16")
        段路径列表.append(str(路径))
        if 终点 >= 总样本:
            break
        起点 = 终点
        序号 += 1
    return 段路径列表


def 处理(输入路径, 声音库目录):
    """把输入（单个音频文件或目录）统一处理为 24kHz 单声道全长音频。

    输入可以是：
      - 单个音频文件（wav/flac/mp3）
      - 目录（取其下所有支持的音频文件，按文件名排序后拼接）

    输出：
      - 声音库目录/全长24k单声道.wav
      - 若全长超过 10 分钟：自动切分到 声音库目录/数据集分段/ 并提示需截取

    返回 dict：{全长路径, 时长秒, 采样率, 声道, 分段数, 提示}
    """
    声音库目录 = Path(声音库目录)
    声音库目录.mkdir(parents=True, exist_ok=True)
    输入路径 = Path(输入路径)

    if 输入路径.is_dir():
        文件列表 = sorted(
            (路径 for 路径 in 输入路径.iterdir()
             if 路径.is_file() and 路径.suffix.lower() in 支持扩展名),
            key=lambda 路径: 路径.name)
        if not 文件列表:
            raise RuntimeError(f"目录 {输入路径} 下没有支持的音频文件"
                               f"（{'、'.join(sorted(支持扩展名))}）")
    elif 输入路径.is_file():
        文件列表 = [输入路径]
    else:
        raise RuntimeError(f"输入路径不存在：{输入路径}")

    片段列表 = []
    总时长 = 0.0
    处理明细 = []
    for 文件 in 文件列表:
        数据, 采样率 = _读取音频(文件)
        片段列表.append(数据)
        总时长 += len(数据) / 采样率
        处理明细.append({"文件": str(文件), "时长秒": round(len(数据) / 采样率, 3)})

    全长 = np.concatenate(片段列表) if len(片段列表) > 1 else 片段列表[0]
    全长路径 = 声音库目录 / "全长24k单声道.wav"
    sf.write(全长路径, 全长, 采样率, subtype="PCM_16")

    分段数 = 0
    提示 = ""
    if 总时长 > 超过分钟提示阈值 * 60:
        分段路径列表 = 分段(全长路径, 声音库目录 / "数据集分段", 段时长秒=30)
        分段数 = len(分段路径列表)
        提示 = (f"全长音频 {总时长 / 60:.1f} 分钟超过 {超过分钟提示阈值:.0f} 分钟，"
                f"建议截取信息密度高的段落；已自动切分为 {分段数} 段（数据集分段/）")

    返回 = {
        "全长路径": str(全长路径),
        "时长秒": round(总时长, 2),
        "采样率": 采样率,
        "声道": 1,
        "分段数": 分段数,
        "提示": 提示,
        "处理明细": 处理明细,
    }
    return 返回
