# -*- coding: utf-8 -*-
"""
下载器 — 模型下载（直链 / HuggingFace 公共 API 免密钥）
========================================================
- 直链下载：httpx 流式下载 + 进度
- HuggingFace 下载：huggingface_hub.snapshot_download（公共仓库免密钥），
  可选镜像（HF_ENDPOINT，如 https://hf-mirror.com 加速）
- 下载完成后自动注册模型（模型文件生成），前端轮询任务状态
"""
import sys
import os
import time
import uuid
import threading

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

from 模型管理 import 管理器, 模型空间目录


class 下载管理器:
    def __init__(self):
        self.任务 = {}   # 任务ID → {状态, 进度, 详情, 目标名, 目标目录}
        self._锁 = threading.Lock()

    def _新任务(self, 目标名, 目标目录):
        任务ID = uuid.uuid4().hex[:12]
        self.任务[任务ID] = {
            "任务ID": 任务ID, "状态": "排队中", "进度": 0.0,
            "详情": "", "目标名": 目标名, "目标目录": 目标目录,
        }
        return 任务ID

    def _更新(self, 任务ID, **kwargs):
        with self._锁:
            if 任务ID in self.任务:
                self.任务[任务ID].update(kwargs)

    def 下载直链(self, url, 目标名, 目标目录=None):
        """HTTP(S) 直链下载（后台线程），完成后自动注册"""
        目标目录 = 目标目录 or 模型空间目录
        目标路径 = os.path.join(目标目录, 目标名)
        os.makedirs(目标目录, exist_ok=True)
        任务ID = self._新任务(目标名, 目标目录)
        线程 = threading.Thread(
            target=self._直链线程, args=(任务ID, url, 目标路径), daemon=True)
        线程.start()
        return 任务ID

    def _直链线程(self, 任务ID, url, 目标路径):
        try:
            import httpx
            临时路径 = 目标路径 + ".下载中"
            self._更新(任务ID, 状态="下载中")
            with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                总大小 = int(r.headers.get("content-length") or 0)
                self._更新(任务ID, 详情=f"总大小 {总大小 / 1024 / 1024:.1f} MB")
                已下载 = 0
                with open(临时路径, "wb") as f:
                    for 块 in r.iter_bytes(chunk_size=1024 * 512):
                        f.write(块)
                        已下载 += len(块)
                        if 总大小:
                            self._更新(任务ID, 进度=round(已下载 / 总大小, 4))
                os.rename(临时路径, 目标路径)
            self._注册完成(任务ID)
        except Exception as e:
            self._更新(任务ID, 状态="失败", 详情=str(e))

    def 下载HuggingFace(self, 仓库名, 目标名=None, 镜像=None):
        """HuggingFace 公共仓库下载（免密钥）；镜像如 https://hf-mirror.com"""
        目标名 = 目标名 or 仓库名.replace("/", "--")
        目标目录 = os.path.join(模型空间目录, 目标名)
        任务ID = self._新任务(目标名, 目标目录)
        线程 = threading.Thread(
            target=self._HF线程, args=(任务ID, 仓库名, 目标目录, 镜像), daemon=True)
        线程.start()
        return 任务ID

    def _HF线程(self, 任务ID, 仓库名, 目标目录, 镜像):
        try:
            self._更新(任务ID, 状态="下载中")
            from huggingface_hub import snapshot_download
            kwargs = {"repo_id": 仓库名, "local_dir": 目标目录}
            if 镜像:
                os.environ["HF_ENDPOINT"] = 镜像
            # 过滤不必要的文件可减少下载量，默认全量
            snapshot_download(**kwargs)
            self._注册完成(任务ID)
        except Exception as e:
            self._更新(任务ID, 状态="失败", 详情=str(e))

    def _注册完成(self, 任务ID):
        任务 = self.任务.get(任务ID, {})
        目标名 = 任务.get("目标名")
        目标目录 = 任务.get("目标目录")
        try:
            描述 = 管理器.注册模型(目标名, 目标目录)
            self._更新(任务ID, 状态="完成", 进度=1.0,
                       详情=f"已注册（hidden_size={描述['hidden_size']}）")
        except Exception as e:
            self._更新(任务ID, 状态="完成", 进度=1.0,
                       详情=f"下载完成但注册失败: {e}")

    def 状态(self, 任务ID):
        return self.任务.get(任务ID, {"状态": "未知", "详情": "任务不存在"})

    def 全部状态(self):
        return [dict(v) for v in self.任务.values()]


下载器 = 下载管理器()
