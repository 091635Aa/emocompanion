# -*- coding: utf-8 -*-
"""缘圆 · 统一控制台日志系统

功能:
  - 按 category(模块/功能域) + level 分类记录
  - 文件按天轮转, 单条日志 JSON 行格式便于搜索
  - 支持文本/关键字/时间范围/分类/级别查询
  - 保留人类可读的控制台输出

用法:
  from logger import get_logger
  log = get_logger("chat")
  log.info("收到用户消息", user="xxx")
"""
import json
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 分类常量(前端搜索下拉用)
CATEGORIES = [
    "system", "chat", "tts", "text_engine", "emotion", "role",
    "session", "mount", "debug", "api", "startup", "error"
]

# 彩色控制台
_COLOR = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[35m",  # magenta
    "RESET": "\033[0m",
}


class _JsonFormatter(logging.Formatter):
    """把日志记录序列化成 JSON 行, 包含 category 与 extra 字段。"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        msec = int(record.msecs)
        entry = {
            "t": f"{ts}.{msec:03d}",
            "ts": record.created,
            "level": record.levelname,
            "category": getattr(record, "category", "system"),
            "msg": record.getMessage(),
            "src": f"{record.filename}:{record.lineno}",
        }
        # 合并 extra 字段
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            entry.update(extra)
        # 合并 log() 关键字参数
        for k in ("user", "sid", "emotion", "backend", "err", "duration", "model", "role", "route_emo"):
            if hasattr(record, k):
                entry[k] = getattr(record, k)
        return json.dumps(entry, ensure_ascii=False)


class _ConsoleFormatter(logging.Formatter):
    """带分类与颜色的控制台格式。"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        cat = getattr(record, "category", "system")
        color = _COLOR.get(record.levelname, "")
        reset = _COLOR["RESET"]
        if sys.platform == "win32" and not os.environ.get("TERM"):
            color = reset = ""
        return f"{color}[{ts}][{record.levelname:8s}][{cat:12s}] {record.getMessage()}{reset}"


class _ExtraAdapter(logging.LoggerAdapter):
    """支持 log.info("msg", key=value) 写 extra 的适配器。"""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra["category"] = self.extra.get("category", "system")
        for k, v in list(kwargs.items()):
            if k not in ("exc_info", "stack_info", "extra"):
                extra[k] = v
                kwargs.pop(k)
        kwargs["extra"] = extra
        return msg, kwargs


# 根 logger
_ROOT = logging.getLogger("yy.console")
_ROOT.setLevel(logging.DEBUG)
# 避免重复添加 handler(模块重载时)
if not _ROOT.handlers:
    # JSON 文件按天轮转
    fh = logging.handlers.TimedRotatingFileHandler(
        str(LOG_DIR / "console.log"), when="midnight", interval=1, backupCount=7,
        encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JsonFormatter())
    _ROOT.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ConsoleFormatter())
    _ROOT.addHandler(ch)

_loggers = {}
_lock = threading.Lock()


def get_logger(category: str = "system") -> _ExtraAdapter:
    """获取带 category 的日志适配器。"""
    if category not in _loggers:
        with _lock:
            if category not in _loggers:
                _loggers[category] = _ExtraAdapter(_ROOT, {"category": category})
    return _loggers[category]


def _walk_log_files(days: int = 7) -> List[Path]:
    """收集最近 N 天的日志文件(console.log 及轮转文件)。"""
    files = []
    cutoff = time.time() - days * 86400
    for p in LOG_DIR.glob("console.log*"):
        try:
            if p.stat().st_mtime >= cutoff:
                files.append(p)
        except Exception:
            continue
    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)


def query_logs(
    category: Optional[str] = None,
    level: Optional[str] = None,
    search: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    days: int = 7,
) -> dict:
    """查询日志文件, 返回结构化结果。"""
    results = []
    total = 0
    pattern = re.compile(re.escape(search), re.I) if search else None

    def _match(entry: dict) -> bool:
        nonlocal total
        if category and entry.get("category") != category:
            return False
        if level and entry.get("level") != level.upper():
            return False
        if start and entry.get("t", "") < start:
            return False
        if end and entry.get("t", "") > end:
            return False
        if pattern:
            text = json.dumps(entry, ensure_ascii=False)
            if not pattern.search(text):
                return False
        return True

    for p in _walk_log_files(days):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if not _match(entry):
                        continue
                    total += 1
                    if total > offset and len(results) < limit:
                        results.append(entry)
        except Exception:
            continue
    return {"total": total, "offset": offset, "limit": limit, "logs": results}


def get_stats(days: int = 1) -> dict:
    """返回日志级别与分类统计。"""
    level_counts = {}
    cat_counts = {}
    for p in _walk_log_files(days):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    level_counts[e.get("level", "INFO")] = level_counts.get(e.get("level", "INFO"), 0) + 1
                    cat_counts[e.get("category", "system")] = cat_counts.get(e.get("category", "system"), 0) + 1
        except Exception:
            continue
    return {"levels": level_counts, "categories": cat_counts}


if __name__ == "__main__":
    get_logger("chat").info("测试日志", user="tester", sid="sess_xxx")
    get_logger("tts").warning("模型未加载", backend="gguf")
    print(query_logs(search="测试", limit=5))
