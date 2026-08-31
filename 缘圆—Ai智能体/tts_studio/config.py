# -*- coding: utf-8 -*-
"""配置持久化：模型/音色别名（重命名）、风格预设、最近使用的参数，存于 config.json。"""
import copy
import json
import os
import sys
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def app_dir():
    """可写数据目录：打包后为 EXE 所在目录（config.json/audio_cache/.env 放这里），开发时为项目目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return BASE_DIR


def res_dir():
    """只读资源目录：打包后为 PyInstaller 解包目录（static/.env 等），开发时为项目目录。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", BASE_DIR))
    return BASE_DIR


CONFIG_PATH = app_dir() / "config.json"


def load_env_file(path=None):
    """加载 .env 到环境变量（不覆盖已存在的环境变量）。优先 EXE 旁，其次打包资源。"""
    if path is None:
        p = app_dir() / ".env"
        if not p.exists():
            p = res_dir() / ".env"
    else:
        p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v

DEFAULTS = {
    "model_aliases": {},          # 模型ID -> 显示别名（重命名结果）
    "voice_aliases": {},          # 音色ID -> 显示别名
    "style_presets": [            # 全局风格指令预设
        {"name": "默认·温柔自然",
         "instruction": "用温柔、清澈、带一点俏皮的少女语气，自然地说话。"},
        {"name": "睡前故事",
         "instruction": "语速缓慢轻柔，像哄孩子入睡一样，声音低柔温暖，尾音放轻。"},
        {"name": "情绪递进·独白",
         "instruction": "女声独白，情绪层层递进：开头委屈不安、声音微微发颤；"
                        "讲到害怕时呼吸加重；然后如释重负、带着笑意；"
                        "最后转为俏皮撒娇又带一点温柔。语速自然，感情真挚。"},
    ],
    "last_used": {
        "model": "qwen-audio-3.0-tts-plus",
        "voice": "qwen-audio-3.0-tts-plus-yuanyuan-c6cf949d19734ab5a5552a9c5ce2da9f",
        "format": "wav",
        "sample_rate": 48000,
    },
    "pronunciations": [          # 发音纠正表：词 -> 带声调拼音（合成时自动注音）
        {"word": "缘圆", "ph": "yuan3 yuan4"},
    ],
}

_lock = threading.Lock()
_data = None


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load():
    global _data
    with _lock:
        if _data is None:
            if CONFIG_PATH.exists():
                try:
                    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                except Exception:
                    raw = {}
                _data = _deep_merge(DEFAULTS, raw)
            else:
                _data = copy.deepcopy(DEFAULTS)
        return _data


def save():
    with _lock:
        CONFIG_PATH.write_text(
            json.dumps(_data, ensure_ascii=False, indent=2), encoding="utf-8")


def model_alias(model_id):
    return load()["model_aliases"].get(model_id, model_id)


def voice_alias(voice_id):
    return load()["voice_aliases"].get(voice_id, voice_id)


def rename_model(model_id, alias):
    d = load()
    alias = (alias or "").strip()
    if alias:
        d["model_aliases"][model_id] = alias
    else:
        d["model_aliases"].pop(model_id, None)
    save()


def rename_voice(voice_id, alias):
    d = load()
    alias = (alias or "").strip()
    if alias:
        d["voice_aliases"][voice_id] = alias
    else:
        d["voice_aliases"].pop(voice_id, None)
    save()


def style_presets():
    return load()["style_presets"]


def add_style_preset(name, instruction):
    d = load()
    name = (name or "").strip()
    if not name:
        raise ValueError("预设名称不能为空")
    for p in d["style_presets"]:
        if p["name"] == name:
            p["instruction"] = instruction
            save()
            return p
    p = {"name": name, "instruction": instruction}
    d["style_presets"].append(p)
    save()
    return p


def delete_style_preset(name):
    d = load()
    d["style_presets"] = [p for p in d["style_presets"] if p["name"] != name]
    save()


def last_used():
    return load()["last_used"]


def set_last_used(**kw):
    d = load()
    d["last_used"].update({k: v for k, v in kw.items() if v is not None})
    save()


def pronunciations():
    return load()["pronunciations"]


def upsert_pronunciation(word, ph):
    """新增或更新一条发音纠正。返回 (entry, 是否新增)。"""
    d = load()
    word = (word or "").strip()
    ph = (ph or "").strip()
    if not word or not ph:
        raise ValueError("词和拼音都不能为空")
    for p in d["pronunciations"]:
        if p["word"] == word:
            p["ph"] = ph
            save()
            return p, False
    p = {"word": word, "ph": ph}
    d["pronunciations"].append(p)
    save()
    return p, True


def delete_pronunciation(word):
    d = load()
    d["pronunciations"] = [p for p in d["pronunciations"] if p["word"] != word]
    save()
