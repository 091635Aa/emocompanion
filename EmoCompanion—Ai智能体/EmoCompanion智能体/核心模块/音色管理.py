# -*- coding: utf-8 -*-
"""音色管理：系统音色 + 本地复刻 + 在线音色的聚合与标签。"""
from . import 配置持久化, 实时通话, 语音合成

# 可绑定 omni 复刻音色并用于实时通话的全模态实时模型
实时通话模型集合 = set(实时通话.实时模型ID列表)


def 系统音色列表():
    """内置系统音色（来自 语音合成）。"""
    return 语音合成.系统音色列表


def 本地复刻音色():
    """本地复刻音色（来自 语音合成，读取 voice_id*.txt）。"""
    return 语音合成.本地复刻音色()


def 在线音色(密钥=""):
    """DashScope 在线查询的复刻音色（失败时返回空列表）。"""
    return 语音合成.查询在线音色(密钥)


def 全部音色(模型ID=""):
    """聚合全部音色：系统 + 本地复刻 + 在线，按音色ID去重。

    每个条目保留原始字段（id/name/lang/kind/source/target_model），并补充「别名」字段；
    显示名称 name 优先取别名，其次取原始名称。

    通话可用性按声音复刻时的绑定模型（target_model）复核：
    - 在线复刻音色：绑定模型 ∈ 实时通话模型集合 → 通话可用=True，否则仅合成；
    - 本地 omni 复刻音色（qwen-omni-vc-*）：若在线可查到绑定模型则据此复核，
      避免误把绑定非实时模型（如 qwen3.5-omni-flash）的音色用于通话导致合成失败。
    """
    别名表 = 配置持久化.读取配置()["音色别名"]
    输出 = []
    已见 = set()

    # 先查在线音色，构建 音色ID → target_model 映射，供本地复刻复核
    try:
        在线音色们 = 在线音色(密钥="")
    except Exception:
        在线音色们 = []
    在线绑定表 = {音色["id"]: (音色.get("target_model") or "") for 音色 in 在线音色们}

    def 附加(音色):
        条目 = dict(音色)
        条目["别名"] = 别名表.get(音色.get("id")) or ""
        条目["name"] = 条目["别名"] or 音色.get("name") or 音色.get("id")
        输出.append(条目)
        已见.add(音色["id"])

    for 音色 in 系统音色列表():
        条目 = dict(音色)
        条目["kind"] = "system"
        附加(条目)
    for 音色 in 本地复刻音色():
        if 音色["id"] in 已见:
            continue
        # 按在线绑定模型复核通话可用性；在线查不到时保留原有判定
        if 音色.get("通话可用"):
            目标模型 = 在线绑定表.get(音色["id"], "")
            if 目标模型:
                音色["通话可用"] = 目标模型 in 实时通话模型集合
        附加(音色)
    for 音色 in 在线音色们:
        if 音色["id"] in 已见:
            continue
        音色["通话可用"] = (音色.get("target_model") or "") in 实时通话模型集合
        附加(音色)
    隐藏们 = set(配置持久化.隐藏音色列表())
    return [条目 for 条目 in 输出 if 条目["id"] not in 隐藏们]


def 音色标签(音色, 别名表):
    """返回音色的显示标签：别名 -> 名称 -> 音色ID。"""
    if not 音色:
        return ""
    别名 = 别名表.get(音色.get("id")) if 音色.get("id") else None
    return 别名 or 音色.get("name") or 音色.get("id") or ""
