# -*- coding: utf-8 -*-
"""Qwen3-RapSynth · 学习型韵律预测器（Phase 2 实装，遵守 ProsodyPredictorBase 协议）

=== 定位（诚实标注） ===
当前无合规说唱数据集 + 未装 MFA。因此先用「规则先验(teacher) 生成逐音节伪标签」训练一个小型
LSTM，做**教师蒸馏/管道冒烟**：验证「训练 → 推理 → 注入」全链路可跑、可替换规则基线。
真实数据就绪后，仅需替换 `build_label_dataset()` 的数据来源为 MFA 对齐的真实 F0/时长，重训即可。

=== 模型 ===
输入：逐行歌词（中文字符序列）+ BPM + 风格
输出：每音节 3 个量—— F0 偏移(半音) / 时长缩放(相对规则) / 能量系数(0.6~1.6)
每行输出聚成与 LinePlan 兼容的字段，交给 `integration.injector` 消费。

=== 用法 ===
  python scripts/train_prosody_predictor.py --epochs 8 --device cpu
  推理：`LearnedProsodyPredictor(weights_path).predict(lyrics,bpm,style)`
  自动选型：`prosody_model.rules.get_predictor()` —— 有权重返回学习模型，否则规则基线
"""
import os
import math
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    import torch
except Exception:  # 本模块仅在学习模型启用时要求 torch，回退由调用方兜底
    torch = None

from .rules import ProsodyPlan, LinePlan, ProsodyPredictorBase, STYLES, count_syllables  # noqa: E402

# ---------------- 轻量词典（中文字符 + 常见标点 + 数字） ----------------
_BASE_CHARS = (
    "的我一了是不知道在你都没说就去来看这那又和被要还而小得也着之它现"
    "风月光年心节奏把这该她他话都地很你在同但等能到把玩花街巷落尘埃下"
    "夜灯亮沉思潮推腔回燃枪口利弊口瓣落绕想说对枪低砸撑开路开出在挥握"
    "任有啊哦嗯呀啊它怎回样日子多好大中上就身体很自过再想干声嗓音渗透"
    "不若难道哪个谁这样才当初然后因为所以可是但是而且还能更最刚就遍全"
)
_STYLE_IDS = {s: i for i, s in enumerate(STYLES)}
_VOCAB = {"<unk>": 0, "<pad>": 1}
for _c in _BASE_CHARS:
    _VOCAB.setdefault(_c, len(_VOCAB))
_VOCAB_SIZE = len(_VOCAB)


def _char_ids(line: str) -> List[int]:
    return [_VOCAB.get(c, _VOCAB["<unk>"]) for c in line]


def _syll_f0_delta(style: str, i: int, n: int) -> float:
    """规则 teacher 的逐音节 F0 偏移（半音），把行级轮廓细化到音节。"""
    if style == "旋律说唱":
        t = i / max(n - 1, 1)
        return float(3.0 * math.sin(math.pi * t))      # 拱形 ±3 半音
    if style == "硬核":
        return float(-2.0 if i % 2 == 0 else 1.5)       # 下探 + 弱拍上抬
    return float(0.35 if i % 2 == 1 else -0.2)          # 快嘴：微量起伏


def _syll_dur_scale(style: str, i: int, n: int) -> float:
    if style == "快嘴":
        return 0.85 if i % 2 == 0 else 1.15             # 疏密交替
    if style == "硬核":
        return 1.0
    return 1.0 + 0.15 * math.sin(math.pi * i / max(n - 1, 1))


def _syll_energy(style: str, i: int, n: int, line_energy: float) -> float:
    beat = 1.30 if i % 2 == 0 else 0.75                 # 强拍/弱拍
    return float(np.clip(line_energy * beat, 0.5, 1.7))


# ---------------- 模型 ----------------
def build_model(n_vocab=_VOCAB_SIZE, n_style=len(STYLES),
                emb_dim=40, hidden=64):
    import torch
    from torch import nn
    return _ProsodyNet(n_vocab, n_style, emb_dim, hidden)


class _ProsodyNet(torch.nn.Module):
    """字符 LSTM + 条件(style/bpm) + 逐音节回归头。"""

    def __init__(self, n_vocab, n_style, emb_dim, hidden):
        import torch
        from torch import nn
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb_dim, padding_idx=n_vocab - 1)
        self.rnn = nn.LSTM(emb_dim, hidden, num_layers=1, batch_first=True,
                           bidirectional=True)
        self.style_emb = nn.Embedding(n_style, 16)
        self.fc_cond = nn.Sequential(nn.Linear(16 + 1, 24), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 24, 48), nn.ReLU(), nn.Linear(48, 3))
        # params: f0_delta(半音), dur_scale(log), energy(logit->0.5~1.7)

    def forward(self, x, length, style, bpm_norm):
        import torch
        emb = self.emb(x)                                 # B,T,E
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, length.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        se = self.style_emb(style)                        # B,16
        cond = self.fc_cond(torch.cat([se, bpm_norm.unsqueeze(-1)], dim=-1))  # B,24
        cond = cond.unsqueeze(1)                          # B,1,24
        logits = self.head(torch.cat([out, cond.expand(-1, out.size(1), -1)], dim=-1))
        f0d = logits[..., 0]
        drl = logits[..., 1]
        energy = 0.5 + 1.2 / (1 + torch.exp(-logits[..., 2]))
        return f0d, drl, energy


class LearnedProsodyPredictor(ProsodyPredictorBase):
    """学习模型：加载权重后按协议 predict(lyrics,bpm,style)->ProsodyPlan。"""

    def __init__(self, weights_path: str, device: str = "cpu"):
        import torch
        self.device = device
        self.model = build_model()
        self.model.load_state_dict(
            torch.load(weights_path, map_location=device)["model"])
        self.model.to(device).eval()

    def is_learned(self) -> bool:
        return True

    def predict(self, lyrics: str, bpm: float, style: str,
                base_f0: float = 180.0) -> ProsodyPlan:
        import torch
        if style not in _STYLE_IDS:
            style = "快嘴"
        beat_sec = 60.0 / max(bpm, 20.0)
        spb = 0.5 if style != "旋律说唱" else 1.0
        plan = ProsodyPlan(bpm=bpm, style=style, beat_sec=beat_sec)
        t = 0.0
        lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            n = max(count_syllables(line), 1)
            ids = _char_ids(line)[:64]
            ids = ids + [_VOCAB["<pad>"]] * (max(n, 1) - len(ids))
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            length = torch.tensor([len(ids)], dtype=torch.long)
            st = torch.tensor([_STYLE_IDS[style]], dtype=torch.long,
                              device=self.device)
            bn = torch.tensor([bpm / 133.0], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                f0d, drl, energy = self.model(x, length, st, bn)
            f0d = f0d[0].tolist()[:n] + [f0d[0][-1].item()] * max(0, n - min(len(ids), n))
            drl = drl[0].tolist()[:n] + [drl[0][-1].item()] * max(0, n - min(len(ids), n))
            en = energy[0].tolist()[:n] + [energy[0][-1].item()] * max(0, n - min(len(ids), n))
            mean_f0 = base_f0 * float(np.mean([2 ** (d / 12) for d in f0d[:n]]))
            dur = n * spb * beat_sec * float(np.mean(drl[:n]))
            e = float(np.clip(np.mean(en[:n]), 0.5, 1.7))
            plan.lines.append(LinePlan(
                index=i, text=line, syllables=n,
                start_sec=t, duration_sec=dur, mean_f0=mean_f0,
                f0_style="learned", energy=e,
                jump=1 if style != "旋律说唱" else 2,
                syllable_f0_delta=[float(d) for d in f0d[:n]]))
            t = t + dur + (0.5 * beat_sec if style != "硬核" else 1.0 * beat_sec)
        return plan


# ---------------- 教师蒸馏数据 ----------------
def build_label_dataset(lyrics_lines, bpms, styles, base_f0=180.0):
    """由规则 teacher 生成逐音节伪标签。返回 (X, Y)，供训练烟测与管道冒烟。"""
    X, Y = [], []
    for line in lyrics_lines:
        n = max(count_syllables(line), 1)
        line_energy = {"快嘴": 1.15, "旋律说唱": 0.95, "硬核": 1.35}.get("快嘴", 1.0)
        for style in styles:
            le = {"快嘴": 1.15, "旋律说唱": 0.95, "硬核": 1.35}[style]
            for bpm in bpms:
                f0d = [_syll_f0_delta(style, j, n) for j in range(n)]
                drl = [_syll_dur_scale(style, j, n) for j in range(n)]
                en = [_syll_energy(style, j, n, le) for j in range(n)]
                X.append((line, style, bpm))
                Y.append((np.array(f0d, dtype="float32"),
                          np.array(drl, dtype="float32"),
                          np.array(en, dtype="float32")))
    return X, Y


def _collate_batch(batch, device):
    import torch
    xs, ys_f0, ys_dr, ys_en = [], [], [], []
    for (line, style, bpm), (f0, dr, en) in batch:
        ids = _char_ids(line)
        if not ids:
            ids = [_VOCAB["<unk>"]]
        xs.append((ids, style, bpm))
        ys_f0.append(f0); ys_dr.append(dr); ys_en.append(en)
    max_t = max(len(a[0]) for a in xs)
    xm = torch.zeros(len(xs), max_t, dtype=torch.long).fill_(_VOCAB["<pad>"])
    lens = []
    for i, (ids, style, bpm) in enumerate(xs):
        ids = ids[:max_t]
        xm[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        lens.append(len(ids))
    st = torch.tensor([_STYLE_IDS[a[1]] for a in xs], dtype=torch.long)
    bn = torch.tensor([a[2] / 133.0 for a in xs], dtype=torch.float32)
    mask = (xm != _VOCAB["<pad>"]).float()
    return dict(device=device, x=xm.to(device), length=torch.tensor(lens, dtype=torch.long),
                style=st.to(device), bpm=bn.to(device), mask=mask.to(device),
                n_syll=[a.shape[0] for a in ys_f0])


def train_predictor(lyrics_lines, bpms, styles, epochs=6, batch=32,
                    device="cpu", out_path=None):
    """训练小型 LSTM 逼近 teacher 标签。返回模型与 loss 序列。"""
    import torch
    from torch import nn
    X, Y = build_label_dataset(lyrics_lines, bpms, styles)
    idx = np.arange(len(X))
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    mse = nn.MSELoss()
    history = []
    order = np.random.RandomState(0).permutation(len(X))
    n_batch = max(1, len(order) // batch)
    for ep in range(epochs):
        sum_loss, cnt = 0.0, 0
        for k in range(0, len(order), batch):
            b = order[k:k + batch]
            samp = [(X[i], Y[i]) for i in b]
            inp = _collate_batch(samp, device)
            f0d, drl, en = model(inp["x"], inp["length"], inp["style"], inp["bpm"])
            # 逐样本长度截断到其目标音节数
            loss = 0.0
            for s, (_cond, (f0_arr, dr_arr, en_arr)) in enumerate(samp):
                n_s = inp["n_syll"][s]
                gt = torch.tensor(f0_arr[:n_s], dtype=torch.float32, device=device)
                loss = loss + mse(f0d[s, :n_s], gt)
                loss = loss + mse(torch.log(torch.clamp(drl[s, :n_s], 1e-3, 3)),
                                  torch.log(torch.tensor(dr_arr[:n_s], dtype=torch.float32, device=device)))
                loss = loss + mse(en[s, :n_s], torch.tensor(en_arr[:n_s], dtype=torch.float32, device=device))
            loss = loss / len(samp)
            opt.zero_grad(); loss.backward(); opt.step()
            sum_loss += loss.item(); cnt += 1
        avg = sum_loss / max(cnt, 1)
        history.append(avg)
        if (ep + 1) % 2 == 0 or epochs <= 3:
            print(f"  epoch {ep+1}/{epochs}  loss={avg:.4f}")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({"model": model.state_dict()}, out_path)
    return model, history