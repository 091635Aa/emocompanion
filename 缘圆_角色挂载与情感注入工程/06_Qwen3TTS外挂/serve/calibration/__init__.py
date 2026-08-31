# -*- coding: utf-8 -*-
"""calibration 前向关包。

用法:
  python -m calibration.metrics        # 自检(probe 一段 wav 的韵律)
"""
from calibration.metrics import (  # noqa: F401
    prosody_features, composite, edit_similarity, speaker_emb, speak_rate,
)