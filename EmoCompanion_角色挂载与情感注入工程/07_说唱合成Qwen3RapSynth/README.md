# Qwen3-RapSynth

基于 **Qwen3-TTS** 与外挂韵律路由的说唱合成研究框架。

> 工具栏提示：Qwen3-TTS 开源 Base **并不**暴露 pitch/duration/energy 等声学参数（已用源码+真实合成双重验证）。本仓库采用**间接控制面**：音色走 x-vector，节奏/音高/能量走采样级后处理 + 拍网格编排。详见 [docs/00_技术报告.md](docs/00_技术报告.md) 与 [docs/01_控制面探查报告.md](docs/01_控制面探查报告.md)。

## 快速开始
```bash
# 全链路：歌词 + 风格 + BPM → 说唱干声（需 GPU，首次加载 Base+LoRA）
python scripts/generate_rap.py --style 快嘴 --bpm 96

# 框架校验（不触发 GPU）
python scripts/generate_rap.py --no-tts --style 旋律说唱 --bpm 84

# 控制面探查
python scripts/probe_control.py --out probe_result.json

# 韵律数据准备（wav + 同名 txt）
python scripts/prepare_prosody_data.py --src data/raw --out data/split
```

## 风格
`快嘴` / `旋律说唱` / `硬核`（三档模板，内置示例歌词）。样例已生成到 `output/samples/`。

# 韵律预测器（规则基线 + 学习模型）
`generate_rap.py` 通过 `prosody_model.rules.get_predictor()` 自动选型：存在学习模型权重则走 LSTM，否则退回规则基线（已训练产物`prosody_model/weights/prosody_lstm.pt`，`f0_style="learned"`）。

```bash
python scripts/train_prosody_predictor.py --epochs 8 --device cpu
```

> 当前学习模型为**教师蒸馏烟测**（规则 teacher 伪标签）。接真实 MFA 对齐 F0/时长标签后，替换 `learned.build_label_dataset()` 重训即为正式版。

## 报告
- `docs/00_技术报告.md` — 总纲（方法/实验/结果/工程）
- `docs/01_控制面探查报告.md` — Phase 0 假设检验与间接控制面
- `docs/03_韵律预测器.md` — 规则基线 + 学习模型统一接口
- `docs/04_评估.md` — 客观指标与对比协议

## 依赖
numpy, librosa, soundfile, torch, peft, qwen_tts（清华镜像安装）。

> 科研框架阶段，遵守「框架默认不训练」：学习模型在合规数据就绪后按 `ProsodyPredictorBase` 接口接入。