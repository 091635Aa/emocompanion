# Tasks

> 科研框架阶段完成。标注「✅ 已交付」表示本会话实装并通过运行验证；标注「🔒 待数据/电源」表示框架/接口已就绪、需外部合规数据或额外 GPU 富余后运行，未在本会话伪造完成。

- [x] Task 0: 初始化研究仓库与探针环境
  - [x] 0.1 创建 `07_说唱合成Qwen3RapSynth/` 目录结构（prosody_model / integration / tts / eval / scripts / docs / output）
  - [x] 0.2 复用推理环境与已训练外挂包路径，编写 `scripts/probe_control.py`（已运行，`output/samples/probe_result.json`）
- [x] Task 1: 控制面可行性探查（扎破假设）
  - [x] 1.1 源码审读 + 签名枚举 + 真实合成探针：确认公开 Base **不**暴露 pitch_curve/duration/energy
  - [x] 1.2 切换间接控制面（x-vector 音色 + time-stretch/pitch 后处理 + 拍网格编排）
  - [x] 1.3 成稿 `docs/01_控制面探查报告.md`
  - [x] 汇报点 1：结论已纳入 `docs/00_技术报告.md`
- [ ] Task 2: 数据准备与韵律标注
  - [x] 2.1 工具链 `scripts/prepare_prosody_data.py` 实装（wav+txt → CSV/JSONL：F0/时长/能量，字符级占位对齐，UTF-8 无 BOM）
  - [x] 2.2 优先 MFA 对齐已在文档标注；未装 MFA 时走内置字符级软对齐回退
  - [x] 2.3 `docs/02_数据与标注.md` 骨架已并入 `docs/03`（数据来源与数量待合规数据集就位后补全）
  - [x] 汇报点 2：数据源需用户提供开源/合规说唱干声+歌词（🔒 待数据）
- [ ] Task 3: 外挂路由模型升级为韵律参数预测器
  - [x] 3.1 规则先验基线实装（`prosody_model/rules.py`：快嘴平直 / 旋律拱形 / 硬核下探；已随端到端跑通）
  - [x] 3.2 学习模型统一接口 + LSTM 实装（`prosody_model/learned.py`），batch=1+grad_accum 约束写入
  - [x] 3.3 小型 LSTM 教师蒸馏烟测跑通：`scripts/train_prosody_predictor.py`（CPU，loss 10.0→2.06，权重 prosody_lstm.pt，generate_rap 自动选型 f0_style="learned"）——真实数据/MFA 标签后重训为正式版
  - [x] 3.4 逐音节 F0 轮廓注入实装并自检通过（`injector.apply_f0_contour` + learned `syllable_f0_delta`；`test_f0_contour.py` 实测四窗 F0≈目标，跨度 61.7Hz）——「音素≈音节级」控制接口可用
  - [x] 汇报点 3：规则基线客观数字见 `docs/04_评估.md`
- [ ] Task 4: 与 Qwen3-TTS 集成
  - [x] 4.1 间接注入层实装（`integration/injector.py` 对拍 time-stretch/pitch/energy + 编排），独立可关、不影响既有外挂
  - [x] 4.2 控制粒度消融实跑（`bench_ablation.py`）：音节级 vs 行级 vs 无注入已出实测数字（docs/04）——音节级 onset 误差优于基线 ~10%
  - [x] 4.3 `scripts/generate_rap.py` 打通 `歌词+BPM → 干声 wav`（✅ 已产出三风格真实干声）
  - [x] 汇报点 4：注入方式详见 `docs/01` 与 `docs/00`
- [ ] Task 5: 生成质量评估
  - [x] 5.1 `eval/metrics.py` 实装（对拍/onset/实测 beat 误差 / BPM 漂移 / F0 RMSE / 押韵命中 / 能量对比），三风格已出数字
  - [x] 5.2 基线对比已实测（`bench_ablation.py`，快嘴全表在 docs/04）：音节级优于无注入基线 ~10%；行级≈基线（诚实：间接后处理的真实增益待音素级学习注入）
  - [x] 5.3 成稿 `docs/04_评估.md`（含当前短板与增益方向）
  - [x] 汇报点 5：基线数字见 `docs/04`
- [ ] Task 6: 工程化与交付
  - [x] 6.1 CLI `scripts/generate_rap.py` 一键生成（歌词+风格+BPM）（WebUI 可选，CLI 满足交付口径）
  - [x] 6.2 ✅ 输出 ≥3 段不同风格干声：`快嘴_96bpm / 旋律说唱_84bpm / 硬核_72bpm .wav`
  - [x] 6.3 成稿 `docs/00_技术报告.md` + `README.md`

# Task Dependencies
- Task 1 依赖 Task 0 ✓（探针）
- Task 2 与 Task 3.1 并行 ✓（数据工具与规则基线独立）
- Task 3.2/3.3 依赖 Task 2（训练目标来自标注数据）→ 学习训练🔒 待数据
- Task 4 依赖 Task 1 与 Task 3（控制面 + 预测器）✓
- Task 5 依赖 Task 4（端到端干声）✓；基线对比项🔒
- Task 6 依赖 Task 4 与 Task 5 ✓