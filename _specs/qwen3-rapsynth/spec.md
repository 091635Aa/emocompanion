# Qwen3-RapSynth —— 基于 Qwen3-TTS 与外挂韵律路由的说唱合成框架 Spec

## Why
已有的 Qwen3-TTS 可插拔角色外挂（`voice_lora` / `emotion_lora` + `target_speaker_embedding.pt`）解决了**音色克隆与情感风格**，但韵律（节奏、押韵、重音、F0 起伏）仍由 Base 默认生成，无法按伴奏 BPM 与歌词节奏对齐。本任务目标是把「外挂路由模型」从情感/音色控制升级为**韵律参数预测器**，构建「文本歌词 + 伴奏节拍 → 韵律参数 → 原生合成」的端到端说唱合成系统。

> 科研定位：**引导而非强制**。任务书中的技术路线（MFA 对齐、Transformer/LSTM/扩散韵律模型、音素级 vs 音节级控制）为参考而非硬性约束；每个阶段自查并汇报，允许调整方向。**核心前提 Qwen3-TTS「原生支持 pitch_curve/duration」未经核实**，须在第一阶段用实测扎破/修正，再决定控制面设计。

## What Changes
- 在 `缘圆_角色挂载与情感注入工程/07_说唱合成Qwen3RapSynth/` 新建研究仓库（数据 / 韵律模型 / 集成 / 评估 / 演示）。
- 新增一个可运行的 `RapSynth` 管道：`lyrics+BPM → 韵律参数(F0/时长/能量) → 注入 Qwen3-TTS → 干声 wav`。
- 复用既有工程资产：Base + 双路 adapter + `target_speaker_embedding.pt`、`serve/` 推理模板；**不污染 Base 权重、不破坏已交付外挂包**。
- 产出：技术文档、可运行代码仓库、≥3 段不同风格（快嘴 / 旋律 / 硬核）合成样例。

## Impact
- Affected specs: `qwen3tts-character-addon`（复用其外挂包与推理路径，不修改）；`yuanyuan-tts-webapp`（可选挂载合成/评估入口）。
- Affected code: 新增 `07_说唱合成Qwen3RapSynth/`；对 `06_Qwen3TTS外挂/out/voice_lora|emotion_lora` 只读复用。
- 软硬件约束：RTX 3080 16GB、GPU 负载 ≤5%，训练默认 batch=1（防 OOM）；模型加载 ≤3.9s；笔记本电源有限，**框架/探查阶段不触发大规模训练**。流水线统一 CUDA + 清华镜像 + gh-proxy 下载。

---

## ADDED Requirements

### Requirement: 控制面可行性探查（Phase 0，先扎破假设）
系统 SHALL 以实测方式确认 Qwen3-TTS 真正暴露的底层声学控制原语，再定控制面设计。

#### Scenario: Pitch/Duration 假设检验
- **WHEN** 进入本任务第一阶段
- **THEN** 用最小探针实验验证 Base / 已训练外挂是否接受 `pitch_curve` / `duration` / 能量等显式注入；
  - 若**不支持显式注入**（更可能的现实）：改为**间接控制面**——韵律注入通过「特殊 prompt / 条件前缀 / 参考音频 / 说话速率调度 / 逐句切分与拼接」实现，并在 spec 文档补记实测结论与取舍。
- **THEN** 产出 `docs/01_控制面探查报告.md`，记录 API 签名、支持粒度、瓶颈与最终采用的控制面。

### Requirement: 数据准备与韵律标注（Phase 1）
系统 SHALL 构建「说唱人声 + 对齐 + 韵律标签」数据集，作为韵律预测器的训练目标。

#### Scenario: 数据与标注
- **WHEN** 需要训练韵律预测器
- **THEN** 采集合规/开源说唱人声干声 + 歌词 + BPM；对选定子集做强制对齐（优先 MFA）得到**音素级**时间戳；在此基础上提取 F0、音素时长、能量、重音标签；输出 CSV/JSONL（与 `06` 训练格式风格一致，UTF-8 无 BOM）。
- **THEN** 记录数据来源与授权（科研伦理：仅用开源/合规数据）。

### Requirement: 外挂路由模型升级为韵律参数预测器（Phase 2）
系统 SHALL 将现有路由模型改造为：输入 `text + BPM/句拍`，输出 `每音素的 F0 序列 / 时长序列 / 能量系数`。

#### Scenario: 模型与训练
- **WHEN** 训练韵律预测器
- **THEN** 选型（Transformer / LSTM / 扩散等）论证后实施；16GB 下 `batch_size=1 + grad_accum`，尺度小、收敛快；仅在校验后按需触发训练（框架默认不训练）。
- **THEN** 预留「纯规则先验」基线（按行尾韵/重音/等时拍生成韵律）作为对照，证明学习模型的有效增益。

### Requirement: 与 Qwen3-TTS 集成（Phase 3）
系统 SHALL 把韵律参数注入 Qwen3-TTS，替换/引导默认韵律，并做控制粒度消融。

#### Scenario: 注入与消融
- **WHEN** 合成歌词
- **THEN** 逐句/逐块将预测的强拍与时长映射到采样级或段级控制；消融音素级 vs 音节级 vs 无注入三种粒度并记录设定。
- **THEN** 保持 Base + adapter 方式不变，注入层独立可关（开关不改变既有外挂能力）。

### Requirement: 生成质量评估（Phase 4）
系统 SHALL 用主客观指标对比本方案与基线（直接用 Qwen3-TTS + 情感标签）。

#### Scenario: 评测
- **WHEN** 输出干声
- **THEN** 记录 主观 MOS、音高准确率、节奏对齐误差（onset 偏差 / BPM 漂移）、重音/押韵命中率；给出与基线对比表与显著性结论。

### Requirement: 工程化与演示（Phase 5）
系统 SHALL 提供 CLI / WebUI「歌词+风格+BPM → 干声」一键生成，并交付样例与报告。

#### Scenario: 交付
- **WHEN** 系统可用
- **THEN** 提供 `generate_rap.py`（CLI）+ 可选 Web 页；输出 ≥3 段不同风格干声（快嘴 / 旋律说唱 / 硬核）落盘；成稿 `docs/00_技术报告.md`（方法 / 实验 / 结果分析）。

## MODIFIED Requirements
无（本能力为新增）。

## REMOVED Requirements
无。