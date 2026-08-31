# Qwen3-TTS 可插拔角色外挂系统 —— 决策与工程规范 Spec

## Why
现有的角色挂载/情感注入工程聚焦于**文本 LLM 推理期注入**（persona 提示、logits/KV/状态向量注入）。本 spec 面向一个**不同的、互补的能力**：通过**有监督微调（LoRA）**，把某一直播角色（缘圆）的**音色、口语风格、情感韵律**封装为独立、可热插拔、不污染 `Qwen3-TTS-12Hz-1.7B-Base` 原始权重的「角色外挂包」，实现情感标签可控的语音合成。

本次任务要求在启动阶段输出一份**决策报告**（数据策略 / 技术方案 / 训练配置 / 交付物清单），并将其固化为可落地的工程规范，同时把**低显存快速优化（生成加速、全显卡兼容：N 系 20/30/40/50 + AMD）**作为显式工程约束写入需求。

## What Changes
- 建立 `.trae/specs/qwen3tts-character-addon/` 下的三项规范文档（本 spec、tasks.md、checklist.md）。
- 在项目内新增「决策报告」文档，覆盖四大部分（数据策略、技术方案、训练配置、交付物清单）。
- 确认数据源与格式：优先使用**纠正后 6951 合格样本**（`情感打标训练集_20260817_043155_合格6951_总7030.jsonl`）而非未清洗全量 7030；明确原始版 = 含 79 条不合格样本的初始输出，纠正版 = 经质量复核剔除/修正后的版本。
- 设计 LoRA 微调 + 情感条件控制（reference token + 情感标签 prompt 前缀）的可插拔方案，LoRA adapter 独立导出为外挂包，不改 Base 权重。
- 明确 INT4 推理 + 全显卡兼容运行时策略（CUDA / ROCm / Vulkan / CPU Q4_K_M 回退）。
- 引入 Langfuse 作为训练/推理可观测性上报的监控层（可选启用）。

## Impact
- Affected specs: 角色挂载与情感注入综合方案（本 spec 为**新增能力维度**，不推翻推理期注入方案，二者互补：文本人格 → LLM 注入；语音人格 → TTS LoRA）。
- Affected code: 无运行时代码改动（本 spec 为方案/规范阶段）。
- Affected data: `缘圆_角色挂载与情感注入工程/02_角色参数与数据/微调数据/`（将被清洗、划分、格式化为 Qwen3-TTS 训练集）。
- 依赖项：基础模型 `Qwen3-TTS-12Hz-1.7B-Base`（≈1.7B，int4 后 ≈1GB 权重）+ 训练机显存 24GB（本机 RTX 3080 16GB 亦可）。

---

## ADDED Requirements

### Requirement: 数据策略
系统 SHALL 基于纠正后的 6951 合格打标样本构建 Qwen3-TTS 训练数据。

#### Scenario: 数据源选择
- **WHEN** 决策报告进入「数据策略」章节
- **THEN** 明确选用纠正版 6951（排除 79 条不合格，如情感方向与标签冲突、字段缺失、置信度 <0.6），并以 `质量评估报告_latest.json` 作为筛选依据，而非原始未清洗全量 7030。

#### Scenario: 格式转换与划分
- **WHEN** 准备模型可消费的训练数据
- **THEN** 将 transcript 转为 `input_text`、音频片段（`F:\打标\数据层\分割片段\*.mp4`）映射为 `input_audio` 元数据，情感标签（`情感标签`/`discrete_emotion_primary`）映射为情感条件；按 90/10 划分 train/val（同角色不跨集泄漏），导出 JSONL/CSV 清单。

### Requirement: 微调方案选型
系统 SHALL 采用 **LoRA（PEFT）**作为角色外挂的微调载体，权重独立可插拔，不修改 Base。

#### Scenario: 外挂技术选型
- **WHEN** 决策报告进入「技术方案」
- **THEN** 技术选型定为 LoRA（低显存、adapter 独立、易热插拔），对比说明 Adapter、全参微调的取舍；框架在「官方微调脚本 / Qwen3-TTS-EasyFinetuning / 自建 transformers+PEFT」三选一，给出理由。

#### Scenario: 情感条件控制
- **WHEN** 设计情感可控合成
- **THEN** 采用「reference token（音色）+ 情感标签 prompt 前缀（<emotion>开心/俏皮/悲伤</emotion>）+ 可选情感 ref 音频」的条件化方案，让模型在 LoRA 空间学会把情感标签映射到韵律（f0/语速/能量），而非仅拷音色。

### Requirement: 训练与优化配置
系统 SHALL 在 24GB（本机 16GB 亦可）显存下以合理配置完成 LoRA 训练，并给出收敛监控与验证方案。

#### Scenario: 关键超参
- **WHEN** 决策报告进入「训练配置」
- **THEN** 给出 LoRA rank/alpha (如 r=16, α=32)、量化方式（训练期 bnb 4bit 基座 / 需保持 Base 可插拔），epochs/lr/batch 等参数，并说明每项依据。

#### Scenario: 收敛与验证
- **WHEN** 训练过程中
- **THEN** 定义监控指标（train/val loss ≥2 连续轮不降则早停；主观指标 MOS、情绪区分度、音色余弦相似度），并规划回译评估：合成音频 → ASR transcript 对比原文的保真误差 与 情感转写一致性。

### Requirement: 部署与交付
系统 SHALL 将 LoRA adapter 封装为独立「角色外挂包」，支持动态加载与情感控制。

#### Scenario: 外挂包封装
- **WHEN** 训练完成导出
- **THEN** 交付 `adapter_config.json + adapter_model.safetensors + 情感标签词表 + 角色元数据 + README`，与 Base 权重分离；推理端「Base + 动态 attach adapter」加载，支持情感标签实时切换与热插拔角色切换。

### Requirement: 低显存快速优化与全显卡兼容（NVIDIA / AMD）
系统 SHALL 在 INT4 基础上最大化推理加速（目标生成速度翻倍方向），并兼容主流显卡（N 系 20/30/40/50 与 AMD）。

#### Scenario: 推理加速
- **WHEN** 调用合成推理
- **THEN** 启用 int4 量化 +（NVIDIA）flash-attn / torch.compile；为加速预留 speculative / 批量 / 缓存复用路径，量化目标在**识别出的加速瓶颈上可测量**（如首包延迟、RTF 实时率），并优先保证**低显存显卡（≤8GB）可跑**。

#### Scenario: 全显卡兼容
- **WHEN** 在不同硬件上运行
- **THEN** 提供运行时抽象：NVIDIA → CUDA(llama.cpp-cuda / torch+cuda)；AMD → Vulkan / ROCm；无 GPU → CPU Q4_K_M 回退；避免强绑定单一后端，保证同一角色包跨硬件可用。

### Requirement: 可观测性（Langfuse）
系统 SHALL 提供可选的数据埋点，将训练过程与合成请求 trace 到 Langfuse，便于监控与回放（推荐启用，不阻塞主流程）。

---

## MODIFIED Requirements
无（本能力为新增，不修改既有推理期注入方案的验收口径）。

## REMOVED Requirements
无。