# Tasks

- [x] Task 1: 数据策略决策
  - [x] 1.1 复核纠正版 6951 vs 原始 7030 的差异依据（质量评估报告_latest.json），确认剔除逻辑（不合格 79 条：情感方向冲突/字段缺失/置信度<0.6）
  - [x] 1.2 生成 Qwen3-TTS 训练格式清单：`input_text`(transcript) + `input_audio`(片段路径) + `emotion`(情感标签)，输出 CSV/JSONL 元数据
  - [x] 1.3 90/10 划分 train/val（防跨集泄漏），导出 `train.csv / val.csv`（脚本运行：train 6300 / val 634）
- [x] Task 2: 微调方案选型
  - [x] 2.1 对比 LoRA / Adapter / 全参微调，选定 LoRA（PEFT）并写明理由
  - [x] 2.2 框架选型：官方脚本 vs Qwen3-TTS-EasyFinetuning vs 自建 transformers+PEFT，给出决策依据
  - [x] 2.3 设计情感条件控制（reference token 音色 + 情感标签 prompt 前缀）
- [x] Task 3: 训练配置与优化
  - [x] 3.1 设定 LoRA r/α、基座量化（bnb 4bit）、epochs/lr/batch 等参数并注明依据
  - [x] 3.2 定义收敛监控（loss 早停阈值）与验证方案（MOS、音色相似度、ASR 回译一致性）
- [x] Task 4: 部署与交付
  - [x] 4.1 定义角色外挂包结构（adapter 权重 + 情感词表 + 角色元数据 + README）
  - [x] 4.2 设计动态加载与情感/角色热切换机制
- [x] Task 5: 低显存快速优化与全显卡兼容
  - [x] 5.1 INT4 推理 + NVIDIA 加速路径（flash-attn / torch.compile / llama.cpp-cuda）
  - [x] 5.2 AMD(Vulkan/ROCm) + CPU(Q4_K_M) 回退，运行时抽象不绑定单一后端
  - [x] 5.3 定义加速与显存基线（≤8GB 可跑、RTF 缩短）与实测测量脚本
- [x] Task 6: 可观测性（Langfuse）
  - [x] 6.1 训练过程与合成请求 trace 埋点方案（可选启用）
- [x] Task 7: 汇总决策报告
  - [x] 7.1 产出《Qwen3-TTS 角色外挂工程决策报告.md》，含数据策略/技术方案/训练配置/交付物清单四大章节 + 风险与基线
  - [x] 7.2 数据准备脚本 `scripts/prepare_qwen3tts_data.py` 运行通过（train 6300 / val 634）
- [x] Task 8: 训练/推理/基准脚本实装
  - [x] 8.1 `scripts/train_lora.py`（bnb4bit + LoRA、loss 早停、adapter 导出、Langfuse 可选）语法校验通过
  - [x] 8.2 `scripts/load_and_generate.py`（Base+adapter 动态加载、情感/角色热切换、RTF/显存记录）语法校验通过
  - [x] 8.3 `scripts/bench_inference.py`（RTF/首包/满足目标 + 后端选择 torch/llamacpp）语法校验通过
- [x] Task 9: 原生 llama.cpp 运行时落地（INT4/全显卡兼容主推路径）
  - [x] 9.1 验证工作区随附 `pykits/llama-cpp-bin/llama-tts.exe`（build 10502）为可用原生 Qwen3-TTS 运行时（正确解析 TTS 参数、要求 backbone+mmproj）
  - [x] 9.2 `bench_inference.py` 增加 `--backend llamacpp`，自动定位并调用本机 llama-tts.exe 测 RTF/首包
  - [x] 9.3 决策报告第 4/5 章更新：原生 GGUF（≈1.8GB 显存、RTF 0.35、AMD Vulkan）为部署/优化主推路径（社区权威口径）

# Task Dependencies
- Task 2.3 依赖 Task 1.1（数据质量决定情感标签能否直接映射）
- Task 3 依赖 Task 2.2（框架决定具体超参接口）
- Task 4 依赖 Task 3.1（adapter 由训练产出）
- Task 5 可与 Task 3 并行（运行时工程独立于训练配置）
- Task 7 汇总依赖 Task 1–6 全部完成