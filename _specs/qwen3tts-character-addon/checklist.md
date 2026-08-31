# Checklist

- [x] 决策报告覆盖数据策略/技术方案/训练配置/交付物清单四大章节
- [x] 数据策略明确选用纠正版 6951 而非原始全量 7030，并给出剔除依据
- [x] 训练格式清单与 train/val 划分方案已产出（脚本运行：train 6300 / val 634）
- [x] 微调方案确定为 LoRA（PEFT），adapter 独立于 Base 权重，可热插拔
- [x] 框架选型给出明确结论（官方 qwen-tts + 自建 PEFT，EasyFinetuning 兜底）与对比依据
- [x] 情感条件控制方案（reference token + 情感标签前缀 + 可选情感 ref）已设计
- [x] 训练关键超参（r=16-32/α、bnb4bit、epochs/lr 2e-6/batch）已给出并注明依据
- [x] 收敛监控（val loss 早停、目标 loss 4.1-4.2）与验证方案（MOS/音色相似度/ASR 回译）已定义
- [x] 角色外挂包封装结构与动态加载/情感热切换机制已设计
- [x] 低显存快速优化需求：INT4 + NVIDIA 加速路径（flash-attn/torch.compile/部分并行）已定义
- [x] 全显卡兼容：AMD(Vulkan/ROCm/DirectML) + CPU(Q4_K_M) 回退已定义，未绑定单一后端
- [x] 加速与显存基线已定义（RTF≤0.6、≤8GB 可跑）并给出测量脚本方案
- [x] Langfuse 可观测性埋点方案已纳入（可选启用）
- [x] 存在清晰的基线（裸 Base 情感可控性/音质参考）用于验收外挂包增益
- [x] `train_lora.py` 已实装并通过语法校验（bnb4bit + LoRA + 早停 + adapter 导出 + Langfuse 可选）
- [x] `load_and_generate.py` 已实装并通过语法校验（动态 attach/detach + 情感/角色热切换）
- [x] `bench_inference.py` 已实装并通过语法校验（RTF/首包/满足目标 + 后端选择 torch/llamacpp）
- [x] 原生 llama.cpp 运行时（llama-tts.exe）已验证可用并被纳入部署主推路径（GGUF≈1.8GB 显存、RTF 0.35、AMD Vulkan 社区口径）