# EmoCompanion · AI 情感引擎（emocompanion）

> 开源公开版 ｜ 整理日期 2026-08-31 ｜ 本仓库为研究工作区的**源码 + 技术文档**整理快照（不含模型权重、训练数据、专利申报原件与商业材料）
>
> **去标识化说明**：项目曾用名「缘圆」已统一替换为 **EmoCompanion**（角色昵称「圆圆」→「小伴」，代码标识 `yuanchat`→`emochat`），指代同一套工程，无功能差异。

EmoCompanion是一套本地化的「AI 角色情感引擎」，覆盖两条主线：

1. **文本侧 · 角色挂载与情感注入**：基于 llama.cpp CUDA + Qwen3-4B GGUF 的六层情感注入管线（骨架/去AI腔/P3锚点/P4共振/P5超融合/兜底监控），角色包 ≤10MB 热切换。
2. **语音侧 · Qwen3-TTS 外挂**：llama-tts.exe (GGUF) 单一部署路径，voice + emotion 双 LoRA 驱动音色与情感，StylePlug 八种说话风格外挂，整段一次合成实现近实时（RTF<1）。

---

## 仓库结构

| 目录 | 内容 |
|------|------|
| `docs/` | **整理文档**：方案逻辑总览、调用测试方法、测试结果汇总（先读这里） |
| `EmoCompanion_角色挂载与情感注入工程/` | **核心工程**：文本引擎后端、05 三档显存测试实验线、06 Qwen3TTS 外挂（tts_gguf / unified_server / integrated_app / finetuning / calibration）、最终归档与接口文档 |
| `最后版本/` | 一体化全流程应用（启动.py + 核心引擎 + 前端 + 测试） |
| `EmoCompanion—Ai智能体/` | 桌面 AI 智能体（核心模块：后端服务/多模态识别/实时通话/语音合成/音色管理 + TDS 版本归档） |
| `P1_5/` | P1.5 语义回响：源码 + 实验报告（E1-E13） |
| `锚点回响_AnchorEcho/` | P3 锚点回响解码器 + 评测体系（HeartBench/EmoCharacter/LLM-Judge） |
| `KV_情感共振解码/` | P4 KV 共振 / P5 超融合 / P6 情感导演解码器 + 评测结果 |
| `情感潮汐解码_ETD/` | Emotion Tidal Decoding：潮汐感知/决策/解码 + 四模式对照 |
| `lora外挂/` | LoRA 训练脚本、评估脚本、P6（权重文件不入库） |
| `最终工程架构/` | 全流程推理框架 + OpenAI 兼容 API 服务 |
| `最终总结报告/` | 项目全流程总结、市场调研、第四代锚点回响专项 |
| `打标_RPG/` | RPG 精灵素材打标系统 |
| `学习伴侣/` | 学习伴侣小应用（FastAPI） |
| `输出文档/` | 专利族（7 件）、学术论文（7 篇）、实验室设计、市场调研、技术组合方案、专利核查（均为 md 源稿） |
| `ppt/` | EmoCompanion AI 情感引擎演示页（单文件 HTML） |
| `tools/` | 工具脚本（LoRA→GGUF 转换、PDF 版式校验、提案工具） |
| `_specs/` | 各子项目开发 spec / checklist / tasks 归档 |

## 快速开始

```bash
# ① 文本引擎（三层情感注入 + FastAPI）
cd EmoCompanion_角色挂载与情感注入工程/04_源码与原型/backend
python app.py            # 自动创建/激活 .venv、下载依赖、启动 8000 端口

# ② 文本+语音一体化服务
cd EmoCompanion_角色挂载与情感注入工程/06_Qwen3TTS外挂/serve
python integrated_app.py

# ③ CLI 单轮对话
python cli.py chat "你好呀"
```

**依赖**：Python 3.10、CUDA GPU（实测 RTX 3080 16GB）、llama.cpp 二进制（llama-tts.exe / llama-server.exe）、GGUF 模型（Qwen3-4B-Q4_K_M、Qwen3-TTS base、voice/emotion LoRA GGUF）。模型与二进制**不在本仓库**，放置路径约定见 [docs/02_调用测试方法.md](docs/02_调用测试方法.md)。

## 核心文档索引

- [docs/01_方案逻辑总览.md](docs/01_方案逻辑总览.md) —— 六层注入架构、解码器族谱、TTS 外挂设计、角色包机制
- [docs/02_调用测试方法.md](docs/02_调用测试方法.md) —— 三个服务的启动方式、全部 API 端点、curl/Python 调用示例、测试脚本索引
- [docs/03_测试结果汇总.md](docs/03_测试结果汇总.md) —— 验收指标表、各里程碑实验结论、风格轴实测数据
- `EmoCompanion_角色挂载与情感注入工程/07_最终归档/最终归档.md` —— 里程碑归档（M0-M4）
- `EmoCompanion_角色挂载与情感注入工程/07_最终归档/接口文档.md` —— 文本引擎 v1.0 API

## 未入库内容（留在本地 d:\AI情感）

- 模型权重：GGUF / safetensors / adapter（`模型空间`、`模型（j盘）`、`pykits/models`、各 `out/` 目录）
- 训练数据：音频数据集、`*.jsonl` 训练集（`打标`、`微调文本`、02 微调数据）
- 二进制运行时：`pykits/llama-cpp-bin`、各类 .venv / node_modules
- 商业材料：`091635Aa_商业化推进`（提案/情报，仅收录其下通用工具脚本到 `tools/proposal_tools/`）、各厂家专项报告、商务邮件草稿
- PDF 产物：`输出文档/PDF转化`（可用 `_tools/batch_md2pdf.py` 重新生成）

## 声明

本项目为研究性质的开源快照，仅供技术交流。模型权重、训练数据、专利申报原件与商业材料不在本仓库范围内。
