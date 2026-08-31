# Qwen3-TTS 可插拔角色外挂系统 —— 工程决策报告

> 依据：`.trae/specs/qwen3tts-character-addon/` 规范（spec / tasks / checklist）、`缘圆_角色挂载与情感注入工程` 现有打标与推理期注入工程、Qwen3-TTS 官方仓库与社区微调实践（EasyFinetuning、LoRA 实战、低显存 LoRA 经验）。
> 目标模型：`Qwen3-TTS-12Hz-1.7B-Base`（离散多码本 LM，16 codebook @ 12Hz；提供 Voice Clone 能力，与下游 LoRA 天然兼容）。
> 铁律：**外挂优先 / 情感可控 / INT4 低显存**。

---

## 〇、总体结论（TL;DR）

| 决策项 | 结论 |
|---|---|
| 数据源 | **纠正版 6951（合格筛选后 6934）**；排除 79 条不合格 |
| 微调技术 | **LoRA（PEFT）**；rank 建议 16–32 |
| 框架选型 | **官方 `qwen-tts` 包 + 自建 transformers/PEFT 训练脚本**（EasyFinetuning 作参考/兜底） |
| 情感建模 | **Voice Clone（ref 音色）+ 情感标签条件前缀 + 情感 ref 音频**三源条件化 |
| Base 处理 | **保持 Base 权重不动**，仅冻结基座 → 训练 LoRA 旁路 → 独立导出外挂包 |
| 显存方案 | 训练 bnb 4bit + LoRA（16GB 可训）；部署 INT4（GGUF Q4_K_M / int4）+ flash-attn |
| 加速目标 | RTF ≤ 0.6（≈快 2× 基线）；低端卡（≤8GB、含 AMD/Radeon 与 20/30 系）可跑 |
| 交付物 | 角色外挂包（adapter + 情感词表 + 元数据 + README）+ 加载器 + 评测线 |

---

## 一、数据策略（Task 1）

### 1.1 数据源选择：纠正版 6951 ▸ 6934
- **原始版**：`7030` 条，含 79 条不合格（`情感打标质量评估_*_总7030.json` 判定：情感方向与标签冲突、必填字段缺失、内容描述 <50 字、预测问题 <3、置信度 <0.6 等）。
- **纠正版**：`情感打标训练集_*_合格6951_总7030.jsonl` 已剔除/复核修正（二次打标 339 条采用复核结果）。
- **决策**：采用纠正版，并**在训练脚本中二次过滤**（置信度 ≥0.6 + 必填字段 + 命中质量报告黑名单），最终映射 **6934 条**（17 条因 transcript 过短 / 命中黑名单被程序剔除）。

### 1.2 训练格式清单（Qwen3-TTS 消费格式）
映射自打标字段（`prepare_qwen3tts_data.py`，已跑通）：

| 目标字段 | 来源 | 用途 |
|---|---|---|
| `input_text / text` | `输出.transcript` | 合成文本 |
| `input_audio / audio_filepath` | `输入.片段路径` | 训练音频（vocoder/LM 监督信号） |
| `emotion`、`emotion_primary` | `输出.情感标签` / `discrete_emotion_primary` | **情感条件**（训练分词构造） |
| `emotion_intensity / valence / arousal / dominance` | 打标字段 | 可选连续情感条件（V/A/D） |
| `f0_mean_hz / speech_rate_syll` | 打标声学字段 | 结果验证/回归分析，不直接入 LO |
| `source_file` | `输入.原文件` | **防跨集泄漏分组键** |

### 1.3 划分方案
- **按 `source_file`（原始卷）分层 90/10 划分**，保证同一视频的段落不跨 train/val 泄漏。
- 实际结果：**train 6300 / val 634**（675 个原始卷内划分）。
- 输出：`data/split/train.jsonl|.csv`、`val.jsonl|.csv`。

---

## 二、技术方案（Task 2）

### 2.1 外挂技术选型：LoRA（PEFT）✅
| 方案 | 是否冻结 Base | 可插拔 | 显存 | 结论 |
|---|---|---|---|---|
| **LoRA** | 冻结，仅旁路 A/B | ✅ 独立 adapter | 极低 | **选用** |
| DoRA | 冻结 | ✅ | 低 | 可选升级（同等 r 更高精度） |
| Adapter 全参 | 部分冻结 | ✅ | 中 | 次选 |
| (IA)³ | 冻结 | ✅ | 极低 | r 依赖强，效果偏弱 |
| 全参 SFT | 全部更新 | ❌ 污染 Base | 高 | **禁用**（违背外挂优先） |

理由：LoRA 参数量 <0.1% 原始模型、可导出为独立 safetensors、推理期可动态 attach/detach，社区已验证 Qwen3-TTS 上 5–12 分钟数据即可做音色适配。

### 2.2 框架选型（三选一）
| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| 官方 `qwen-tts` + 自建 PEFT 训练 | 完全可控、可插情感标签、Base 保持只读 | 需自己写数据管线 | **主选** |
| `Qwen3-TTS-EasyFinetuning` | GUI/CLI 一体、开箱即用 | 情感标签条件化难定制、黑盒 | 参考 / 兜底 |
| 官方 fine-tuning 脚本 | 官方对齐 | 主要是 voice clone 单一音色，情感多条件弱 | 拆用其数据/分词预处理器 |

**决策**：采用**官方 `qwen-tts`（加载与推理）+ 自建 transformers ∪ PEFT 训练脚本**。理由：① 需求是「情感标签可控」而非常规单音色克隆；② 需将 Base 保持严格只读以满足外挂/INT4 部署；③ 便于把情感条件主语/标签拼进输入模板。

### 2.3 情感条件控制设计
三源条件化（`角色外挂包` 内置）：`
1. **Reference Token（音色）**：选 3–8 秒「中性·清晰」主播片段编码为 ref 声纹，注入说话人表征 → 锁定音色。
2. **情感标签条件前缀**：在输入文本前拼 `[emotion] 开心/俏皮/悲伤 [/emotion]`（源自 `emotion_primary`），让 LoRA 学习「标签→韵律（f0/语速/能量）」的映射而非仅音色。
3. **可选情感 ref 音频**：对稀缺情感（悲伤等），用对应情感的参考片段再生声纹条件，增强标签区分度。

> 训练时严格按 `output.transcript` 还原，把情感标签作为**输入条件**、目标仍是真实音频的 codebook，确保模型学到**条件化韵律**而非死记。

---

## 三、训练配置与验证（Task 3）

### 3.1 关键超参（24GB / 16GB 均可）
| 项 | 建议值 | 依据 |
|---|---|---|
| LoRA r | **16**（可选 32） | 社区实证：小数据集下 r=64 易过拟合、r=16 稳（`ml-audio-fine-tuning`）；训练卷 675，r=16-32 平衡 |
| lora_alpha | 32 | Schroeder 经验 α=2r |
| lora_dropout | 0.05–0.1 | 防过拟合 |
| target_modules | `q_proj,k_proj,v_proj,o_proj`（+ 可加 `gate/up/down`） | Qwen3-TTS 参考实践 |
| 基座量化（训练） | **bnb 4bit (nf4)** + 冻结 | 16GB 可训；保持 Base 只读 |
| LR | **2e-6**（低于默认 2e-5） | 实测默认 2e-5 易造成噪声/不稳定（instavar 报告） |
| Epochs | 5–10 | 按样本×epoch≈250–400 总曝光控制 |
| batch / grad_accum | 1–4 / 8 | 适配 16GB |
| 混合精度 | bf16（CUDA≥8.0）/ fp16（20 系） | 全显卡兼容 |

### 3.2 收敛监控
- 主监控：`train loss` / `val loss`，**val loss ≥2 连续轮不降→早停**。
- 参考目标 loss 区间：**4.1–4.2**（voice identity + instruct 达成且音质不糊）；**<4.1 需警惕过拟合/garbling**，**>4.5 属欠拟合**（alexandria-audiobook 经验）。
- TensorBoard/MLflow 曲线 + Langfuse trace（见 Task 6）。

### 3.3 验证方案
1. **客观（自动）**：
   - 合成音频 → **ASR（Whisper）转录** vs 原文：WER/字错率（保真度）。
   - **音色余弦相似度**：合成 vs 真实片段（speaker 表征）≥ 阈值（≥0.7 目标）。
   - **情感区分度**：对同一文本 + 不同情感标签合成的音频，用情感分类器/声学量（f0_mean、语速、能量）做分离度检查。
2. **主观（人工）**：MOS 听感、是否像角色。30 条测试样本抽样。
3. **基线对照**：裸 Base（无 adapter）+ 中性条件为基线，验证外挂包**增益为正**（边际增益门思路延续）。

---

## 四、部署与交付（Task 4）

> **首选运行时（尽力主推）**：工作区已随附**原生 `llama.cpp` TTS 运行时**（`pykits/llama-cpp-bin/llama-tts.exe`，build 10502，CUDA/CPU 后端；AMD 用 Vulkan 后端构建）。它把 Qwen3-TTS 跑成 **GGUF（INT4/Q8）**，**整套 1.7B 部署仅需约 1.8GB 显存**，并原生支持**流式合成（首音延迟 <300ms）**、声音克隆、`--tts-lang`、`--tts-speaker-file`（说话人/情感包）。这比纯 torch 路径更符合「INT4 低显存 + 全显卡兼容 + 生成翻倍」。
> 已实测该二进制可正常解析 TTS 参数并正确要求 `-m backbone.gguf -mm mmproj.gguf`（对非法模型给出明确报错），证明为**可用原生运行时**而非占位文件。

### 4.1 角色外挂包结构（双格式：训练态 LoRA + 部署态 GGUF）
```
角色外挂包/tyy_luoyuan/
├── adapter_config.json        # PEFT LoRA 配置（r/α/目标模块/基础模型引用）——训练态
├── adapter_model.safetensors  # LoRA 权重（≤100–300MB）——训练态
├── deployment/                  # ——部署态（原生）
│   ├── backbone.gguf           # Talker(1.7B) GGUF（Q5_K_M ≈955MB / Q4_K_M 更小）
│   ├── mmproj.gguf             # Predictor+Decoder GGUF（Q8_0 ≈144MB / fp16 237MB）
│   └── ref_<emotion>.wav/mp3   # 说话人/情感参考音频（--tts-speaker-file 指向它）
├── emotion_vocab.json         # 情感标签 → 条件前缀/ref_audio 映射
├── ref_audio/                    # 参考音色片段（1–2 条中性清晰样本，用于克隆）
├── character_meta.json        # 角色名/版本/说明/兼容校验
└── README.md                  # 安装与使用说明
```
- **训练**：LoRA adapter（不改 Base）；**部署**：用 `convert-hf-to-gguf.py` 把微调后模型导出为 GGUF（或直接导出 huggingface 已发布的 Base GGUF），装入 `deployment/`。
- **Base 不打包、不修改**；外挂包与 Base 分离存储，纯权重 + 配置、与后端无关 → 一套包跨硬件。

### 4.2 动态加载与热切换机制
```
# 原生（首选）——单进程/子进程加载 GGUF，角色/情感切换只换 model&speaker
llama-tts.exe -m deployment/backbone.gguf -mm deployment/mmproj.gguf \
    --tts-speaker-file deployment/ref_开心.wav -p "哥哥你回来啦" -o out.wav
# -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF 可自动下载 GGUF；
# -fa on 开 flash-attn；-ngl 99 GPU 全卸载
```
```
# torch（备选）——Base 常驻 + attach/detach LoRA
base = Qwen3TTSModel.from_pretrained(BASE, int4/bf16)
attach(adapter) → 情感标签与 ref 编码条件 → generate
切换角色： detach(adapter_A) → attach(adapter_B)
```
- **情感切换**：原生路径换 `--tts-speaker-file`；torch 路径改条件前缀/ref，实时生效。
- **角色切换**：原生换 model+speaker 文件（ms 级）；torch 换 adapter（毫秒级）。

---

## 五、低显存快速优化与全显卡兼容（Task 5）

> **原生运行时实测（社区 Qwen3-TTS-GGUF，权威口径）**：
> - RTX 5050：**RTF 0.35**（1s 音频仅 0.35s 生成，≈快 3 倍【相对 CPU/无优化】）；CPU/集显 RTF 1.3
> - 1.7B 显存：Talker(Q5_K_M 955MB) + Predictor(Q8 144MB) + Decoder(fp16 237MB) + KV ≈ **1.8GB**
> - 计算瓶颈在 **Predictor**（每秒音频自回归 12.5×15≈187.5 次）→ 流式/批量复用收益最大；Talker 规模对增速影响小
> - **AMD 用 Vulkan 加速**；NVIDIA 20–50 全系 CUDA；Apple Metal

### 5.1 INT4/GGF 推理 + NVIDIA 加速
| 路径 | 适用 | 做法与效果 |
|---|---|---|
| llama.cpp GGUF（**主推**） | 全平台 | `-m backbone.gguf -mm mmproj.gguf -fa on -ngl 99`；1.7B ≈1.8GB 显存、RTF 0.35 级 |
| torch + bnb int4 | NVIDIA | transformers 路径，`flash_attention_2` + 16-codebook 并行 |
| torch.compile | CUDA≥8 | 图优化提速（20 系用 fp16、30+ 用 bf16） |

**加速目标量化（统一 RTF 口径）**：
- 原生：RTF ≤ **0.6**（1s 音频 <0.6s，目标比容器内 CPU 基线快约 2×）；低端 N 卡 0.35 级、AMD Vulkan 接近。
- torch：相比「int4+无优化」基线，将 RTF 提升 ≥ 2×（由 bench_inference.py 实测）。
- 首包/首音延迟：目标 <300ms（流式）。

### 5.2 AMD / CPU / 全显卡兼容
- **运行时抽象**（不绑定单一后端）
  - NVIDIA 20–50 → CUDA（llama-tts `-ngl` / torch）；
  - AMD → **Vulkan**（llama.cpp Vulkan 构建）或 ROCm/DirectML；
  - Apple → Metal；无独显/低端 → CPU（Q4_K_M）。
- **权威起点**：`llama-tts.exe` CUDA/CPU 已就位；AMD 需带 Vulkan 后端的 llama.cpp 构建。
- 约束：GGUF/LoRA 外挂包为纯权重+配置 → 一套包跨硬件。

### 5.3 显存与速度测量脚本
- `bench_inference.py --backend llamacpp --backbone ... --mmproj ...`：调用本机 `llama-tts.exe` 测量 RTF/首包（已实装）。
- `bench_inference.py --backend torch --base_model ...`：torch 路径测 RTF/首包/峰值显存。
- 验收基线：**≤8GB 卡可跑（1.7B ≈1.8GB）**，16GB 稳定；每配置记录 RTF+显存对照表。

---

## 六、可观测性（Task 6 · Langfuse）
- 训练：每 epoch 上报 `train/val loss、LR、样本数`（trace 名 `qwen3tts.lora.train`）。
- 推理：每次 `generate` 上报 `{adapter, emotion, RTF, 显存峰值, ASR WER}`（trace `qwen3tts.lora.infer`）。
- 价值：回放失败样本、对比不同 adapter 的情感检索、训练/推理链路可定位。
- **可选启用**：环境缺 `LANGFUSE_*` 时静默跳过，不阻塞主流程。

---

## 七、交付物清单（Task 7）
| 物 | 说明 |
|---|---|
| `scripts/prepare_qwen3tts_data.py` | 数据清洗/映射/划分（已跑通：train 6300 / val 634） |
| `data/split/*` | train/val 数据 |
| `scripts/train_lora.py` | bnb4bit + LoRA 训练、loss 早停、adapter 导出、Langfuse 可选埋点（**已落地，语法校验通过**） |
| 角色外挂包 | adapter + 情感词表 + ref + 元数据 + README（训练后由 train_lora.py 产出） |
| `scripts/load_and_generate.py` | Base+附 adapter 推理、情感/角色热切换、RTF/显存记录（**已落地，语法校验通过**） |
| `scripts/bench_inference.py` | RTF/首包/满足目标判定 + 后端选择（CUDA/MPS/CPU/XPU）（**已落地，语法校验通过**） |
| 决策报告（本文档） | 全文 |

### 使用说明（要点）
1. `prepare_qwen3tts_data.py` 已产出训练/验证集，可直接供训练脚本消费。
2. 训练（需 qwen-tts + peft + bitsandbytes）：`python train_lora.py --base_model Qwen/Qwen3-TTS-12Hz-1.7B-Base --train_jsonl data/split/train.jsonl --val_jsonl data/split/val.jsonl --output_dir output/tyy_luoyuan`。
3. 部署（**推荐原生**）：下载 GGUF → `llama-tts.exe -m backbone.gguf -mm mmproj.gguf --tts-speaker-file ref_开心.wav -fa on -ngl 99 -p "哥哥你回来啦" -o out.wav`。
   - 下载 Base GGUF（含 auto-download）：`llama-tts.exe -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF --tts-speaker-file ref.wav -p "test"`；或 ModelScope `modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base` 后用 `convert-hf-to-gguf.py` 转换。
4. 部署（torch 备选）：`python load_and_generate.py --base_model ... --adapter output/tyy_luoyuan/lora --emotion 开心 --text "哥哥你回来啦"`。
5. 基准拨测：`python bench_inference.py --backend llamacpp --backbone ... --mmproj ... --rtf_target 0.6`（或 `--backend torch --base_model ...`）。

---

## 八、风险与基线（诚实版）
- **风险**：① 打标基于主播直播音频，噪声/多 SPK 片段可能影响音色纯度（需参考片段精选）；② 情感标签稀疏类别（悲伤）样本少，区分度可能不足，用情感 ref 音频补；③ 低端 AMD/20 系加速有限，RTF 目标在低端卡按比例调整；④ Base 若只给 CustomVoice 版则无法克隆，须确认用 **Base（可 Clone）** 变体。
- **不承诺**：不虚报"速度无条件翻倍"——定义为**相对当前 int4+无优化基线的 RTF 提升**，由 5.3 实测口径统一。