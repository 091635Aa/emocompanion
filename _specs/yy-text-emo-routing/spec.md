# 缘圆文本生成主播化优化 —— 情感向量外挂路由 + 行为层 Spec

## Why
现有文本生成（04 文本引擎 llama.cpp + Qwen3-4B，外挂路由：persona + P3 logits 偏置 + 去AI腔）回复**不像主播**：
1. persona 硬性限制"最多 1~3 句、说完就停"，导致永远不写小作文、长度不随语境变化；
2. 情感仅靠采样参数（temperature/top_k）随机起伏，没有"引擎优化情感 + 角色本身情感"的显式向量路由，情感方向不可控；
3. 前端只有"对方正在输入…"，没有"对方正在发送语音"指示，语音回复阶段的等待感不真实。

本任务采用**外挂路由（角色包式）**：不改主模型权重，扩展 logits 级情感向量注入 + 主播化行为层，让回复更像真人主播——安慰时发小作文、日常时多说几句、句句都有回应，并在 TTS 合成时显示"对方正在发送语音"。

## What Changes
- **文本引擎外挂路由扩展**：在 [engine.py](file:///d:/AI情感/缘圆_角色挂载与情感注入工程/04_源码与原型/backend/engine.py) 的 `PersonaLayers` 新增"情感向量偏置层"，logits 级注入，主模型权重不变：
  - `v_eff = 0.7 × v_角色情感 + 0.3 × v_引擎优化情感`
  - `v_引擎优化情感`：由情感识别（emo_detect）结果查情感向量表得到（**引擎给的优化情感，占 30%**）
  - `v_角色情感`：角色本身情感向量，来自角色包（**占 70%**）
  - 合成后以 `β·tanh(v_eff · 相关度 / T)` 方式逐 token 注入 logits（复用 P3 限幅机制，机制不可坍缩）
- **角色包数据扩展**：[build_role_pack.py](file:///d:/AI情感/缘圆_角色挂载与情感注入工程/04_源码与原型/backend/build_role_pack.py) 预计算"8 情感 × V 维"情感向量表 + 角色本身情感向量，运行期零 transformers，热切换角色包即换情感基底。
- **行为层改写**：改写 persona（去掉"最多 1~3 句"硬限制），新增长度自适应（安慰→小作文、日常→几句）+ 句句有回应兜底。
- **前端指示**：TTS 合成阶段显示"对方正在发送语音…"（区别于"对方正在输入…"）。

## Impact
- Affected code:
  - `04_源码与原型/backend/engine.py`（PersonaLayers 情感向量层 + chat/chat_stream 接口）
  - `04_源码与原型/backend/server.py`（/chat、/chat/stream 支持 emotion 向量参数透传）
  - `04_源码与原型/backend/build_role_pack.py`（情感向量表预计算，产出新数据文件）
  - `06_Qwen3TTS外挂/serve/integrated_app.py`（emo_detect→查表→传向量；persona 更新；长度自适应）
  - `06_Qwen3TTS外挂/serve/webapp/app.js` + `index.html`（"对方正在发送语音"指示）
- 不改变主模型权重（仍 llama.cpp GGUF，不污染 Base）；角色包可热切换。
- 运行期零 transformers 依赖；情感向量计算全部在构建期完成。

---

## ADDED Requirements

### Requirement: 情感向量路由层（logits 级，引擎侧）
系统 SHALL 在文本引擎解码期以 logits 偏置方式注入情感向量路由，不修改主模型权重。

#### Scenario: 情感向量合成注入
- **WHEN** 每次生成对话回复时
- **THEN** 按 `v_eff = 0.7 × v_角色情感 + 0.3 × v_引擎优化情感` 合成有效情感向量，并作为每步 logits 偏置注入（`β·tanh` 限幅，β 可调，默认复用 P3 标定值 β=1.0 × 0.75）；
- **THEN** 引擎主模型权重不变，热切换角色包仅替换情感向量基底，无需重新加载权重。

#### Scenario: 情感识别→查表
- **WHEN** 用户发送消息进入对话
- **THEN** 先用 `emo_detect`（词典优先 → LLM 兜底）识别情感标签，映射到情感向量表对应行作为 `v_引擎优化情感`（30% 权重）；
- **THEN** 情感识别失败或离线时回退到默认"平静"向量，不阻塞主流程。

### Requirement: 角色包扩展情感向量表（构建期预计算）
系统 SHALL 在角色包构建期预计算"8 情感 × V 维"引擎优化情感向量表 + 角色本身情感向量，运行期零 transformers。

#### Scenario: 情感向量表产出
- **WHEN** 执行 `build_role_pack.py`
- **THEN** 基于 Qwen3-4B 嵌入权重与情感锚点构造 8 情感（开心/俏皮/悲伤/平静/兴奋/撒娇/温柔/激动）的引擎优化情感向量（`emotion_vectors.npy`，V×8）；
- **THEN** 基于打标数据/角色特征谱聚合得到角色本身情感向量 `v_角色情感`，一并写入角色包（`role_pack.json` meta + 独立 .npy）；
- **THEN** 角色包数据 ≤ 数 MB（V×8 fp32 ≈ 3.2MB@V=100k），符合角色包轻量约束。

### Requirement: 主播化 persona 与长度自适应（行为层）
系统 SHALL 改写 persona 并加入长度自适应，使回复更像真人主播：安慰时发小作文、日常时多说几句、句句都有回应。

#### Scenario: 长度自适应
- **WHEN** 情感识别为悲伤/温柔/安慰类语境
- **THEN** 调大 `max_new` 并注入"多写几句、展开安慰"指令（小作文模式），回复显著变长；
- **WHEN** 日常寒暄/开心/俏皮语境
- **THEN** 使用中等长度（几句），保持口语化与节奏。

#### Scenario: 句句有回应
- **WHEN** 用户发送任何一条消息
- **THEN** 引擎保证返回非空回复；若文本引擎返回空/过短（<2 字），自动重试一次并附兜底 persona 指令。

### Requirement: 前端"对方正在发送语音"指示
系统 SHALL 在 TTS 合成阶段于前端显示"对方正在发送语音…"指示，与文本生成阶段的"对方正在输入…"区分。

#### Scenario: 语音合成等待指示
- **WHEN** 文本已生成完毕、进入 TTS 合成阶段（want_tts=true）
- **THEN** 前端将状态从"对方正在输入…"切换为"对方正在发送语音…"，音频就绪后恢复；
- **WHEN** 未开启语音回复（want_tts=false）
- **THEN** 仅显示"对方正在输入…"，不出现语音指示。

## MODIFIED Requirements
无（本能力为对既有文本引擎外挂路由的增强，不推翻 P3/去AI腔；原 persona 的"最多 1~3 句"约束被行为层改写，属本 spec 范围内修改）。

## REMOVED Requirements
### Requirement: 文本回复长度硬上限（"最多 1~3 句、说完就停"）
**Reason**: 与"主播化、安慰发小作文、多说几句"目标冲突。
**Migration**: 由"长度自适应"（按情感/语境动态决定 max_new 与长度指令）替代；默认仍保持口语化不啰嗦，仅在安慰/深聊语境放长。
