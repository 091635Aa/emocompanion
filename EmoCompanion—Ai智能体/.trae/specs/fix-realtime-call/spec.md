# 修复实时通话：AI 用定制音色打电话 Spec

## Why

用户反馈"通话没有解决"。全面检测通话链路（真实连上 DashScope Realtime WS 做往返测试）并对照阿里云百炼官方文档（Realtime API 概述 / 客户端事件 / 服务端事件 / 声音复刻 / 通过 WebRTC 实现实时通话最佳实践）后定位：

- **根因**：模型返回的 `response.audio.delta` 事件中音频字段为 `delta`（实测 15 个音频分片全部位于 `delta`），而前端 [通话控制.js](file:///f:/EmoCompanion—Ai智能体/EmoCompanion智能体/前端页面/脚本/通话控制.js#L1128-L1133) 读取 `事件.audio` → 恒为 undefined → **AI 回复只见文字、永远无声**。
- 连接、`session.update`、定制音色（`qwen-omni-vc-emocompanion-voice-20260802030956459-59e8`，绑定 `qwen3.5-omni-plus-realtime`）、文字往返均已在协议层验证通过。

## What Changes

- 修复 [通话控制.js](file:///f:/EmoCompanion—Ai智能体/EmoCompanion智能体/前端页面/脚本/通话控制.js) 的 `response.audio.delta` 处理：音频取 `事件.delta`（保留 `事件.audio` 作为兼容回退）。
- 定制音色通话可用性按 **target_model 校验**：`查询在线音色()` 输出携带 `target_model`；`全部音色()` 按"绑定模型是否为实时模型"复核本地 omni 复刻音色的 `通话可用`，避免误把绑定非实时模型（如 `qwen3.5-omni-flash`）的 `qwen-omni-vc-*` 音色标记为可通话。
- 通话默认自动选用定制音色链路（本地 `音色ID_通话.txt` 读取 → `通话可用` 自动选中）已在协议层验证正确，保持不变；本规格仅加固其判定逻辑。

## Impact

- 受影响规格：plan-v2-omni-realtime（实时通话 / 定制音色）的修复增量。
- 受影响代码：
  - [通话控制.js](file:///f:/EmoCompanion—Ai智能体/EmoCompanion智能体/前端页面/脚本/通话控制.js)（前端音频播放字段）
  - [语音合成.py](file:///f:/EmoCompanion—Ai智能体/EmoCompanion智能体/核心模块/语音合成.py)（在线音色查询输出 target_model）
  - [音色管理.py](file:///f:/EmoCompanion—Ai智能体/EmoCompanion智能体/核心模块/音色管理.py)（通话可用按 target_model 复核）

## ADDED Requirements

### Requirement: AI 回复音频可播放

系统 SHALL 正确播放实时模型返回的音频：`response.audio.delta` 事件的 base64 音频从 `delta` 字段读取（兼容回退 `audio` 字段）。

#### Scenario: 通话中 AI 回复出声
- **WHEN** 用户发起通话并向 AI 说话/发文字，模型生成回复
- **THEN** 对话区显示回复文字，且扬声器播放 AI 语音（16k 输入 / 24k 输出 PCM 链路完整）

#### Scenario: 中途打断
- **WHEN** 模型播报期间用户再次说话（smart/semantic VAD 检测到插话）
- **THEN** 模型立即停止播报并转入聆听，前端状态同步为"打断"

### Requirement: 定制音色通话可用性按绑定模型判定

系统 SHALL 依据声音复刻时的 `target_model` 判定复刻音色是否可用于实时通话：仅绑定 `qwen3.5-omni-plus-realtime` / `qwen3.5-omni-flash-realtime` 的 omni 复刻音色标记为"通话可用"；绑定非实时模型（如 `qwen3.5-omni-flash`）或 TTS 模型的音色仅可用于语音合成。

#### Scenario: 通话自动使用定制音色
- **WHEN** 用户进入通话且存在绑定实时模型的定制音色（当前为 `qwen-omni-vc-emocompanion-voice-20260802030956459-59e8`）
- **THEN** 通话会话 `session.update` 的 `voice` 自动使用该定制音色，AI 以定制声音回复

#### Scenario: 非实时绑定音色不可通话
- **WHEN** 音色面板存在绑定 `qwen3.5-omni-flash`（非实时）的同前缀 omni 音色
- **THEN** 该音色在通话音色列表中标注"仅合成"，不会被自动选用，也不会因绑定模型不一致导致会话合成失败

## MODIFIED Requirements

无（既有能力均保留，仅修复与加固）。

## REMOVED Requirements

无。

## 官方文档依据（2026-07/08 核实）

- 客户端事件：`session.update.session.voice`、`turn_detection.type`（server_vad/semantic_vad）、`input_audio_format=pcm(16k)`、`output_audio_format=pcm(24k)`
- 服务端事件：`response.audio.delta`（base64，示例代码 `base64.b64decode(response['delta'])`）、`response.audio_transcript.delta/done`、`conversation.item.input_audio_transcription.completed`
- 声音复刻：`qwen-voice-enrollment` 创建音色必须指定 `target_model`，后续调用模型必须与之一致，否则合成失败；`qwen-omni-vc-*` 音色绑定实时模型方可实时通话
- 实时通话最佳实践：WebRTC 适合浏览器低延迟场景（内置回声消除/降噪）；当前 WebSocket 桥接协议层已跑通，保持现状

## Sources

- [Qwen-Omni-Realtime（阿里云百炼）](https://help.aliyun.com/zh/model-studio/realtime)
- [Realtime API 概述（AOQ/WebRTC/WebSocket）](https://help.aliyun.com/zh/model-studio/realtime-api-overview)
- [Client events（session.update）](https://help.aliyun.com/en/model-studio/client-events)
- [Server events（response.audio.delta）](https://help.aliyun.com/en/model-studio/server-events)
- [声音复刻 API 参考（qwen-voice-enrollment / target_model）](https://help.aliyun.com/zh/model-studio/qwen-omni-voice-cloning)
- [通过 WebRTC 使用 qwen3.5-omni-plus-realtime 实现实时通话](https://help.aliyun.com/zh/model-studio/best-practice-webrtc-omni-realtime)
