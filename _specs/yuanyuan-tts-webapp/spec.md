# 缘圆 前端后端一体化 Web 调用框架 Spec

## Why
文本生成引擎（04 后端，llama.cpp CUDA + FastAPI）已可用；Qwen3-TTS 双路外挂包（音色 LoRA + 情感 LoRA + `target_speaker_embedding.pt`）已训练完毕。缺一个把两者统一的前后端一体化「结构化生成 Web 框架」：选模型、选生成方式、调推理参数、出结果（文本 / 音频），便于演示与联调。

本次只**搭建框架**，不触发训练（笔记本无外接电源、约半小时）。

## What Changes
- 在 `06_Qwen3TTS外挂/serve/` 重建（原仅有已损坏的 `__pycache__` 编译缓存）一个统一 FastAPI 服务：
  - `serve/tts_engine.py`：TTS 引擎封装，惰性加载 Base + 双路 adapter + `target_speaker_embedding.pt`，提供 `synthesize(text, emotion) -> (wav, sr)`；缺模型/权重时给出清晰报错而非崩溃。
  - `serve/unified_server.py`：统一 FastAPI 应用，原生挂载 TTS 接口 + 把文本聊天代理到 04 文本引擎（可配置 base URL），并托管静态 Web 前端。
  - `serve/web/`：单页前端（index.html + app.js + style.css），「文本生成 / 语音合成」双 tab，结构化生成表单（选 TTS 音色 adapter、情感标签、文本、推理参数；文本侧转发）。
  - `serve/launcher.py`：一键启动（自动建/复用 venv、清华镜像装依赖、起 uvicorn），配套 `启动前端后端一体化.bat`。
- **不改动** 04 已可用的文本引擎（通过 HTTP 代理复用，避免 GPU 争抢与回归风险）。

## Impact
- Affected code: 新增 `06_Qwen3TTS外挂/serve/` 下若干新文件；不修改 04 后端与已训练外挂包。
- Affected runtime: 端口按需（默认 8070）；文本引擎默认代理 `http://127.0.0.1:8000`（04 独立进程，均可配置）。
- 显存：TTS 模型惰性加载，未调用合成不占显存；文本引擎仍在独立进程。

---

## ADDED Requirements

### Requirement: TTS 引擎封装（惰性 + 单例）
系统 SHALL 提供 TTS 封装服务，懒加载 Base 与双路 adapter 并对外暴露 `synthesize`。

#### Scenario: 加载与合成
- **WHEN** 首次请求 `/api/tts/synthesize`
- **THEN** 惰性加载 Base（modelscope 缓存路径）+ voice adapter + emotion adapter + 角色音色 `target_speaker_embedding.pt`，按情感标签组合输入，返回 24kHz `audio/wav`；缺任何组件返回结构化错误（status+message）而非崩溃。

#### Scenario: 模块/权重缺失
- **WHEN** qwen_tts、Base、adapter 或 embedding 不存在
- **THEN** 返回明确错误（含缺失路径提示），服务其余部分（文本/前端）不受影响。

### Requirement: 统一 FastAPI 服务与前端一体化
系统 SHALL 使用单一 Web 服务同时暴露文本生成、TTS 合成与 Web 页面。

#### Scenario: Web 页面
- **GET /** 返回可交互单页：两个 tab（文本生成 / 语音合成）。
- 语音合成 tab：可选音色 adapter（voice / emotion）、情感标签下拉（开心/俏皮/悲伤/…）、文本输入、推理参数、合成并内嵌播放；展示 RTF/耗时/采样率。
- 文本生成 tab：调用/代理 04 文本引擎，可选角色与温度/top_p/max_new 等参数，流式或一次性展示回复。

#### Scenario: 结构化生成（选模型 → 选生成方式 → 推理）
- **WHEN** 用户在页面上选择模型、生成方式与参数后提交
- **THEN** 前端组装结构化请求，调用对应接口，结果（文本/音频）回填前端，形成可复用的「结构化生成」工作流。

### Requirement: 一键启动
系统 SHALL 提供 launcher：自动创建/复用 venv、按需以清华镜像装依赖、启动统一服务。

#### Scenario: 一键启动
- **WHEN** 用户双击 `启动前端后端一体化.bat`
- **THEN** 完成 venv/依赖/env 准备并启动 `http://127.0.0.1:8070`，可选用 `--open` 自动开浏览器。

## MODIFIED Requirements
无。

## REMOVED Requirements
无。