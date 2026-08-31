# 缘圆智能体 V2（全模态实时通话）规格

## Why

现有 [tts_studio](file:///f:/缘圆—Ai智能体/tts_studio) 仅提供基础的 Qwen-Audio TTS 语音合成（V1），音色定制停留在 [voice_clone_work](file:///f:/缘圆—Ai智能体/voice_clone_work) 的脚本阶段。用户需要把产品升级为**可实时通话、可打断、可切换音视频、可定制音色、可多模态识别**的完整 Web 应用，并将 V1 冻结归档为 TDS 版本。

## 版本规划

| 版本 | 代号 | 定位 | 处置 |
|---|---|---|---|
| V1 | TDS 版本 | 基础 TTS 合成工作台（现有 tts_studio） | 完整拷贝归档到 V2 项目 `归档/TDS版本/`，功能冻结，不再迭代 |
| V2 | 缘圆智能体 | 实时通话 + 定制音色 + 多模态识别 + 交互增强 | 新建独立目录，全中文编码开发 |

V2 架构基线（已与用户确认）：
- **目录形态**：新建独立目录 `f:\缘圆—Ai智能体\缘圆智能体\`，V1 归档至其 `归档/TDS版本/`
- **前端技术栈**：原生 HTML/CSS/JS 多文件，零构建、离线可用
- **实时通话协议**：第一阶段 WebSocket 后端中转（FastAPI 桥接 DashScope Realtime WS），后续可平滑升级 WebRTC

## 技术情报（2026-08 已核实，来源见文末）

- 全模态实时模型：`qwen3.5-omni-plus-realtime`（3.5 全模态）/ `qwen3-omni-flash-realtime`（3.0 全模态），两者均可通过 `session.update` 设置音色 `voice`（3.5 默认 `Tina`，3.0 默认 `Cherry`）
- Realtime 端点：`wss://{业务空间ID}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=<模型名>`，兜底 `wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=<模型名>`
- 输入音频固定 `PCM_16000HZ_MONO_16BIT`（16kHz 单声道 16bit），输出固定 `PCM_24000HZ_MONO_16BIT`（24kHz）
- 打断/轮次检测：`enable_turn_detection=True`；`turn_detection.type=smart_turn`（语义打断，可区分"嗯/啊"等附和与真插话）为默认，`server_vad`（声学 VAD）为回退方案
- 视频模式 = 音频流 + 周期性图像帧（Realtime 模型原生支持流式图像输入，如 1FPS 抽帧）
- 音色克隆：上传一段录音即可定制专属音色（Qwen3.5-Omni 原生支持；DashScope voice-enrollment 复刻方案已在 voice_clone_work 验证）

## What Changes

- 新建 `缘圆智能体/` V2 项目（全中文目录/文件名/变量名/函数名），实现实时通话、定制音色、多模态识别、前端交互增强四大能力
- 将现有 tts_studio 完整归档为 `归档/TDS版本/`（含源码、依赖、.env 模板、README 说明）
- 将 V1 的语音合成、配置持久化、DeepSeek 智能助手能力以中文命名迁移至 V2 核心模块
- 定制音色从"命令行脚本"升级为"Web 全流程"：数据集上传 → 最优片段筛选 → 音色注册 → 音色面板管理
- 对话界面新增状态指示（logo 动画、呼吸灯）、图像长按/拖拽上传、5 秒视频录制、屏幕截图
- 集成 `/spec`（查看规格）与 `/goal`（查看目标与进度）对话命令
- **中文编码规范**：目录、文件名、变量名、函数名、类名、JSON 键全中文；保留 Python 魔法文件名（`__init__.py`、`__main__.py`）为英文；若 PyInstaller 打包/import 出现技术性阻断，按"英文实现 + 中文注释"回退，并在本文件记录回退点

## Impact

- 受影响规格：V1 TTS 工作台能力（合成/模型/音色/风格/发音纠正/AI 标注）全部迁移至 V2
- 受影响代码：
  - [app.py](file:///f:/缘圆—Ai智能体/tts_studio/app.py)、[tts.py](file:///f:/缘圆—Ai智能体/tts_studio/tts.py)、[config.py](file:///f:/缘圆—Ai智能体/tts_studio/config.py)、[deepseek.py](file:///f:/缘圆—Ai智能体/tts_studio/deepseek.py)、[static/index.html](file:///f:/缘圆—Ai智能体/tts_studio/static/index.html)
  - [voice_clone_work](file:///f:/缘圆—Ai智能体/voice_clone_work) 全部脚本（select_best_segment/enroll_voice/synthesize/verify_quality/upgrade_voice/synth_emotion）
- 新依赖：`websockets`、`numpy`、`scipy`、`soundfile`（pip 使用清华镜像安装）

## ADDED Requirements

### Requirement: V2 项目目录结构（全中文）

系统 SHALL 按以下目录结构建立 V2 项目（`f:\缘圆—Ai智能体\缘圆智能体\`）：

```
缘圆智能体/
├── 启动入口.py                # 启动 Web 服务并自动打开浏览器
├── 环境配置.py                # 加载密钥配置.env、路径管理、中文编码初始化
├── 配置文件.json              # 持久化：别名/最近使用/音色库/会话设置
├── 密钥配置.env               # API Key（不含真实密钥入库）
├── 依赖清单.txt               # pip 依赖
├── 打包配置.spec              # PyInstaller 配置
├── 图标.ico
├── 核心模块/                  # Python 包（含 __init__.py 魔法文件）
│   ├── 语音合成.py            # TTS 合成（迁移自 V1 tts.py）
│   ├── 实时通话.py            # Realtime WS 桥接：打断/模型切换/音视频模式
│   ├── 多模态识别.py          # 语音识别 + 图像识别 + 视频识别
│   ├── 音色管理.py            # 内置音色 + 定制音色 + 本地 voice_id 读取
│   ├── 智能助手.py            # DeepSeek 标注/指令优化（迁移自 V1 deepseek.py）
│   └── 配置持久化.py          # 配置文件.json 读写（迁移自 V1 config.py）
├── 定制音色/
│   ├── 数据集处理.py          # 上传音频重采样/拼接/分段
│   ├── 最优片段筛选.py        # 评分优选 10~20s 片段（迁移自 select_best_segment.py）
│   ├── 音色注册.py            # DashScope 注册/克隆（迁移自 enroll_voice.py）
│   ├── 质量校验.py            # 合成验证（迁移自 verify_quality.py）
│   ├── 声音库/
│   │   ├── 音色ID.txt         # 最近注册的 voice_id
│   │   ├── 最优片段.wav/.mp3  # 筛选输出
│   │   ├── 选择报告.json       # 评分报告
│   │   └── 数据集/分段音频/    # 用户上传数据集
│   └── 合成输出/              # 试听音频
├── 前端页面/
│   ├── 页面入口.html
│   ├── 页面样式.css
│   ├── 脚本/
│   │   ├── 主控.js            # 应用状态机与页面逻辑
│   │   ├── 请求封装.js        # fetch / WebSocket 封装
│   │   ├── 通话控制.js        # 实时通话：麦克风采集/播放/会话状态
│   │   ├── 媒体采集.js        # 视频录制(5s)/截图/图像上传/拖拽
│   │   ├── 音色面板.js        # 音色选择与定制音色管理
│   │   └── 对话界面.js        # 气泡、状态指示（logo 动画/呼吸灯）
│   └── 资源/
│       ├── 图标.svg           # 呼吸灯/logo 动画素材
│       └── 音效/
├── 归档/TDS版本/              # V1 完整快照
└── 数据缓存/
    ├── 音频缓存/
    ├── 录制视频缓存/
    ├── 截图缓存/
    └── 上传图像缓存/
```

#### Scenario: 目录就绪与中文导入
- **WHEN** 开发人员在 `缘圆智能体/` 下运行 `python 启动入口.py`
- **THEN** 服务在 127.0.0.1:8000 启动，浏览器自动打开；`from 核心模块 import 语音合成` 等中文导入全部正常；无中文编码/导入阻断

### Requirement: 实时通话系统

系统 SHALL 提供支持中途打断的人机实时通话：默认语音通话模式，可一键切换视频通话模式；支持千问 3.5 全模态（`qwen3.5-omni-plus-realtime`）与 3.0 全模态（`qwen3-omni-flash-realtime`）模型切换。

技术方案：
- 链路：浏览器麦克风（AudioWorklet 采集，下采样 16kHz PCM 单声道）→ `前端页面/脚本/通话控制.js` → FastAPI WebSocket 中转 → `核心模块/实时通话.py` 桥接 DashScope Realtime WS → 24kHz PCM 回流 → 浏览器 AudioContext 播放
- 会话配置：`session.update` 设置 `output_modalities=[TEXT, AUDIO]`、`voice`、`turn_detection.type=smart_turn`（回退 `server_vad`）、`enable_turn_detection=True`
- 打断：服务端 smart_turn/server_vad 自动轮次检测；模型播报中检测到用户说话即打断（客户端无额外逻辑，仅需状态指示）
- 视频模式：切换后浏览器以约 1FPS 抽帧摄像头画面，以图像事件随音频流发送；退出视频模式停止抽帧
- 模型切换：通话空闲时选择模型 → 重连建立新会话
- 状态机：`空闲 → 连接中 → 会话建立 → 聆听中 → 回复中 →（打断）聆听中 → 挂断`

#### Scenario: 语音通话与打断
- **WHEN** 用户点击"开始通话"，说话提问，在模型回复中途再次说话
- **THEN** 通话建立，模型语音/文字回流；用户中途插话后模型立即停止播报并转入聆听；对话日志完整记录双方内容
#### Scenario: 视频通话切换
- **WHEN** 通话中用户点击"切换视频"
- **THEN** 摄像头授权弹窗出现（用户手动允许），授权后画面实时抽帧随音频流发送，模型可"看见"画面内容
#### Scenario: 模型切换
- **WHEN** 用户在设置中选择"3.0 全模态"并开始新通话
- **THEN** 新会话使用 `qwen3-omni-flash-realtime`，音色默认切换为 `Cherry`（3.5 默认为 `Tina`）

### Requirement: 定制音色功能

系统 SHALL 提供从数据集筛选最优语音片段并注册定制音色的全 Web 流程，复用 voice_clone_work 已验证方案；实现阶段联网检索 Qwen3.5-Omni 录音克隆最新接口并择优接入。

流程：上传数据集 → `数据集处理.py` 重采样 24kHz 单声道、拼接、分段 → `最优片段筛选.py` 滑窗评分（硬约束：语音占比 ≥ 0.55、最大停顿 ≤ 2.0s、时长 10~20s；评分 = 0.35×信息密度 + 0.25×SNR + 0.20×语音占比 + 0.20×能量调制系数）→ `音色注册.py` 调 DashScope voice-enrollment（或 3.5-Omni 克隆接口）→ 保存 voice_id 与选择报告 → 出现在音色面板

#### Scenario: 数据集定制音色
- **WHEN** 用户在音色面板上传一段 ≥ 1 分钟录音并点击"定制音色"
- **THEN** 后端完成筛选与注册，音色面板新增该定制音色（含试听片段与评分报告），可在合成与通话中选用
#### Scenario: 定制音色管理
- **WHEN** 用户对定制音色执行重命名、试听、设为默认、删除
- **THEN** 操作生效并持久化到 配置文件.json 与本地音色ID文件

### Requirement: 多模态识别

系统 SHALL 集成三大识别模块：语音识别、图像识别、视频识别，统一走 DashScope 多模态接口（`核心模块/多模态识别.py`）。

- 语音识别：Qwen-Audio 系列 ASR（如 `qwen-audio-3.0-asr` 流式/离线），支持音频上传识别与通话中实时转写
- 图像识别：图像上传后模型理解并回复（可与通话上下文串联）
- 视频识别：上传视频文件（或录制视频）后模型理解并回复

#### Scenario: 三种识别
- **WHEN** 用户分别上传音频、图片、视频并点击识别
- **THEN** 各返回对应模型的识别/理解结果，识别结果作为对话内容展示并可继续追问

### Requirement: 前端视频录制

系统 SHALL 在 Web 界面提供调用设备摄像头录制 5 秒视频的功能，录制前弹出摄像头授权（用户手动授权）。

#### Scenario: 录制 5 秒视频
- **WHEN** 用户点击"录制视频"，浏览器请求摄像头权限
- **THEN** 用户手动允许后进入 5 秒倒计时录制，结束自动保存并可上传用于视频识别

### Requirement: 前端屏幕截图

系统 SHALL 支持对页面/屏幕截图并上传。

#### Scenario: 截图上传
- **WHEN** 用户点击"截图"，选择截取当前页面或屏幕区域
- **THEN** 截图生成 PNG 并显示在对话输入区，可发送给模型识别

### Requirement: 前端图像上传（长按 + 拖拽）

系统 SHALL 支持长按上传图像与拖拽图像至对话层两种方式。

#### Scenario: 长按上传
- **WHEN** 用户在移动端/触屏对话输入区按住 1 秒
- **THEN** 触发文件选择器，选中图片后加入对话并上传
#### Scenario: 拖拽上传
- **WHEN** 用户将图片文件拖拽至对话层释放
- **THEN** 图片预览加入对话并上传

### Requirement: 对话状态指示

系统 SHALL 在对话过程中显示回复状态提示：logo 呼吸灯动画、语音/视频通话状态指示（聆听/回复/打断/连接中）。

#### Scenario: 状态可视
- **WHEN** 模型回复、用户插话打断、通话连接中等状态变化
- **THEN** 对话区 logo 动画与呼吸灯颜色/节奏同步变化，用户可即时感知回复状态

### Requirement: 音色选择与管理界面

系统 SHALL 提供内置音色选择与定制音色管理界面（列表、试听、重命名、设为默认、删除、定制入口）。

#### Scenario: 选择与管理音色
- **WHEN** 用户在音色面板选择内置音色或管理定制音色
- **THEN** 所选音色用于合成与通话；管理操作即时生效并持久化

### Requirement: 中文编码规范（V2）

V2 全部代码 SHALL 使用中文命名（目录、文件、变量、函数、类、JSON 键），保留 Python 标准魔法文件名；如遇 PyInstaller/import/第三方库技术性阻断，回退为英文命名 + 中文注释，并记录回退点至本文档。

#### Scenario: 全中文开发
- **WHEN** 开发者浏览 V2 代码
- **THEN** 目录结构、函数名、变量名均为中文，逻辑自解释，无需额外映射

### Requirement: /spec 与 /goal 集成

系统 SHALL 在对话输入框支持 `/spec` 与 `/goal` 命令：`/spec` 展示当前规格文档要点（读取 `.trae/specs` 下的 spec.md）；`/goal` 展示当前开发目标与任务进度（读取目标状态）。

#### Scenario: 斜杠命令
- **WHEN** 用户在对话输入框输入 `/spec` 或 `/goal`
- **THEN** 对话区展示对应规格/目标摘要，不调用模型

## MODIFIED Requirements

### Requirement: 语音合成工作台（V1 迁移）

V2 SHALL 迁移并保留 V1 全部合成能力：模型列表/别名、系统音色/复刻音色、情绪/富语言标签（30 个）、全局风格指令与场景预设、多音字发音纠正、DeepSeek AI 自动标注与指令优化、余额探测。实现基于 [tts.py](file:///f:/缘圆—Ai智能体/tts_studio/tts.py) 与 [app.py](file:///f:/缘圆—Ai智能体/tts_studio/app.py) 中对应逻辑，以中文命名落地到 `核心模块/语音合成.py`、`核心模块/配置持久化.py`、`核心模块/智能助手.py`，合成 API 迁移为 `/api/合成` 等中文路径（同时保留英文路径别名以兼容历史调用，视实现取舍）。

#### Scenario: 中文 API 合成
- **WHEN** 前端调用中文路径 API 并携带模型/音色/文本/指令
- **THEN** 返回合成音频 URL 与元信息，功能与 V1 等价

## REMOVED Requirements

### Requirement: V1 独立运行（tts_studio）

**Reason**: V1 作为 TDS 版本冻结归档，功能并入 V2。
**Migration**: 完整拷贝 `tts_studio` 至 `缘圆智能体/归档/TDS版本/`（含源码、requirements、.env 模板、README 说明）；原目录保留不动，仅作历史参考。

## 开发计划（阶段划分）

1. **阶段一 · 归档与骨架**：V1 归档 TDS；V2 目录骨架、环境配置、配置持久化、启动入口、前端空壳
2. **阶段二 · 核心迁移**：语音合成/智能助手/音色管理迁移（中文命名），合成链路跑通
3. **阶段三 · 实时通话**：Realtime WS 桥接、状态机、打断、模型切换、语音/视频模式
4. **阶段四 · 定制音色**：数据集处理/筛选/注册 Web 化 + 联网检索克隆接口
5. **阶段五 · 多模态识别**：语音/图像/视频识别
6. **阶段六 · 前端交互**：视频录制、截图、长按/拖拽上传、状态指示、音色面板
7. **阶段七 · 集成收尾**：/spec、/goal 命令、打包、端到端验证

## Sources

- [Qwen-Omni-Realtime（阿里云百炼）](https://help.aliyun.com/zh/model-studio/realtime)
- [Realtime API 概述（WebSocket/WebRTC/AOQ）](https://help.aliyun.com/zh/model-studio/realtime-api-overview)
- [Qwen-Omni-Realtime Python SDK](https://help.aliyun.com/zh/model-studio/omni-realtime-python-sdk)
- [Qwen-Audio 实时语音对话 WebSocket API](https://help.aliyun.com/zh/model-studio/fun-audiochat-realtime-websocket-api)
- [阿里千问全模态 Qwen3.5-Omni 上线（上证报）](http://www.cnstock.com/commonDetail/660380)
- [Qwen3.5-Omni 能力评测（语义打断/音色克隆）](https://www.163.com/dy/article/KPF2VN290556C3IR.html)
