# RPG 精灵素材智能打标系统（打标 v2.0）Spec

## Why

打标 v1.0（`f:\打标`）只做音频情感打标。用户需要 2.0 版本：处理 RPG Maker MV / MZ 的图片素材（角色精灵图、地图块、头像、战斗图等）。流程为：**按素材规格自动切割** → **调用同一个核心模型（v1.0 的 30B Qwen3-Omni，已部署于 `f:\打标`，可输入图片）逐块分析打标成 JSON** → **写入向量数据库** → **通过 MCP/CLI 供其他文本 AI 先「简单搜索」（省 token）再「查看详情」**。

## 调研结论（联网固化，供分割规则使用）

### RPG Maker MV / MZ 素材规格（48x48 基础格）

| 素材类型 | 目录 | 整图尺寸 | 单格尺寸 | 布局 |
|---|---|---|---|---|
| 角色精灵图（满版 8 角色） | img/characters | 576x384 | 48x48 | 2 行 x 4 列 = 8 个角色；每角色 3 列 x 4 行 = 12 帧（行序：下/左/右/上） |
| 角色精灵图（单角色，`$` 前缀） | img/characters | 144x192 | 48x48 | 3 列 x 4 行 = 12 帧 |
| 头像 | img/faces | MV 384x192 / MZ 576x288 | MV 96x96 / MZ 144x144 | 4 列 x 2 行 = 8 个头像 |
| 图块 A1（动画） | img/tilesets | 768x576 | 48x48 | 特殊：动画块 6x3 + 静态块 2x3 交错 |
| 图块 A2（地面自动图块） | img/tilesets | 768x576 | 48x48 | 每自动块 2x3，4 行 |
| 图块 A3（建筑） | img/tilesets | 768x384 | 48x48 | 每自动块 2x2，8 列 x 4 行 |
| 图块 A4（墙体） | img/tilesets | 768x720 | 48x48 | 2x3 / 2x2 自动块交替行 |
| 图块 A5（普通图块） | img/tilesets | 384x768 | 48x48 | 8 列 x 16 行 |
| 图块 B/C/D/E | img/tilesets | MZ 768x768 | 48x48 | 16 列 x 16 行 |
| 侧视图战斗图 | img/sv_actors, img/sv_enemies | 576x384 | 64x64 | 9 列 x 6 行 |

> **校准策略**：以上为官方/社区资料确认的基准值。MV 与 MZ 在 faces、tileset B-E 上存在差异（表格中已标注），且社区素材尺寸可能不标准。系统以**实际图片尺寸 + 文件名前缀**做自动识别，无法唯一判定时走「可配置覆盖 + 校验报告」，待用户测试数据到位后校准固化。

### 文件名前缀约定
- `$` 开头 = 单角色精灵图（144x192）
- `!` 开头 = 不偏移对象（门/宝箱等），可与 `$` 组合（`!$`）
- `!` 开头 Parallax = 当作地板

## What Changes

- 在 `j:\打标（RPG）` 从零构建新项目，沿用 v1.0 的**中文命名分层风格**：数据层 / 业务逻辑层 / 接口层 + 配置 + 日志。
- 数据驱动**素材规格表**：固化 MV/MZ 各素材类型切割规则（上表），支持按实际尺寸自动识别与手动覆盖。
- **自动分割**：把精灵图 / 图块 / 头像等图集按规格切为单格 PNG，输出坐标清单（供追溯）。
- **AI 打标**：复用 v1.0 的 30B 核心模型（llama-server `http://127.0.0.1:8766`，OpenAI 兼容 `/v1/chat/completions`，`image_url` 图片输入），为每格生成结构化 JSON 标签；解析失败自动重试。
- **向量索引**：本地小模型（CPU，默认 `BAAI/bge-small-zh-v1.5`）生成文本嵌入，写入 ChromaDB（本地持久化），查询不占用 30B 显存/token。
- **查询接口（省 token）**：
  - MCP 服务（stdio）：`search_assets` / `get_asset` / `list_assets` / `asset_stats`
  - CLI：`python 查询.py search "森林"`、`detail <id>`、`list [类型]`
  - 搜索返回**紧凑一行式**结果（素材ID、名称、类型、一句话描述、标签），详情接口才返回完整 JSON，最大程度节省 token。

## Impact

- 影响规格：全新项目，无既有规格受影响。
- 影响代码：`j:\打标（RPG）` 当前为空目录，全部为新建代码。
- 复用资源：30B 模型服务由 `f:\打标` 提供（llama-server 已运行于 127.0.0.1:8766），**无需重新下载大模型**。
- 外部依赖（清华源 pip 安装）：`chromadb`、`mcp`、`sentence-transformers`、`Pillow`、`numpy`。
- 嵌入模型下载：ModelScope（国内快）为主 / HF 镜像为备，缓存到 `l:\RAG模型缓存`。

## ADDED Requirements

### Requirement: 项目骨架与中文命名规范
系统 SHALL 在 `j:\打标（RPG）` 从零搭建模块化项目结构，包含 数据层、业务逻辑层、接口层；文件、目录、变量名 SHALL 使用中文命名，注释 SHALL 使用中文；SHALL 提供统一 JSON 配置文件 `配置\系统配置.json`。

#### Scenario: 首次初始化
- **WHEN** 运行初始化脚本
- **THEN** 生成完整中文目录树与默认配置，目录含 源素材 / 分割素材 / 打标结果 / 向量库 / 汇总报告 / 进度 / 日志 等

### Requirement: 素材规格表（数据驱动）
系统 SHALL 内置素材规格注册表，覆盖 RPG Maker MV/MZ 的 characters / faces / tilesets（A1-A5、B-E）/ sv_actors / sv_enemies 等类型，每条记录包含：类型名、对应目录前缀、整图尺寸、单格尺寸、行列布局、文件名前缀语义（`$`/`!`）、自动图块特殊布局说明；SHALL 支持按文件实际尺寸自动匹配规格，支持在配置中手动覆盖单文件规格；匹配不确定时 SHALL 输出校验警告并允许跳过。

#### Scenario: 自动识别
- **WHEN** 输入一张 576x384 且无 `$` 前缀的角色图
- **THEN** 自动判定为 MV/MZ 满版角色精灵图（8 角色 x 12 帧），无需人工配置

#### Scenario: 尺寸不标准
- **WHEN** 图片尺寸与注册表全部条目不匹配
- **THEN** 输出校验警告到日志与汇总报告，文件标记为「未识别」并可手动配置切割规格后重跑

### Requirement: 素材自动分割
系统 SHALL 将每个源图集按匹配到的规格切割为单格 PNG；角色图按「8 角色 12 帧」或「单角色 12 帧」切割，头像按 8 格切割，图块按 48x48 网格切割（A1-A4 自动图块按 2x3/2x2 自动块切为「整块」并保留元信息）；SHALL 输出切割坐标清单（JSON，含源文件、行、列、像素区域、角色/图块序号）；SHALL 支持断点续切（已切的源文件跳过）。

#### Scenario: 满版角色图切割
- **WHEN** 对 576x384 角色精灵图执行分割
- **THEN** 输出 8 个角色子图（每张 144x192），每个角色再保留 12 帧坐标，命名含角色序号（如 `Hero_角色3.png`）

#### Scenario: 断点续切
- **WHEN** 分割中途中断后再次执行
- **THEN** 已完成的源文件跳过，仅处理未完成的

### Requirement: AI 打标（图片 → JSON）
系统 SHALL 复用 v1.0 的 30B 核心模型（Qwen3-Omni，llama-server OpenAI 兼容 `/v1/chat/completions`，图片以 `image_url`/base64 传入）对每个分割格生成结构化 JSON 标签；JSON schema SHALL 至少包含：素材ID、来源文件、切割坐标、类型、内容描述、视觉特征、标签列表、适用场景、置信度；输出 SHALL 通过 schema 校验，解析失败自动重试（换随机种子），重试耗尽标记失败并归档。

#### Scenario: 单格分析
- **WHEN** 输入一张角色帧 PNG
- **THEN** 返回符合 schema 的结构化标签 JSON，字段完整、格式合法，写入 `数据层\打标结果`

### Requirement: 打标流水线（断点续打 + 并行）
系统 SHALL 提供全自动流水线：读取分割清单 → 逐格调用 30B 模型打标 → 结果落盘；SHALL 记录进度（每格状态），中断后断点续打；SHALL 支持多线程并行（默认 2，可配置）；SHALL 单格失败重试 N 次（默认 3），失败隔离不中断整体；SHALL 输出日志与汇总。

#### Scenario: 断点续打
- **WHEN** 打标中途中断后再次启动
- **THEN** 已完成的格子不再重复处理，仅处理未完成的

### Requirement: 向量索引
系统 SHALL 将打标结果中的（类型 + 内容描述 + 标签）文本经本地小模型（CPU，默认 `BAAI/bge-small-zh-v1.5`）生成嵌入，写入 ChromaDB（本地持久化目录 `数据层\向量库`）；每条记录 SHALL 保存：素材ID、名称、类型、内容描述、标签、来源文件、图片相对路径、缩略图路径、原始 JSON 路径；SHALL 支持全量重建索引与增量追加。

#### Scenario: 建立索引
- **WHEN** 打标完成后执行索引
- **THEN** 向量库写入全部素材记录，可按文本语义检索

### Requirement: MCP 查询服务（省 token）
系统 SHALL 提供 MCP 服务（stdio 传输，官方 `mcp` SDK），暴露工具：
- `search_assets(查询文本, top_k)`：返回**紧凑列表**（每条一行式：素材ID、名称、类型、一句话描述、标签），供 AI 先做概览搜索；
- `get_asset(素材ID)`：返回完整 JSON 详情 + 图片/缩略图相对路径；
- `list_assets(类型?)`：按类型列举素材（紧凑格式）；
- `asset_stats()`：素材总数 / 类型分布 / 索引状态。

#### Scenario: AI 先搜后查
- **WHEN** 文本 AI 调用 `search_assets("森林场景草地")`
- **THEN** 返回 Top-k 紧凑结果（低 token）；AI 选中某 ID 后再调 `get_asset` 拿到完整详情

### Requirement: CLI 查询入口
系统 SHALL 提供命令行入口 `python 查询.py`，支持：`search <文本> [--top-k N]`、`detail <素材ID>`、`list [类型]`、`stats`；输出与 MCP 工具一致的紧凑/详情格式。

#### Scenario: 命令行检索
- **WHEN** 执行 `python 查询.py search "沙漠" --top-k 3`
- **THEN** 终端输出 3 条紧凑素材结果

### Requirement: 汇总报告
系统 SHALL 在打标与索引完成后生成汇总报告：素材总数、各类型数量、打标成功率、失败清单、未识别文件清单、向量库统计；输出为 JSON（机器可读）+ 简单 HTML（人类可读，可选）。

#### Scenario: 汇总生成
- **WHEN** 全量任务结束
- **THEN** 生成 `数据层\汇总报告` 下 JSON 与 HTML 报告
