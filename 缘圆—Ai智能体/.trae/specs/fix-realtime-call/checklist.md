# 验收检查清单（修复实时通话：AI 用定制音色打电话）

## 实时通话出声
- [x] 通话中 AI 回复有声音：`response.audio.delta` 分支读取 `事件.delta`（兼容 `事件.audio`），不再静默丢包
- [x] 文本 + 音频双模态回流正常：回复文字显示 + 24kHz PCM 播放
- [x] 中途打断状态联动正常（smart/semantic VAD 插话检测 → 停止播报 → 聆听）

## 定制音色通话
- [x] 通话会话 `voice` 自动使用定制音色 `qwen-omni-vc-yuanyuan-voice-20260802030956459-59e8`（绑定 qwen3.5-omni-plus-realtime）
- [x] 在线音色查询输出包含 `target_model`；本地 omni 复刻音色按绑定模型复核 `通话可用`
- [x] 绑定非实时模型（qwen3.5-omni-flash）的同前缀音色标注"仅合成"，不会用于通话

## 端到端
- [x] 定制音色会话往返实测：`response.audio.delta` 分片 > 0 且字段为 delta
- [x] 应用可启动，通话页无 JS 错误，音色面板与定制音色自动选中正常
- [x] 临时诊断脚本 `诊断通话.py` 已删除
