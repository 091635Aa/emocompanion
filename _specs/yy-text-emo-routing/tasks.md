# Tasks

- [x] Task 1: 角色包扩展 —— 情感向量表预计算（build 期）
  - [x] 1.1 在 `build_role_pack.py` 中构造 8 情感（开心/俏皮/悲伤/平静/兴奋/撒娇/温柔/激动）的锚点，基于 Qwen3-4B 嵌入权重生成 `emotion_vectors.npy`（V×8，引擎优化情感向量）
  - [x] 1.2 从打标数据/角色特征谱聚合角色本身情感向量 `v_角色情感`，写入 `role_pack.json` meta + 独立 .npy
  - [x] 1.3 运行 `build_role_pack.py` 产出新数据文件，验证文件大小 ≤ 数 MB 且能被 json/npy 加载

- [x] Task 2: 引擎情感向量路由层（engine.py）
  - [x] 2.1 在 `PersonaLayers` 新增情感向量偏置层：接受 `v_角色情感` + `v_引擎优化情感`，按 `v_eff = 0.7×角色 + 0.3×优化` 合成，`β·tanh` 限幅后逐 token 注入 logits
  - [x] 2.2 `chat` / `chat_stream` 增加参数：可传入情感向量（或情感标签 → 内部查表），支持强度旋钮 `scale_emo`，关闭时为 0 不影响原 P3/去AI腔
  - [x] 2.3 语法校验 + 单测：给定 v_eff 时 logits 偏置正确叠加、关闭时输出与裸引擎一致

- [x] Task 3: 服务层透传（server.py）
  - [x] 3.1 `/chat` 与 `/chat/stream` 请求模型新增 `emotion` / `emo_vector` / `scale_emo` 字段，透传到 engine
  - [x] 3.2 `/debug` 输出情感向量路由状态（当前情感、权重、scale_emo），便于联调
  - [x] 3.3 语法校验 + 调用冒烟（POST /chat 带 emotion 参数返回 200）

- [x] Task 4: 集成层联动（integrated_app.py）
  - [x] 4.1 在 talk/stream 生成前，用 `emo_detect_fast` 识别用户消息情感 → 查情感向量表得到 `v_引擎优化情感`，随请求传给文本引擎
  - [x] 4.2 情感识别失败时回退默认"平静"向量，不阻塞主流程
  - [x] 4.3 更新 persona：去掉"最多 1~3 句"硬限制，加入长度自适应指令（安慰→小作文、日常→几句）
  - [x] 4.4 长度自适应：按情感/语境动态设置 max_new（安慰类调大），并对空回复/过短回复自动重试一次（兜底 persona）
  - [x] 4.5 启动一体化对话台，验证 talk 与 stream 端到端正常

- [x] Task 5: 前端"对方正在发送语音"指示（webapp）
  - [x] 5.1 app.js：文本就绪（text_done）后、进入 TTS 合成阶段时，把状态指示从"对方正在输入…"切换为"对方正在发送语音…"
  - [x] 5.2 音频就绪/`want_tts=false` 时恢复/不显示语音指示；index.html 若无对应文案则补充
  - [x] 5.3 浏览器联调：开启语音回复时可见"对方正在发送语音…"，关闭时仅"对方正在输入…"

- [x] Task 6: 端到端验证（主播化行为验收）
  - [x] 6.1 安慰场景（"我好难过，陪我说说话吧"）：回复明显变长（小作文），情感向量生效
  - [x] 6.2 日常寒暄（"你好呀，今天过得怎么样"）：回复中等长度、口语化，句句有回应
  - [x] 6.3 空回复/超短回复兜底重试生效；TTS 阶段前端显示"对方正在发送语音…"
  - [x] 6.4 主模型权重未变（仍 llama.cpp GGUF），角色包可热切换，情感向量路由开关关闭时与原始行为一致

# Task Dependencies
- Task 2 依赖 Task 1（情感向量表是引擎层输入）
- Task 3 依赖 Task 2（服务层调用引擎新接口）
- Task 4 依赖 Task 3（集成层调用服务层新参数）
- Task 5 可与 Task 4 并行（前端独立于后端逻辑）
- Task 6 依赖 Task 1–5 全部完成
