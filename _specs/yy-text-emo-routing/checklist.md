# Checklist

- [x] build_role_pack.py 产出 `emotion_vectors.npy`（V×8，8 情感）与角色本身情感向量，数据文件可加载且体积 ≤ 数 MB
- [x] engine.py 的 PersonaLayers 情感向量层按 `v_eff = 0.7×角色 + 0.3×优化` 合成并以 β·tanh 限幅注入 logits；scale_emo=0 时与原始行为一致
- [x] engine.chat/chat_stream 支持情感向量/标签参数；chat_stream 同样生效
- [x] server.py /chat 与 /chat/stream 接受 emotion/emo_vector/scale_emo 并正确透传；/debug 显示路由状态
- [x] integrated_app.py 在生成前用 emo_detect 识别情感→查表→传给引擎；识别失败回退"平静"不阻塞
- [x] persona 已去掉"最多 1~3 句"硬限制，含长度自适应指令
- [x] 安慰语境回复明显变长（小作文）；日常语境中等长度、口语化；句句都有非空回应，空/过短自动重试
- [x] 前端在 TTS 合成阶段显示"对方正在发送语音…"，want_tts=false 时仅"对方正在输入…"
- [x] 主模型权重未变（仍 GGUF），角色包可热切换，情感向量路由可开关
- [x] 一体化对话台 talk 与 stream 端到端正常，日志含情感向量路由明细
