# Tasks

- [x] Task 1: TTS 引擎封装 `serve/tts_engine.py`
  - [x] 1.1 常量：Base 路径（modelscope 缓存）、voice/emotion adapter 目录、`target_speaker_embedding.pt`
  - [x] 1.2 惰性加载单例：Base(bf16/cuda) + PeftModel 挂双 adapter + 读角色音色 embedding；缺失时抛结构化异常
  - [x] 1.3 `synthesize(text, emotion) -> (numpy_float32_wav, sr)`：情感前缀组合文本，经 Qwen3-TTS 生成接口产出 24kHz 音频；记录耗时/RTF
- [x] Task 2: 统一服务 `serve/unified_server.py`
  - [x] 2.1 挂载原生 TTS 接口：`GET /api/tts/models`（列 adapter/情感）、`GET /api/tts/health`、`POST /api/tts/synthesize`（返回 audio/wav）
  - [x] 2.2 文本代理：`POST /api/text/chat` 转发到 04 文本引擎（configurable base_url，默认 http://127.0.0.1:8000），不透传失败
  - [x] 2.3 CORS 全开 + 静态托管 `serve/web/`，`/` 返回前端页
- [x] Task 3: Web 前端 `serve/web/`
  - [x] 3.1 index.html + style.css：双 tab 布局（文本生成 / 语音合成）
  - [x] 3.2 app.js：语音 tab 拉 `/api/tts/models` 填下拉（音色 adapter + 情感标签），文本输入、参数控件、提交合成并内嵌 `<audio>` 播放、显示耗时/RTF
  - [x] 3.3 app.js：文本 tab 组装结构化请求 → `/api/text/chat`；两栏展示
- [x] Task 4: 一键启动 `serve/launcher.py` + bat
  - [x] 4.1 自动建/复用 venv、清华镜像装依赖（fastapi/uvicorn/psutil/numpy）、设 env、起 uvicorn(8070)
  - [x] 4.2 `启动前端后端一体化.bat` 调用 launcher，支持 `--open` 开浏览器
- [x] Task 5: 语法/启动验证
  - [x] 5.1 `python -m py_compile` 全部新建 .py 通过
  - [x] 5.2 以 `--skip-tts`（不加载真模型）启动服务，确认 `/` 与 `/api/tts/health`、`/api/tts/models` 可用、TTS 缺组件返回结构化错误而非崩溃

# Task Dependencies
- Task 2 依赖 Task 1（server 引用 tts_engine）
- Task 3 依赖 Task 2（前端调用统一接口）
- Task 4 依赖 Task 2（启动统一服务）
- Task 5 依赖 Task 1-4（对整体做验证）