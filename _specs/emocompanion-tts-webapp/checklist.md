# Checklist

- [x] `serve/tts_engine.py` 提供惰性加载单例与 `synthesize(text, emotion)`，缺模型返回结构化异常
- [x] `serve/unified_server.py` 暴露 `GET /api/tts/models`、`POST /api/tts/synthesize`（audio/wav）、`GET /api/tts/health`、`GET /`
- [x] 文本代理 `/api/text/chat` 转发到 04 引擎（base_url 可配置）且不透传失败
- [x] `serve/web/` 单页含「文本生成 / 语音合成」双 tab、模型/情感/参数选择控件
- [x] 语音 tab 可拉取模型列表、合成并内嵌播放、显示耗时/RTF；缺模型时给出清晰错误
- [x] `serve/launcher.py` + bat 一键启动（venv + 清华镜像依赖 + uvicorn），默认 http://127.0.0.1:8070
- [x] 新建 .py 全部 `py_compile` 通过
- [x] `--skip-tts` 启动：`/` 与 `/api/tts/health`、`/api/tts/models` 可用，TTS 缺组件返回结构化错误且服务不崩溃