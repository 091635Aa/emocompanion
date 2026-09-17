@echo off
chcp 65001 >nul
title EmoCompanion · 一体化对话台
cd /d "%~dp0"
echo ============================================
echo   EmoCompanion 一体化对话台 (文本生成 + 本地TTS)
echo   端口 8071，文本引擎默认 http://127.0.0.1:8000
echo   请先启动 04 文本引擎: 04_源码与原型\backend\启动EmoCompanion引擎.bat
echo   使用系统 Python（含 torch/qwen_tts，支持 tf 情感外挂后端）
echo ============================================
"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe" integrated_app.py --host 0.0.0.0 --port 8071 --text-base http://127.0.0.1:8000
pause
