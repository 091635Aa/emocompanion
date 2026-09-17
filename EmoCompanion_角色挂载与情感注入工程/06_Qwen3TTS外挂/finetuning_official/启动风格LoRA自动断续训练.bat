@echo off
chcp 65001 >nul
title 风格LoRA自动断续训练
echo ============================================
echo   EmoCompanion · 说话风格 LoRA 自动断续训练
echo   逻辑: 检测GPU空闲->训练->自动转GGUF
echo   全程不动 RVC / 一体化对话台 / 文本引擎
echo   GPU忙时自动等待, 空闲才断续推进
echo ============================================
cd /d "D:\AI情感\EmoCompanion_角色挂载与情感注入工程\06_Qwen3TTS外挂\finetuning_official"
"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe" style_lora_autodrive.py
pause