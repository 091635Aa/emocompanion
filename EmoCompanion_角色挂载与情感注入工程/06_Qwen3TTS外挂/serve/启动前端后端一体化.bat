@echo off
chcp 65001 >nul
title EmoCompanion · 前端后端一体化
cd /d "%~dp0"
echo 启动 EmoCompanion前端后端一体化...
python launcher.py --open --port 8070
pause