@echo off
chcp 65001 >nul
title 缘圆情感引擎 - 一键启动
echo ============================================
echo    缘圆情感引擎  v1.0
echo    角色挂载 + 情感注入 + llama.cpp 加速
echo ============================================
echo.
cd /d "%~dp0"
python app.py --open
pause
