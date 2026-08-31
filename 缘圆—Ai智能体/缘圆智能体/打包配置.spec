# -*- mode: python ; coding: utf-8 -*-
"""缘圆智能体 V2 PyInstaller 打包配置。

构建（在 项目根 下执行）：
    f:\缘圆—Ai智能体\tts_studio\.venv\Scripts\pyinstaller.exe --noconfirm --clean 打包配置.spec

说明：
  - exe 名称使用中文（V1 已验证 PyInstaller 可行）。
  - 前端页面/ 递归打包为只读资源（_MEIPASS/前端页面）。
  - 定制音色/ 仅打包 *.py / *.md（Tree 排除声音库大音频与 __pycache__），
    模块在运行时由 _MEIPASS 下的源码直接导入；声音库数据运行时在 EXE 旁重建。
  - 密钥配置.env 作为数据放入根（_MEIPASS/密钥配置.env），环境配置 会读取。
"""

# 定制音色/ 仅打包 *.py / *.md（递归收集；声音库大音频与 __pycache__ 自然排除），
# 模块运行时由 _MEIPASS 下的源码直接导入；声音库数据运行时在 EXE 旁重建。
from pathlib import Path

定制音色数据 = []
for 文件 in sorted(Path('定制音色').rglob('*')):
    if 文件.is_file() and 文件.suffix.lower() in ('.py', '.md'):
        定制音色数据.append((str(文件), str(文件.parent)))

# 内置浏览器（pywebview + pythonnet/clr_loader）：收集全部数据/DLL/隐藏导入，
# 否则打包后无法创建 WebView2 窗口（WebView2Loader.dll、Python.Runtime.dll 等缺失）。
from PyInstaller.utils.hooks import collect_all  # noqa: E402

内置浏览器数据, 内置浏览器二进制, 内置浏览器隐藏导入 = [], [], []
for 包名 in ('webview', 'pythonnet', 'clr_loader'):
    try:
        数据, 二进制, 隐藏 = collect_all(包名)
        内置浏览器数据 += 数据
        内置浏览器二进制 += 二进制
        内置浏览器隐藏导入 += 隐藏
    except Exception as 异常:
        print(f"[打包配置] 收集 {包名} 失败（可忽略，内置浏览器可能不可用）：{异常}")

a = Analysis(
    ['启动入口.py'],
    pathex=[],
    binaries=[('图标.ico', '.')] + 内置浏览器二进制,
    datas=[('前端页面', '前端页面'), ('密钥配置.env', '.')] + 定制音色数据 + 内置浏览器数据,
    hiddenimports=[
        # uvicorn 动态导入的各子模块（缺一不可，否则打包后无法启动）
        'uvicorn.logging', 'uvicorn.loops', 'uvicorn.protocols',
        'uvicorn.protocols.http', 'uvicorn.protocols.websockets',
        'uvicorn.lifespan', 'uvicorn.lifespan.on',
        # multipart：FastAPI/Starlette 表单解析（/api/上传、/api/定制音色）
        'multipart',
        # websockets：实时通话 /api/通话 桥接
        'websockets',
        # soundfile / scipy：数据集处理与最优片段筛选（24kHz 重采样、VAD、寻峰）
        'soundfile', 'scipy',
        # 内置浏览器：pywebview EdgeChromium 后端 + pythonnet
        'webview', 'webview.platforms', 'webview.platforms.edgechromium',
        'webview.platforms.mshtml',
        'clr', 'pythonnet', 'clr_loader', 'clr_loader.netcore',
    ] + 内置浏览器隐藏导入,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='缘圆智能体',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['图标.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='缘圆智能体',
)
