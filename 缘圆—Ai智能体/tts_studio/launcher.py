# -*- coding: utf-8 -*-
"""缘圆模块启动器：后台启动本地服务，用内嵌轻量浏览器窗口展示界面（不打开默认浏览器）。

打包：pyinstaller --onefile --windowed --name 缘圆模块 --icon logo.ico launcher.py
"""
import multiprocessing
import socket
import threading
import time


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    multiprocessing.freeze_support()

    import requests
    import uvicorn
    import webview

    import app
    import config

    port = _free_port()

    def run_server():
        uvicorn.run(app.app, host="127.0.0.1", port=port, log_level="warning")

    threading.Thread(target=run_server, daemon=True).start()

    # 等待本地服务就绪
    for _ in range(100):
        try:
            requests.get(f"http://127.0.0.1:{port}/", timeout=1)
            break
        except Exception:
            time.sleep(0.15)

    url = f"http://127.0.0.1:{port}/"
    # 记录本次地址（便于排查，同时避免下次随机端口）
    try:
        (config.app_dir() / "last_url.txt").write_text(url, encoding="utf-8")
    except Exception:
        pass

    try:
        webview.create_window(
            "缘圆模块 · 语音合成工作台",
            url,
            width=1280,
            height=860,
            min_size=(960, 640),
            background_color="#0f1117",
        )
        webview.start()
    except Exception:
        import traceback
        try:
            (config.app_dir() / "error.log").write_text(
                traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
