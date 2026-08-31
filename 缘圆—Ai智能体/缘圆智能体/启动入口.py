# -*- coding: utf-8 -*-
"""缘圆智能体 V2 启动入口。

启动：   python 启动入口.py                    # 默认 127.0.0.1:8000，以应用内置浏览器窗口打开
        python 启动入口.py --浏览器 系统       # 改用系统默认浏览器打开
        python 启动入口.py --host 0.0.0.0 --port 9000
查询模型：python 启动入口.py --list-models
查询音色：python 启动入口.py --list-voices

内置浏览器：使用 pywebview（Windows 上封装系统自带的 Edge WebView2）弹出独立应用窗口，
不占用系统默认浏览器。关键点：WebView2 默认会拒绝未处理的媒体权限请求，而 pywebview
不处理 PermissionRequested 事件，因此这里手动挂接授权处理器，自动允许麦克风 / 摄像头 /
屏幕捕获，保证实时通话、视频通话、视频录制功能可用。
"""
import argparse
import threading
import time

import 环境配置  # noqa: F401  模块加载即完成密钥加载与数据目录初始化
from 核心模块 import 配置持久化, 语音合成
from 核心模块.后端服务 import 应用


def 打印模型清单():
    print("=" * 78)
    print(f"{'模型ID':<28}{'别名':<20}{'标签':<6}{'指令':<6}说明")
    print("-" * 78)
    for 模型 in 语音合成.支持的模型列表:
        别名 = 配置持久化.模型别名(模型["id"])
        print(f"{模型['id']:<28}{(别名 if 别名 != 模型['id'] else '-'):<20}"
              f"{'支持' if 模型['tags'] else '—':<6}{'支持' if 模型['instruction'] else '—':<6}{模型['note']}")
    print("=" * 78)


def 打印音色清单():
    from 核心模块 import 音色管理
    音色们 = 音色管理.全部音色()
    print("=" * 78)
    print(f"{'来源':<14}{'音色ID':<46}名称")
    print("-" * 78)
    for 音色 in 音色们:
        print(f"{音色.get('kind', '?'):<14}{音色['id']:<46}{音色.get('name', '')}")
    print("=" * 78)


def 等待服务就绪(地址, 超时秒=30):
    """轮询等待 Web 服务可访问，返回是否就绪。

    注意：urllib.request 不会自动对 URL 中的中文路径做百分号编码（本应用路由几乎
    全是中文），因此这里用唯一的纯 ASCII 探活路径 —— 根路径 /，避免 UnicodeEncodeError。
    """
    import urllib.request
    截止 = time.time() + 超时秒
    while time.time() < 截止:
        try:
            with urllib.request.urlopen(地址 + "/", timeout=1.5) as 响应:
                if 响应.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def 启动服务(主机, 端口):
    """在后台守护线程运行 uvicorn，随主进程退出而终止。"""
    import uvicorn
    uvicorn.run(应用, host=主机, port=端口, log_level="info")


def 运行内置浏览器(主机, 端口):
    """用 pywebview（Edge WebView2）打开应用窗口，并自动授权媒体权限。"""
    import webview
    from 环境配置 import 数据目录
    缓存目录 = 数据目录() / "内置浏览器缓存"
    try:
        缓存目录.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    窗口 = webview.create_window(
        "缘圆智能体", f"http://{主机}:{端口}",
        width=1180, height=820, min_size=(980, 680),
        background_color="#0f1117",
    )

    def 授权媒体权限():
        """窗口就绪后：定位 WebView2 控件并自动授权媒体权限。

        WebView2 默认会拒绝未处理的媒体权限请求，而 pywebview 不处理
        PermissionRequested 事件，因此这里手动挂接授权处理器。
        注意：CoreWebView2 必须在其所属的 UI 线程上访问（跨线程会抛异常），
        所以用 控件.Invoke 调度到 UI 线程执行（与 pywebview 内部 evaluate_js 同机制）。
        """
        try:
            from Microsoft.Web.WebView2.Core import (
                CoreWebView2PermissionKind, CoreWebView2PermissionState,
                CoreWebView2PermissionRequestedEventArgs)
            from System import Action, EventHandler
        except Exception as 异常:
            print("[内置浏览器] 无法导入 WebView2 权限类型：", 异常, flush=True)
            return
        # 等待 native（WinForms BrowserView）就绪，最多 30 秒
        for _ in range(150):
            if getattr(窗口, "native", None) is not None:
                break
            time.sleep(0.2)
        控件 = None
        # pywebview 6.x (winforms)：窗口.native = BrowserView，其 .browser = EdgeChrome，
        # EdgeChrome.webview = WebView2 控件（含 CoreWebView2）。
        for 候选 in (getattr(窗口, "native", None), getattr(窗口, "gui", None)):
            if 候选 is None:
                continue
            for 对象 in (getattr(候选, "browser", None), 候选):
                if 对象 is not None and getattr(对象, "webview", None) is not None:
                    控件 = 对象.webview
                    break
            if 控件 is not None:
                break
        if 控件 is None:
            print("[内置浏览器] 未取得 WebView2 控件，媒体权限需在页面内确认", flush=True)
            return

        def 处理权限(发送者, 参数):
            try:
                允许类型 = [CoreWebView2PermissionKind.Microphone,
                            CoreWebView2PermissionKind.Camera]
                if hasattr(CoreWebView2PermissionKind, "ScreenCapture"):
                    允许类型.append(CoreWebView2PermissionKind.ScreenCapture)
                if 参数.PermissionKind in 允许类型:
                    参数.State = CoreWebView2PermissionState.Allow
            except Exception as 异常:
                print("[内置浏览器] 权限授权异常：", 异常, flush=True)

        最后异常 = None
        for _ in range(50):  # 最多约 25 秒等待 WebView2 初始化完成
            结果 = {}

            def 尝试在UI线程():
                try:
                    if getattr(控件, "CoreWebView2", None) is None:
                        结果["未就绪"] = True
                        return
                    控件.CoreWebView2.add_PermissionRequested(
                        EventHandler[CoreWebView2PermissionRequestedEventArgs](处理权限))
                    结果["已挂"] = True
                except Exception as 异常:
                    结果["异常"] = 异常

            try:
                控件.Invoke(Action(尝试在UI线程))
            except Exception as 异常:
                结果["异常"] = 异常
            if 结果.get("已挂"):
                print("[内置浏览器] 已启用自动授权：麦克风 / 摄像头 / 屏幕捕获", flush=True)
                return
            最后异常 = 结果.get("异常")
            time.sleep(0.5)
        print("[内置浏览器] 挂接权限处理器失败：", 最后异常 or "CoreWebView2 未就绪", flush=True)

    webview.start(授权媒体权限, storage_path=str(缓存目录), private_mode=False)
    return True


def 运行系统浏览器(地址):
    import webbrowser
    print("以系统默认浏览器打开：", 地址)
    webbrowser.open(地址)


def 主函数():
    解析器 = argparse.ArgumentParser(description="缘圆智能体 V2")
    解析器.add_argument("--list-models", action="store_true", help="查询支持的模型")
    解析器.add_argument("--list-voices", action="store_true", help="查询音色")
    解析器.add_argument("--host", default="127.0.0.1", help="监听地址")
    解析器.add_argument("--port", type=int, default=8000, help="监听端口")
    解析器.add_argument("--浏览器", choices=("内置", "系统"), default="内置",
                        help="打开方式：内置 = 应用内置浏览器窗口（默认）；系统 = 系统默认浏览器")
    参数 = 解析器.parse_args()

    if 参数.list_models:
        打印模型清单()
        return
    if 参数.list_voices:
        打印音色清单()
        return

    地址 = f"http://{参数.host}:{参数.port}"
    print(f"缘圆智能体 V2 已启动： {地址}", flush=True)
    print(f"查询支持的模型： python {__file__} --list-models", flush=True)

    服务线程 = threading.Thread(target=启动服务, args=(参数.host, 参数.port), daemon=True)
    服务线程.start()
    if not 等待服务就绪(地址):
        print("Web 服务启动超时，请检查端口是否被占用：", 参数.port, flush=True)
        return

    if 参数.浏览器 == "内置":
        try:
            运行内置浏览器(参数.host, 参数.port)
            print("应用窗口已关闭，服务退出。", flush=True)
            return
        except Exception as 异常:
            print("[内置浏览器] 启动失败，回退到系统默认浏览器。原因：", 异常, flush=True)

    运行系统浏览器(地址)
    # 系统浏览器模式（或内置模式回退）：进程常驻，等待用户退出
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("已退出。")


if __name__ == "__main__":
    主函数()
