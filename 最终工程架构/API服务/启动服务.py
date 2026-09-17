# -*- coding: utf-8 -*-
"""
启动服务 — CLI 入口
====================
用法：
    F:\\打标\\.venv\\Scripts\\python.exe 启动服务.py [--端口 8000] [--host 127.0.0.1] [--https]
    --host 0.0.0.0 允许局域网访问；--https 启用自签名 HTTPS（自动生成证书）
"""
import sys
import os
import argparse

本服务目录 = os.path.dirname(os.path.abspath(__file__))
if 本服务目录 not in sys.path:
    sys.path.insert(0, 本服务目录)


def 主流程():
    parser = argparse.ArgumentParser(description="语义回响 API 服务")
    parser.add_argument("--端口", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--https", action="store_true", help="启用 HTTPS（自动生成自签名证书）")
    args = parser.parse_args()

    ssl参数 = {}
    if args.https:
        import 证书工具
        证书路径, 密钥路径 = 证书工具.生成证书()
        ssl参数 = {"ssl_keyfile": 密钥路径, "ssl_certfile": 证书路径}
        协议 = "https"
    else:
        协议 = "http"

    import uvicorn
    print(f"[启动] {协议}://{args.host}:{args.端口}  |  简化版: /  专业版: /pro  文档: /docs")
    uvicorn.run(
        "主程序:app", host=args.host, port=args.端口,
        log_level="info", **ssl参数)


if __name__ == "__main__":
    主流程()
