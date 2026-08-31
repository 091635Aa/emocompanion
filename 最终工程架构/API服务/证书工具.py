# -*- coding: utf-8 -*-
"""
证书工具 — 生成自签名 HTTPS 证书（cryptography）
=================================================
生成 证书\\server.crt + 证书\\server.key（有效期 3650 天，SAN=localhost/127.0.0.1）
"""
import os

证书目录 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "证书")


def 生成证书(覆盖=False):
    """生成自签名证书，返回 (证书路径, 密钥路径)"""
    os.makedirs(证书目录, exist_ok=True)
    证书路径 = os.path.join(证书目录, "server.crt")
    密钥路径 = os.path.join(证书目录, "server.key")
    if os.path.exists(证书路径) and os.path.exists(密钥路径) and not 覆盖:
        return 证书路径, 密钥路径

    from datetime import datetime, timedelta, timezone
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    密钥 = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    名称 = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1"),
    ])
    now = datetime.now(timezone.utc)
    证书 = (
        x509.CertificateBuilder()
        .subject_name(名称)
        .issuer_name(名称)
        .public_key(密钥.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(
            [x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1")),
             x509.DNSName("localhost")]),
            critical=False)
        .sign(密钥, hashes.SHA256())
    )
    with open(证书路径, "wb") as f:
        f.write(证书.public_bytes(serialization.Encoding.PEM))
    with open(密钥路径, "wb") as f:
        f.write(密钥.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    print(f"[证书] 已生成: {证书路径}")
    return 证书路径, 密钥路径


if __name__ == "__main__":
    证书, 密钥 = 生成证书()
    print(证书, 密钥)
