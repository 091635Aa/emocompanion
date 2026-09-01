# -*- coding: utf-8 -*-
"""_regression/aggregate_runner —— 离线回归集聚合 runner

一键运行所有 test_*.py，聚合通过/失败断言与汇总，输出单条 JSON/文本报告。
用于 R7 之后每一轮的快速健康回归：任何改动机器人前先 `python aggregate_runner.py`。
"""
import importlib.util
import json
import os
import re
import sys
import time

# 结果文件与报告集中落在 _regression/结果/ 下
结果目录 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "结果")
os.makedirs(结果目录, exist_ok=True)

_PASS_RE = re.compile(r"\[PASS\](.*)")
_FAIL_RE = re.compile(r"\[FAIL\](.*)")
_SUM_RE = re.compile(r"==\s*结果:\s*(\d+)\s*通过\s*/\s*(\d+)\s*失败\s*==")
_SUM_RE2 = re.compile(r"(\d+)\s*通过\s*/\s*(\d+)\s*失败")


def 运行单文件(path: str):
    """在独立解释器中跑一个 test_*.py，捕获 stdout 与退出码。"""
    import subprocess
    _t0 = time.time()
    p = subprocess.run([sys.executable, path], capture_output=True, text=True, cwd=os.path.dirname(path))
    _dt = round(time.time() - _t0, 3)
    out = (p.stdout or "") + (p.stderr or "")
    passes = len(_PASS_RE.findall(out))
    fails = len(_FAIL_RE.findall(out))
    # 优先用脚本自带汇总；无则用[PASS]/[FAIL]计数
    m = _SUM_RE.search(out)
    if m:
        passes, fails = int(m.group(1)), int(m.group(2))
    else:
        m2 = _SUM_RE2.search(out)
        if m2:
            passes, fails = int(m2.group(1)), int(m2.group(2))
    return {
        "file": os.path.basename(path),
        "exit_code": p.returncode,
        "pass": passes, "fail": fails,
        "duration_s": _dt,
        "error": None if p.returncode == 0 else (out[-800:] if out else "非零退出"),
    }


def 收集文件(here, 额外目录=None):
    """收集 test_*.py：_regression 根 + 每个额外目录（深一层递归）。"""
    目录 = [here] + (额外目录 or [])
    files = []
    seen = set()
    for d in 目录:
        if not os.path.isdir(d):
            continue
        for dp, _, fs in os.walk(os.path.abspath(d)):
            if ".venv" in dp or "__pycache__" in dp or "/结果" in dp:
                continue
            for f in sorted(fs):
                if f.startswith("test_") and f.endswith(".py"):
                    full = os.path.abspath(os.path.join(dp, f))
                    if full not in seen:
                        seen.add(full)
                        files.append(full)
    return sorted(files)


def 主():
    here = os.path.dirname(os.path.abspath(__file__))
    # 额外目录从命令行参数传入
    额外目录 = [a for a in sys.argv[1:] if os.path.isdir(a)]
    files = 收集文件(here, 额外目录)
    if not files:
        print("未发现 test_*.py 回归集")
        sys.exit(1)

    结果 = []
    t0 = time.time()
    耗时阈值 = float(os.environ.get("REGRESSION_TIME_WARN", 4.0))  # 单套件告警秒数
    for f in files:
        r = 运行单文件(f)
        结果.append(r)
        tag = "OK " if r["fail"] == 0 and r["exit_code"] == 0 else "X  "
        warn = f"  [WARN 耗时 {r['duration_s']}s>{耗时阈值}s]" if r["duration_s"] > 耗时阈值 else ""
        print(f"[{tag}] {os.path.relpath(f, here)}: {r['pass']}P/{r['fail']}F ({r['duration_s']}s){warn}" + ("" if not r["error"] else f"  <- {r['error'][:120]}"))

    慢套件 = [r for r in 结果 if r["duration_s"] > 耗时阈值]
    if 慢套件:
        print(f"\n== 耗时告警: {len(慢套件)} 个套件超阈值 {耗时阈值}s ==")
        for r in 慢套件:
            print(f"  {r['file']}: {r['duration_s']}s")

    T通过 = sum(r["pass"] for r in 结果)
    T失败 = sum(r["fail"] for r in 结果)
    T耗时 = round(time.time() - t0, 2)
    印章 = time.strftime("%Y%m%d_%H%M%S")

    print(f"\n== 聚合结果: {T通过} 通过 / {T失败} 失败（{len(结果)} 集, {T耗时}s）==")

    报告 = {
        "generated_at": 印章,
        "total_pass": T通过, "total_fail": T失败, "n_suites": len(结果),
        "duration_s": T耗时,
        "suites": 结果,
    }
    jp = os.path.join(结果目录, f"regression_{印章}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(报告, f, ensure_ascii=False, indent=2)
    print(f"报告落盘: {jp}")
    sys.exit(1 if T失败 else 0)


if __name__ == "__main__":
    主()