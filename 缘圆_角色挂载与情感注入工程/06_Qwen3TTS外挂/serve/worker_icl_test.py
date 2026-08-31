# -*- coding: utf-8 -*-
"""常驻 tf worker + 多段 ICL 实测脚本"""
import json
import os
import subprocess
import sys
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))


def read_wav(p):
    with wave.open(p, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    return sr, n / float(sr)


def main():
    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "tts_tf_worker.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1)
    print(f"worker pid={proc.pid}", flush=True)

    t0 = time.time()
    ready = False
    for line in proc.stdout:
        line = line.strip()
        print(f"[worker] {line}", flush=True)
        if line:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("event") == "ready":
                ready = True
                break
        if time.time() - t0 > 300:
            break
    if not ready:
        print("FAIL: worker not ready", flush=True)
        proc.kill()
        sys.exit(1)
    print(f"[ok] ready after {time.time()-t0:.1f}s", flush=True)

    refs_dir = os.path.join(HERE, "refs")

    def seg(emo):
        wav = os.path.join(refs_dir, f"{emo}.wav")
        txt = os.path.join(refs_dir, f"{emo}.txt")
        text = None
        if os.path.isfile(txt):
            raw = open(txt, encoding="utf-8").read().strip()
            text = f"[{emo}]{raw}"
        return {"audio": wav, "text": text, "emotion": emo}

    out_dir = os.path.join(HERE, "..", "out", "worker_test")
    os.makedirs(out_dir, exist_ok=True)

    # 请求1：多段 ICL（平静锚点 + 开心目标）
    req1 = {"id": "t1", "text": "嘿嘿~你终于来啦，我等你好久啦",
            "emotion": "开心", "adapter": "voice", "tone": 0.35,
            "refs": [seg("平静"), seg("开心")], "out_dir": out_dir, "prefix": "t1"}
    proc.stdin.write(json.dumps(req1, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    t1 = time.time()
    resp1 = None
    while time.time() - t1 < 600:
        line = proc.stdout.readline().strip()
        if not line:
            continue
        if not line.startswith("{"):   # 跳过模型/transformers 的 stdout 打印
            continue
        print(f"[resp1] {line[:200]}", flush=True)
        resp1 = json.loads(line)
        break
    if not resp1 or not resp1.get("ok"):
        print("FAIL req1:", resp1, flush=True)
        proc.kill()
        sys.exit(1)
    sr, dur = read_wav(resp1["wav"])
    meta = resp1.get("meta", {})
    print(f"[ok] t1 多段ICL 合成成功 dur={dur:.2f}s "
          f"seconds={meta.get('seconds')}s n_refs={meta.get('n_refs')} "
          f"strategy={meta.get('strategy')}", flush=True)

    # 请求2：悲伤（验证复用 + 单段情感 ICL）
    req2 = {"id": "t2", "text": "今天有点想你了，你要照顾好自己",
            "emotion": "悲伤", "adapter": "voice", "tone": 0.35,
            "refs": [seg("平静"), seg("悲伤")], "out_dir": out_dir, "prefix": "t2"}
    proc.stdin.write(json.dumps(req2, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    t2 = time.time()
    resp2 = None
    while time.time() - t2 < 300:
        line = proc.stdout.readline().strip()
        if not line:
            continue
        if not line.startswith("{"):
            continue
        print(f"[resp2] {line[:200]}", flush=True)
        resp2 = json.loads(line)
        break
    if not resp2 or not resp2.get("ok"):
        print("FAIL req2:", resp2, flush=True)
        proc.kill()
        sys.exit(1)
    meta2 = resp2.get("meta", {})
    print(f"[ok] t2 复用后合成 seconds={meta2.get('seconds')}s "
          f"n_refs={meta2.get('n_refs')} strategy={meta2.get('strategy')}", flush=True)

    proc.kill()
    print("[done] 实测通过", flush=True)


if __name__ == "__main__":
    main()
