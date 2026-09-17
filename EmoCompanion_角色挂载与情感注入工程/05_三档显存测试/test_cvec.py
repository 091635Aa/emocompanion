# -*- coding: utf-8 -*-
"""P1.7 GGUF Control Vector 线上验证 —— llama-cli --control-vector 有/无对比
前提: 先运行 p17_bank.py + export_cvec.py 生成 cvec_emocompanion_p17.gguf
"""
import os, sys, json, re, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sweep as rs
import p15_sweep as p15

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_CLI = r"d:\AI情感\pykits\llama-cpp-bin\llama-cli.exe"
MODEL = r"d:\AI情感\pykits\models\Qwen3-4B-Q4_K_M.gguf"
CVEC = os.path.join(HERE, "cvec_emocompanion_p17.gguf")

QUESTIONS = p15.build_testset()[:3]

def build_prompt(q):
    return ("<|im_start|>system\n" + p17.PERSONA + "<|im_end|>\n"
            "<|im_start|>user\n" + q + "<|im_end|>\n"
            "<|im_start|>assistant\n")

def run_llama(prompt, use_cvec):
    base = [LLAMA_CLI, "-m", MODEL, "-p", prompt, "-n", "96", "--temp", "0.9",
            "--top-p", "0.9", "--top-k", "50", "--seed", "42", "-ngl", "99", "-c", "2048"]
    if use_cvec:
        base += ["--control-vector", CVEC]
    r = subprocess.run(base + ["--no-cnv", "--no-display-prompt"],
                       capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=900)
    out = (r.stdout or "").strip()
    if r.returncode != 0 or not out:
        r = subprocess.run(base, capture_output=True, text=True,
                           encoding="utf-8", errors="ignore", timeout=900)
        out = (r.stdout or "").strip()
        if out.startswith(prompt):
            out = out[len(prompt):]
    return out.strip()

def main():
    results = {}
    for mode, use_cvec in (("plain", False), ("cvec", True)):
        texts = []
        for q in QUESTIONS:
            t = run_llama(build_prompt(q), use_cvec)
            texts.append({"u": q, "text": t[:300], "tone_hit": sum(t.count(w) for w in rs.TONE_WORDS)})
            print(f"[{mode}] {q[:16]} → {t[:60]}... (词命中{texts[-1]['tone_hit']})", flush=True)
        results[mode] = texts
    print("[judge] load Qwen3-8B", flush=True)
    jm, jt = p15.load_judge()
    for mode in results:
        fids = []
        for it in results[mode]:
            out = p15.ask(jm, jt, p15.JUDGE_FID.format(u=it["u"], t=it["text"]))
            m = re.search(r"\d{1,3}", out)
            fids.append(min(100, int(m.group())) if m else None)
        valid = [x for x in fids if x is not None]
        results[mode + "_fidelity"] = round(float(sum(valid) / len(valid)), 1) if valid else None
        print(f"  {mode} 保真 {results[mode + '_fidelity']}", flush=True)
    json.dump(results, open(os.path.join(HERE, "p17_cvec_test.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved: p17_cvec_test.json")

if __name__ == "__main__":
    main()
