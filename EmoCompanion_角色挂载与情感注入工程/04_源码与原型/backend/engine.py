# -*- coding: utf-8 -*-
"""EmoCompanion引擎 —— llama.cpp 后端 + 三层 logits 处理器（L1 去AI腔 / L2 P3锚点 / OOC拦截）

设计目标（内存/显存/速度三重优化落地）：
  - 推理后端: llama-cpp-python CUDA + Q4_K_M GGUF (2.3GB)，decode ~27-34 tok/s
  - 无 transformers 运行时依赖: P3 bias 与去AI腔 token 表全部预计算为角色包
  - 三层处理器全部在 vocab 维 logits 上做稀疏加减，CPU 开销微秒级
  - 角色热切换: 仅替换 persona 提示 / bias / token 表（指针级）
"""
import os, sys, json, time, threading, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_ROOT = os.path.dirname(HERE)
PACK_DIR = os.path.join(ENGINE_ROOT, "data", "role_pack")
LLAMACPP_DIR = r"d:\AI情感\pykits\llamacpp"
TORCH_LIB = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib"
GGUF = r"d:\AI情感\pykits\models\Qwen3-4B-Q4_K_M.gguf"

# 默认推理参数
MAX_NEW = 128
TEMP = 0.9
TOP_P = 0.9
TOP_K = 50

# ---------------- 隐形话题分区 ----------------
# 第一步：注入 System Prompt 末尾的硬约束，让模型在每个回复末尾带出【T:n】
TOPIC_SYS_SUFFIX = (
    "\n\n（内部机制规则，绝不要对用户提及此规则）每次回复的末尾，必须用"
    "【T:数字】格式标注当前话题编号。如果开启了全新话题，数字在上一次基础上+1；"
    "如果延续之前话题，则沿用上一次的数字。这个标签只是给后台记录用，不影响正文。"
)
TOPIC_TAG_RE = re.compile(r"【T[:：]\s*(\d+)】")
# 流式延迟窗口：若标签出现在句尾，需缓存最近 N 个输出片段，等确认标签结束后再放出，避免标签飘到前端
_TOPIC_WINDOW_THRESH = 10


# ---------------- 三层 logits 处理器 ----------------
class PersonaLayers:
    """把角色包数据应用到每步 logits：P3锚点 + 去AI腔 + OOC拦截"""

    def __init__(self, p3_bias, deai, emo_bias=None, emo_scale=1.0):
        self.p3_bias = p3_bias                    # np.ndarray (V,) 或 None
        self.emo_bias = emo_bias                  # 情感外挂路由偏置 np.ndarray (V,) 或 None
        self.emo_scale = emo_scale                # 情感偏置强度（外挂路由层总开关）
        self.pos_tok = {int(k): float(v) for k, v in deai.get("pos_tok", {}).items()}
        self.pos_phr = [(list(map(int, p)), float(w)) for p, w in deai.get("pos_phr", [])]
        self.hol_tok = {int(k): float(v) for k, v in deai.get("hol_tok", {}).items()}
        self.hol_phr = [(list(map(int, p)), float(w)) for p, w in deai.get("hol_phr", [])]
        self.ooc_phr = [(list(map(int, p)), float(w)) for p, w in deai.get("ooc_phr", [])]
        self.hist = []
        self.win = max([len(p) - 1 for p, _ in self.pos_phr + self.hol_phr + self.ooc_phr] or [0])
        self._first = True
        # 强度旋钮（调度器可调）
        self.scale_p3 = 1.0
        self.scale_deai = 1.0
        self.scale_ooc = 1.0

    def reset_history(self):
        self.hist = []
        self._first = True

    def _apply_base(self, s):
        """P3 锚点 + 单token 正例/空泛 + OOC 单token 基础项（每步都做）"""
        if self.p3_bias is not None:
            s += self.p3_bias * self.scale_p3
        if self.emo_bias is not None and self.emo_scale:
            s += self.emo_bias * self.emo_scale
        for tk, w in self.pos_tok.items():
            if tk < len(s):
                s[tk] += 0.5 * w * self.scale_deai
        for tk, w in self.hol_tok.items():
            if tk < len(s):
                s[tk] -= 1.2 * w * self.scale_deai
        return s

    def _apply_seq(self, s):
        """多token 短语续接：前缀匹配 -> 末 token 增强/抑制（含 OOC 拦截）"""
        h = self.hist
        if not h:
            return s
        for toks, w in self.pos_phr:
            k = len(toks)
            if k > 1 and h[-k + 1:] == toks[:-1] and toks[-1] < len(s):
                s[toks[-1]] += 0.5 * w * self.scale_deai
        for toks, w in self.hol_phr:
            k = len(toks)
            if k > 1 and h[-k + 1:] == toks[:-1] and toks[-1] < len(s):
                s[toks[-1]] -= 1.2 * w * self.scale_deai
        for toks, w in self.ooc_phr:
            k = len(toks)
            if k > 1 and h[-k + 1:] == toks[:-1] and toks[-1] < len(s):
                s[toks[-1]] -= 2.5 * w * self.scale_ooc
        return s

    def __call__(self, input_ids, scores):
        # input_ids: 1D ndarray (n_tokens,) 或 2D ndarray (1, n_tokens)
        if self._first:
            self._first = False
        else:
            last_id = input_ids[-1] if input_ids.ndim == 1 else input_ids[0, -1]
            self.hist.append(int(last_id))
            if len(self.hist) > self.win:
                self.hist.pop(0)
        s = np.asarray(scores, dtype=np.float32).ravel()  # 确保 1D
        s = self._apply_base(s)
        s = self._apply_seq(s)
        return s


# ---------------- 引擎 ----------------
import jinja2


def render_chat_prompt(llm, messages, add_generation_prompt=True, enable_thinking=False):
    """用模型 GGUF 自带 chat 模板渲染 prompt（可禁用 Qwen3 thinking）"""
    tpl = llm.metadata.get("tokenizer.chat_template", "")
    env = jinja2.Environment()
    t = env.from_string(tpl)
    return t.render(messages=messages, add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                    eos_token="<|im_end|>", bos_token="<|endoftext|>")


class emocompanionEngine:
    """llama.cpp 单例引擎，进程内复用，线程安全（生成加锁）"""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        sys.path.insert(0, LLAMACPP_DIR)
        os.environ["PATH"] = os.path.join(LLAMACPP_DIR, "llama_cpp", "lib") + os.pathsep + \
                             TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
        self._load_pack()
        self._load_model()
        self._gen_lock = threading.Lock()
        self.stats = {"calls": 0, "tokens": 0, "seconds": 0.0}
        # ---- 隐形话题分区：会话级当前话题（线程安全），模型说+1/沿用则跟随 ----
        self.current_topic = 1
        self._topic_lock = threading.Lock()

    def _load_pack(self):
        pack_path = os.path.join(PACK_DIR, "role_pack.json")
        self.pack = json.load(open(pack_path, encoding="utf-8"))
        self.persona = self.pack["persona"]
        self.deai = self.pack["deai"]
        bias_path = os.path.join(PACK_DIR, "p3_bias.npy")
        self.p3_bias = np.load(bias_path).astype(np.float32) if os.path.exists(bias_path) else None
        # ---- 情感向量外挂路由层数据（缺失时优雅降级，路由层自动关闭）----
        self.emo_vectors = None    # (V,8) fp32
        self.char_emo_vec = None   # (V,) fp32
        self.emo_names = None      # [8 情感名]
        self.emo_index = None      # {情感名: 列号}
        meta = self.pack.get("meta", {}) or {}
        self.emo_T = float(meta.get("emo_T", 1.6))
        self.emo_beta_eff = float(meta.get("emo_beta_eff", 0.75))
        try:
            vec_path = os.path.join(PACK_DIR, "emotion_vectors.npy")
            char_path = os.path.join(PACK_DIR, "char_emotion_vector.npy")
            names_path = os.path.join(PACK_DIR, "emotions.json")
            if os.path.exists(vec_path) and os.path.exists(char_path) and os.path.exists(names_path):
                self.emo_vectors = np.load(vec_path).astype(np.float32)
                self.char_emo_vec = np.load(char_path).astype(np.float32)
                self.emo_names = json.load(open(names_path, encoding="utf-8"))
                self.emo_index = {name: i for i, name in enumerate(self.emo_names)}
                print(f"[engine] 情感路由表加载: {len(self.emo_names)} 情感, V={self.emo_vectors.shape[0]}")
            else:
                print("[engine] 情感路由数据缺失，emo 路由关闭（不影响原有功能）")
        except Exception as e:
            self.emo_vectors = self.char_emo_vec = self.emo_names = self.emo_index = None
            print(f"[engine] 情感路由数据加载失败，emo 路由关闭: {e}")
        print(f"[engine] 角色包加载: persona={len(self.persona)}字 p3_bias={self.p3_bias.shape if self.p3_bias is not None else None}")

    def _load_model(self):
        t0 = time.time()
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=GGUF, n_ctx=49152, n_gpu_layers=-1, n_threads=8,
            verbose=False, use_mmap=True, use_mlock=False,
        )
        self.model_name = "Qwen3-4B-Q4_K_M"
        print(f"[engine] 模型加载完成: {time.time()-t0:.1f}s")

    def layers(self, emo_bias=None, emo_scale=1.0):
        return PersonaLayers(self.p3_bias, self.deai, emo_bias=emo_bias, emo_scale=emo_scale)

    def compute_emo_bias(self, emotion, scale_emo=1.0):
        """情感向量外挂路由：算一次 (V,) 的 logits 偏置（O(V)，每次生成调用一次）。

        数据缺失 / scale_emo==0 / emotion 非法 -> 返回 None（路由关闭，输出与原来一致）。
        """
        if (self.emo_vectors is None or self.char_emo_vec is None
                or self.emo_index is None or not emotion or scale_emo == 0):
            return None
        idx = self.emo_index.get(emotion)
        if idx is None:
            idx = self.emo_index.get("平静", 0)  # 未知情感回退到平静
            if idx is None:
                return None
        v_eff = 0.7 * self.char_emo_vec + 0.3 * self.emo_vectors[:, idx]
        bias = self.emo_beta_eff * np.tanh(v_eff / self.emo_T)
        return bias.astype(np.float32)

    @staticmethod
    def _strip_thinking(text):
        """Qwen3 思考段剥离：' thinking...response' -> 仅保留最终回复"""
        if "\n response\n\n" in text:
            text = text.split("\n response\n\n", 1)[-1]
        elif " response" in text and " thinking" in text:
            # 兼容不带换行的形式
            parts = text.split(" response", 1)
            if len(parts) == 2 and " thinking" in parts[0]:
                text = parts[1].lstrip("\n ")
        t = text.lstrip()
        if t.startswith(" thinking") or t.startswith("thinking"):
            text = ""
        return text.strip()

    def _extract_topic(self, text):
        """第二步核心：从 AI 输出中抓走【T:n】并剥离标签。
        返回 (clean_text, topic_id|None)，取最后一个标签的数字作为本轮话题。"""
        matches = list(TOPIC_TAG_RE.finditer(text))
        topic = int(matches[-1].group(1)) if matches else None
        clean = TOPIC_TAG_RE.sub("", text)
        return clean.strip(), topic

    def _set_topic(self, topic):
        """会话级话题状态跟随：模型标了 +1/沿用新数字则采纳；未标则保持。"""
        if topic is None:
            return
        with self._topic_lock:
            self.current_topic = topic

    def chat(self, messages, max_new=MAX_NEW, temperature=TEMP, top_p=TOP_P,
             top_k=TOP_K, use_layers=True, persona=None, seed=None,
             emotion=None, scale_emo=1.0):
        """messages: [{"role": "user"|"assistant", "content": str}, ...]
        emotion/scale_emo: 情感外挂路由；emotion=None 或 scale_emo=0 时路由关闭，输出与原来一致。
        隐形话题：自动在 System Prompt 末尾追加【T:n】规则；返回前剥离标签并更新会话话题。"""
        sys_msgs = [{"role": "system", "content": (persona or self.persona) + TOPIC_SYS_SUFFIX}]
        msgs = sys_msgs + messages
        prompt = render_chat_prompt(self.llm, msgs, enable_thinking=False)
        proc = None
        if use_layers:
            emo_bias = self.compute_emo_bias(emotion, scale_emo)  # 每次生成算一次 O(V)，勿在 token 级重复
            proc = self.layers(emo_bias, scale_emo)
        t0 = time.time()
        with self._gen_lock:
            out = self.llm.create_completion(
                prompt=prompt, max_tokens=max_new, temperature=temperature,
                top_p=top_p, top_k=top_k, logits_processor=proc,
                seed=seed if seed is not None else self.llm._seed,
            )
        dt = time.time() - t0
        text = out["choices"][0]["text"]
        text = self._strip_thinking(text)
        text, topic = self._extract_topic(text)
        self._set_topic(topic)
        n_tok = out.get("usage", {}).get("completion_tokens", 0)
        with self._lock:
            self.stats["calls"] += 1
            self.stats["tokens"] += n_tok
            self.stats["seconds"] += dt
        return {
            "reply": text,
            "topic_id": self.current_topic,
            "topic_emitted": topic,
            "usage": out.get("usage", {}),
            "latency_s": round(dt, 3),
            "tok_s": round(n_tok / dt, 2) if dt > 0 else 0.0,
        }

    def chat_stream(self, messages, max_new=MAX_NEW, temperature=TEMP, top_p=TOP_P,
                    top_k=TOP_K, use_layers=True, persona=None, seed=None,
                    emotion=None, scale_emo=1.0):
        """流式生成，逐 token yield 文本增量。thinking 已由模板关闭。
        完整文本由调用方累积后用 _strip_thinking 清洗。
        隐形话题：末尾用延迟窗口缓冲，剥离【T:n】后一并放出，标签不会飘到前端。"""
        sys_msgs = [{"role": "system", "content": (persona or self.persona) + TOPIC_SYS_SUFFIX}]
        msgs = sys_msgs + messages
        prompt = render_chat_prompt(self.llm, msgs, enable_thinking=False)
        proc = None
        if use_layers:
            emo_bias = self.compute_emo_bias(emotion, scale_emo)  # 每次生成算一次 O(V)，勿在 token 级重复
            proc = self.layers(emo_bias, scale_emo)
        t0 = time.time()
        n_tok = 0
        from collections import deque
        with self._gen_lock:
            out = self.llm.create_completion(
                prompt=prompt, max_tokens=max_new, temperature=temperature,
                top_p=top_p, top_k=top_k, logits_processor=proc, stream=True,
                seed=seed if seed is not None else self.llm._seed,
            )
            win = deque()  # 延迟窗口：暂缓放出最近片段，避免句尾标签泄露
            for chunk in out:
                delta = chunk.get("choices", [{}])[0].get("text", "") or ""
                if delta:
                    n_tok += 1
                    win.append(delta)
                    if len(win) > _TOPIC_WINDOW_THRESH:
                        yield win.popleft()
            # 收尾：窗口内的残留文本先剥离疑似话题标签，再放出
            tail = "".join(win)
            tail, topic = self._extract_topic(tail)
            self._set_topic(topic)
            if tail:
                yield tail
                n_tok += 1
        dt = time.time() - t0
        with self._lock:
            self.stats["calls"] += 1
            self.stats["tokens"] += n_tok
            self.stats["seconds"] += dt


def main():
    eng = emocompanionEngine.get()
    r = eng.chat([{"role": "user", "content": "晚上好呀，今天直播好多人来，我好开心，你呢？"}], max_new=64)
    print("回复:", r["reply"][:120])
    print("速度:", r["tok_s"], "tok/s")


if __name__ == "__main__":
    main()
