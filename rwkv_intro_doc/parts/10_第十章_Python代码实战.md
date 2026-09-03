# 第十章 Python 代码实战：加载模型、对话与状态管理

这一章我们用真实可运行的 Python 代码，带你实现一个完整的 RWKV 应用：加载模型 → 对话 → 流式输出 → **状态保存/恢复**（这是 RWKV 独有的杀手锏）。代码尽量简单，注释齐全，跟着敲一遍就能跑。

## 10.1 环境准备

```bash
# 核心依赖
pip install rwkv rwkv_tokenizer torch

# 若用 HuggingFace 生态
pip install transformers huggingface_hub
```

> 提示：`rwkv` 库是官方推理库，支持 .pth 直接加载、State 读写；`transformers` 适合想用统一接口的人。下面以官方 `rwkv` 库为主。

## 10.2 最小可用：加载模型并生成一句话

```python
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import torch

# 1. 指定模型文件（.pth）与策略
model_path = "models/rwkv7-g1-2.9b-20250519-ctx4096.pth"
strategy = "cuda fp16"          # 有 NVIDIA 显卡用这个
# strategy = "cpu fp32"         # 纯 CPU 用这个

model = RWKV(model=model_path, strategy=strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v304")   # 配套 tokenizer

# 2. 生成参数
args = PIPELINE_ARGS(
    temperature=0.8,
    top_p=0.9,
    top_k=80,
    repetition_penalty=1.1,
    max_gen=200,          # 最多生成 200 个 token
)

# 3. 生成一段文本
print(pipeline.generate("请用一句话介绍 RWKV：", token_count=200, args=args))
```

## 10.3 多轮对话：用"状态 S"实现真正的连续记忆

RWKV 最大的魅力在于：**状态 S 可以在多轮之间传递**。这样模型不是靠"把所有历史再喂一遍"，而是带着压缩后的记忆继续聊。

```python
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

model = RWKV(model="models/rwkv7-g1-2.9b.pth", strategy="cuda fp16")
pipeline = PIPELINE(model, "rwkv_vocab_v304")

state = None  # 状态从空开始

def chat(user_msg, state):
    # 组装提示：system + 当前用户消息
    prompt = f"你是一个温柔的心理陪伴助手。\n\n用户：{user_msg}\n助手："
    args = PIPELINE_ARGS(temperature=0.85, top_p=0.95, max_gen=150,
                         repetition_penalty=1.1)
    # generate 会返回 (文本, 更新后的 state)
    text, state = pipeline.generate(prompt, token_count=150, args=args,
                                    state=state)
    return text.strip(), state

# 第一轮
reply1, state = chat("我今天加班到很晚，好累……", state)
print("助手：", reply1)

# 第二轮：模型还记得上一轮的情绪
reply2, state = chat("能不能给我讲个放松的小故事？", state)
print("助手：", reply2)
```

**为什么这是"记忆"而不是"复读"？** 因为 `state` 是压缩后的摘要，不是把原文再抄一遍。这既省内存，又让"越聊越懂你"成为可能。

## 10.4 状态保存与恢复：跨会话续聊（杀手锏）

RWKV 的状态可以**序列化保存到硬盘**，下次启动时直接恢复——相当于"模型记得你是谁、你们聊过什么"。

```python
import pickle

def save_state(state, path="my_chat_state.pkl"):
    # state 通常是 list/tuple 结构，pickle 即可
    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"状态已保存到 {path}")

def load_state(path="my_chat_state.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
```

**完整的多会话流程：**

```python
# 场景：AI 陪伴机器人，每个用户有专属记忆
def get_state_for(user_id):
    try:
        return load_state(f"state_{user_id}.pkl")
    except FileNotFoundError:
        return None  # 新用户，从空白记忆开始

def on_user_message(user_id, msg):
    state = get_state_for(user_id)          # 取出该用户的记忆
    reply, new_state = chat(msg, state)     # 带着记忆对话
    save_state(new_state, f"state_{user_id}.pkl")  # 存回记忆
    return reply
```

> 这一步就是"**角色记忆可持续**"的工程实现——Transformer 要实现类似效果，要么把历史全喂进去（越来越贵），要么自己做一套记忆系统；RWKV 直接给你原生状态，天然适合"陪伴/长期关系"类产品。

## 10.5 流式输出：打字机效果

```python
def chat_stream(user_msg, state):
    prompt = f"用户：{user_msg}\n助手："
    args = PIPELINE_ARGS(temperature=0.85, top_p=0.95,
                         max_gen=150, repetition_penalty=1.1)

    out = ""
    for chunk in pipeline.generate(prompt, token_count=150, args=args,
                                   state=state, stream=True):
        # 每次只返回新增的几个 token
        print(chunk, end="", flush=True)
        out += chunk
    print()
    return out
```

流式输出的价值：长回答时用户不需要干等，体验和 ChatGPT 一样"一个字一个字蹦出来"。

## 10.6 完整版：一个可用的"陪伴聊天机器人"

把上面的拼起来，写一个最小可用的完整脚本：

```python
"""rwkv_chatbot.py —— 一个带长期记忆的最小陪伴机器人"""
import pickle
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

MODEL_PATH = "models/rwkv7-g1-2.9b-20250519-ctx4096.pth"
STRATEGY = "cuda fp16"

model = RWKV(model=MODEL_PATH, strategy=STRATEGY)
pipeline = PIPELINE(model, "rwkv_vocab_v304")
SYSTEM = "你是小暖，一个温柔耐心、善于倾听的心理陪伴助手。"


def make_state(user_id):
    try:
        with open(f"state_{user_id}.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def keep_state(user_id, state):
    with open(f"state_{user_id}.pkl", "wb") as f:
        pickle.dump(state, f)


def respond(user_id, user_msg):
    state = make_state(user_id)
    prompt = f"{SYSTEM}\n\n用户：{user_msg}\n小暖："
    args = PIPELINE_ARGS(temperature=0.85, top_p=0.95,
                         max_gen=200, repetition_penalty=1.1)
    reply, state = pipeline.generate(prompt, token_count=200,
                                     args=args, state=state)
    keep_state(user_id, state)
    return reply.strip()


if __name__ == "__main__":
    uid = "alice"
    while True:
        msg = input("你：")
        if msg in ("exit", "quit"):
            break
        print("小暖：", respond(uid, msg))
```

**这个 40 行的脚本，已经实现了**：

- 多轮对话的连续记忆（状态传递）；
- 跨会话的长期记忆（状态持久化到磁盘）；
- 角色人设（system prompt）；
- 可扩展：加 TTS 语音、情感检测、情绪追踪，就是一个完整的 AI 陪伴产品原型（第十二章展开）。

## 10.7 用 transformers 生态加载（备选）

如果你更习惯 transformers，RWKV 也有对应的集成：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "BlinkDL/rwkv-7-3b"   # 以实际 repo 为准
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).half().cuda().eval()

inputs = tokenizer("你好，介绍一下自己：", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=100, do_sample=True,
                     temperature=0.8, top_p=0.9)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## 10.8 性能与内存观测

| 观测点 | 方法 |
| --- | --- |
| 显存占用 | `nvidia-smi`，观察是否恒定 |
| 生成速度 | 记录生成 token 数 / 耗时 |
| 多轮是否变慢 | 对比第 1 轮与第 20 轮速度 |
| 状态大小 | 打印 `sys.getsizeof` / 查看 .pkl 文件大小 |

RWKV 的特点：**只要模型不变，状态大小不变，显存不变，速度不变**——你可以亲手验证这一点，这也是它最惊艳的地方。

<div class="tip">

**本章小结**：用官方 `rwkv` 库，核心就三步——加载模型、用 pipeline.generate 生成、把返回的 state 传下去/存起来。多轮记忆靠"状态传递"，跨会话记忆靠"状态持久化"，这两招是 Transformer 很难低成本复制的，也是我们做 AI 陪伴产品的根基。

</div>
