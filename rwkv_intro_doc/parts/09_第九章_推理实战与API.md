# 第九章 推理实战：命令行、API 与采样参数

这一章我们把"怎么用"讲透：从一次推理的内部流程，到命令行交互、API 调用、以及那些"温度、Top_P"到底怎么调。读完这一章，你能熟练地让 RWKV 干各种活。

## 9.1 一次推理的内部流程

无论是命令行还是 API，一次推理都要走下面这条流水线：

![一次推理的完整数据流](img/fig16_api_flow.png)

1. **Tokenize（分词）**：把文本切成 token（词元/子词），比如"我喜欢吃苹果"切成若干 token；
2. **前向计算（forward）**：把 token 序列喂给 RWKV，逐层更新状态，输出下一个 token 的概率分布；
3. **采样（sample）**：按概率（结合温度、Top_P 等参数）选出一个 token；
4. **Decode（解码）**：把选出的 token 拼回文本，输出；
5. 重复 2–4，直到生成结束标记（EOS）或达到最大长度。

> **RWKV 的关键细节**：多轮对话时，"状态 S"会一直往下传——所以它不需要每次都重新处理整段历史，这也是它"聊得再久也不慢"的秘密。

## 9.2 命令行交互：Ollama 实战

### 9.2.1 基础对话

```bash
ollama run mollysama/rwkv-7-g1:2.9b
```

进入后直接输入即可。常用终端命令：

| 命令 | 作用 |
| --- | --- |
| `/set temperature 0.8` | 调整温度 |
| `/set num_ctx 8192` | 设置上下文长度 |
| `/set think` / `/set nothink` | 打开/关闭思考模式 |
| `/bye` | 退出 |

### 9.2.2 用自定义人设（Modelfile）

想让 RWKV 扮演特定角色（比如"温柔的心理咨询师"），写一个 `Modelfile`：

```dockerfile
FROM mollysama/rwkv-7-g1:2.9b

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

SYSTEM """
你是一位温柔耐心的心理咨询师，名叫小暖。
你擅长倾听，用简短温暖的话语回应来访者。
永远不要说教，先共情，再给建议。
"""
```

然后创建并运行：

```bash
ollama create my-therapist -f Modelfile
ollama run my-therapist
```

这就是后面"套用到我们自己的 AI 陪伴工程"最直接的一步（见第十二章）。

## 9.3 API 调用：OpenAI 兼容接口

### 9.3.1 用 curl 测试

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "mollysama/rwkv-7-g1:2.9b",
  "messages": [
    {"role": "system", "content": "你是友好的中文助手"},
    {"role": "user", "content": "什么是 RWKV？"}
  ],
  "stream": false,
  "options": {"temperature": 0.7, "top_p": 0.9}
}'
```

### 9.3.2 用 Python 的 openai 库调用

因为接口是 OpenAI 兼容的，所以直接用 openai 库即可：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # 本地服务不校验 key
)

resp = client.chat.completions.create(
    model="mollysama/rwkv-7-g1:2.9b",
    messages=[
        {"role": "system", "content": "你是一个温柔的心理陪伴助手。"},
        {"role": "user", "content": "我今天好累，工作太多了……"},
    ],
    temperature=0.8,
    max_tokens=300,
)
print(resp.choices[0].message.content)
```

### 9.3.3 流式输出（Streaming）

大段回答时，用流式输出体验更好（一个字一个字往外蹦）：

```python
stream = client.chat.completions.create(
    model="mollysama/rwkv-7-g1:2.9b",
    messages=[{"role": "user", "content": "讲个睡前故事"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

## 9.4 采样参数详解：怎么让 AI"听话"

这些参数决定生成的"随机性 vs 稳定"，非常重要：

| 参数 | 作用 | 建议 |
| --- | --- | --- |
| `temperature` | 温度：越高越随机、越有创意；越低越确定 | 0.6–0.9 创作；0.1–0.3 事实/代码 |
| `top_p` | 核采样：只从累计概率前 P 的 token 里选 | 0.8–0.95 常用 |
| `top_k` | 只从概率最高的 K 个 token 里选 | 40–80 常用 |
| `repeat_penalty` | 重复惩罚：抑制说车轱辘话 | 1.0–1.3 |
| `max_tokens` | 单次最多生成多少个 token | 按需求设置 |
| `num_ctx` | 上下文窗口长度 | RWKV 可放心开大 |

**调参口诀**：

- 想要**稳定、专业**的回答（客服、代码、事实问答）→ `temperature` 调低（0.2–0.4）；
- 想要**有创意、会聊天**（写故事、闲聊、情感陪伴）→ `temperature` 调高（0.7–1.0）；
- 发现**重复啰嗦** → 调大 `repeat_penalty` 或适当降 `temperature`。

## 9.5 长上下文实战：让 RWKV 读完整本书

RWKV 的长上下文优势，在"让模型读长文档"时体现得最明显。

### 9.5.1 一次性塞入长文本

```python
long_text = open("一本小说.txt", encoding="utf-8").read()  # 比如 3 万字
resp = client.chat.completions.create(
    model="mollysama/rwkv-7-g1:2.9b",
    messages=[
        {"role": "system", "content": "你是一个文学分析助手。"},
        {"role": "user", "content": f"请阅读以下文本并概括主线剧情：\n\n{long_text}"},
    ],
    max_tokens=1000,
)
print(resp.choices[0].message.content)
```

对 Transformer 来说，3 万字可能已经让 KV Cache 爆掉；RWKV 因为状态固定，可以轻松处理。这也是后面 RAG（第十二章）的基础。

### 9.5.2 超长输入的注意点

- 模型对"超长上下文的记忆质量"仍有限制（毕竟是压缩摘要），太长时建议分段总结；
- 实际使用时建议搭配"先分段摘要 → 再全局整合"的策略，效果更好。

## 9.6 用 vLLM 服务生产环境

高并发场景下，用 vLLM 把 RWKV 包装成标准 OpenAI 服务：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model BlinkDL/rwkv7-g1 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
resp = client.chat.completions.create(
    model="BlinkDL/rwkv7-g1",
    messages=[{"role": "user", "content": "你好"}],
)
```

## 9.7 常见调用问题速查

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| 输出乱码/全是特殊字符 | 模型文件与 tokenizer 不匹配 | 确认用了配套的 tokenizer |
| 回复质量低 | 量化档太低（Q4 以下） | 换 Q8_0 / FP16 |
| 速度太慢 | CPU 推理 + 模型太大 | 换更小模型 / GGUF 量化 |
| 上下文太短 | `num_ctx` 没设大 | 调大上下文长度 |
| 多轮对话"失忆" | 每次请求没带状态 | 用流式/状态保存方案（第十章） |

<div class="tip">

**本章小结**：一次推理 = Tokenize → 前向 → 采样 → Decode，循环直到结束。命令行走 Ollama，程序调用走 OpenAI 兼容 API（支持流式）。温度/Top_P 决定"随机性 vs 稳定"，按用途调参。RWKV 的长上下文优势让"读完整本书再回答"变得轻松。

</div>
