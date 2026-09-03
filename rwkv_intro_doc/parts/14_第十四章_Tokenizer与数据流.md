# 第十四章 补课：文本是怎么变成数字的（Tokenizer 与数据流）

前面我们一直在说"token"，但可能有些小白还不清楚：**AI 读的并不是文字，而是一串数字。** 这一章专门补上这块拼图，讲清楚"文本 → token → 向量 → 模型 → 文本"的完整数据流，以及 RWKV 的 tokenizer 有什么特别之处。

## 14.1 为什么模型不能直接读文字

计算机里的神经网络只能处理**数字**，不能直接处理汉字、英文单词。所以我们要先把文字翻译成"模型能懂的数字表示"。这个过程分两步：

1. **Tokenize（分词）**：把句子切成若干"词元（token）"；
2. **Embedding（向量化）**：把每个 token 映射成一个高维数字向量。

模型拿到的是"一串向量"，处理完再反向解码成文字。

## 14.2 什么是 Token：最小处理单位

Token 是"模型处理文字的最小单位"。它可能是：

| 文本 | 可能的切分方式 |
| --- | --- |
| 一个汉字"爱" | 1 个 token（中文常按字/词切） |
| "人工智能" | 可能是 1 个 token（整词）或 4 个 token（逐字） |
| "hello" | 可能是 1 个 token |
| "world" | 1 个 token |
| "unbelievable" | 可能切成 "un" + "believ" + "able" 多个 token |

- **词级切分**：按空格/词切。缺点：词汇表巨大，且没见过的新词无法处理；
- **字符级切分**：按字母/字切。缺点：序列太长，效率低；
- **子词切分（主流）**：在"词"和"字"之间取平衡。常见的有 BPE（Byte Pair Encoding）、WordPiece、SentencePiece 等。

> 一个通用规律：**英文大约 1 个单词 ≈ 1.3 个 token；中文大约 1 个汉字 ≈ 1~2 个 token。** 这也解释了为什么"同样字数，中文模型的 token 开销往往更高"。

## 14.3 Tokenize 之后：Embedding（向量化）

切好 token 后，每个 token 会通过一张"查找表（embedding 表）"映射成一个固定维度的向量：

- 比如维度 d=768 或 d=4096；
- "爱" 和 "喜欢" 因为语义相近，它们的向量在高维空间里也会**靠得比较近**；
- 这张表是**模型训练时学出来的**。

> **小白版**：Token 是"字"，Embedding 是"这个字的含义坐标"。相似含义的词，坐标也相近。模型处理的就是这些"含义坐标"。

## 14.4 完整数据流：一段文字在模型里走一趟

![一次推理的完整数据流](img/fig16_api_flow.png)

用"我喜欢吃苹果"这句话举例：

| 步骤 | 发生了什么 | 数据形态 |
| --- | --- | --- |
| 1. 原始文本 | "我喜欢吃苹果" | 字符串 |
| 2. Tokenize | ["我","喜欢","吃","苹果"] | 一串 token id（整数） |
| 3. Embedding | 每个 id 查表得向量 | [4, d] 的矩阵 |
| 4. 模型前向 | 逐层计算，更新状态 S | 隐藏状态 + 输出分布 |
| 5. 采样 | 按概率选下一个 token | 一个 token id |
| 6. Decode | 把 id 查回文字 | 字符串 |
| 7. 循环 | 把新 token 接回输入，继续 | 直到结束 |

## 14.5 RWKV 的 Tokenizer 有什么特别

RWKV 使用社区自研的 tokenizer（如 `rwkv_vocab_v304`），并与 `rwkv` 库绑定使用。要点：

- **与模型配套**：加载模型时必须用配套的 tokenizer，否则会乱码（这是新手最常见的坑之一）；
- **多语言友好**：RWKV 的 tokenizer 对中文等多语言支持不错，这是它多语言表现好的基础之一；
- **字节级覆盖**：能处理各种 Unicode 字符（emoji、特殊符号等）。

```python
# 用 rwkv 库时，tokenizer 已经内置在 pipeline 里
from rwkv.utils import PIPELINE
pipeline = PIPELINE(model, "rwkv_vocab_v304")  # tokenizer 名称

# 看看"文字 → token"长什么样
ids = pipeline.encode("我喜欢吃苹果")
print(ids)          # [t1, t2, t3, t4, ...] token 整数序列

# 看看"token → 文字"（解码）
text = pipeline.decode(ids)
print(text)         # 我喜欢吃苹果
```

## 14.6 为什么"上下文长度"是以 token 计的

所有大模型的"上下文长度"（context length）都是用 **token 数**来衡量的，而不是字数。因为模型真正"看过"的是 token 序列：

- 一个模型的上下文是 4096 token，意味着它能"同时记住"最近约 4096 个 token 的内容；
- 中文一句约 20 字 ≈ 20-40 token，所以 4096 token ≈ 一百多句中文；
- RWKV 因为内存恒定，把上下文长度调大**几乎不增加推理成本**——这是它和 Transformer 的又一大区别（Transformer 上下文翻倍，KV Cache 也翻倍）。

```bash
# Ollama 里设置上下文长度
/set num_ctx 32768   # 例如把上下文调到 32K token
```

## 14.7 采样输出：模型如何"说下一个字"

模型前向计算的最后，会输出一个"**下一个 token 的概率分布**"——对词汇表里每个 token 都给出一个概率（比如"苹"40%、"火"30%、"鸡"20%……）。然后：

- **贪心采样**：永远选概率最高的那个 → 稳定但死板；
- **温度采样**：把概率分布"压扁"或"拉尖"后采样 → 温度越高越随机；
- **Top-P / Top-K**：先砍掉概率太低的候选，再采样 → 避免选到离谱的 token。

![采样决定"随机性 vs 稳定"](img/fig16_api_flow.png)

这就是第九章讲的参数（temperature/top_p）发挥作用的地方。

## 14.8 新手常见问题

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| 输出全是乱码/问号 | tokenizer 与模型不匹配 | 用配套 tokenizer |
| 英文很好、中文很怪 | tokenizer 中文覆盖率低 | 换多语言 tokenizer / World 模型 |
| 生成的文字超长不停止 | 没设 max_tokens 或没触发 EOS | 设置 max_tokens、增加停止词 |
| 输入被截断 | 超过上下文长度 | 调大 num_ctx，或分段处理 |

<div class="tip">

**本章小结**：AI 读的是数字不是文字。流程 = 文本 → tokenize 成 token → embedding 成向量 → 模型前向（更新状态 S）→ 采样 → decode 回文字，循环生成。RWKV 的 tokenizer 需与模型配套，中文约占 1-2 token/字，且上下文长度以 token 计。

</div>
