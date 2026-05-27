# Seq2Seq（序列到序列）

> Seq2Seq 是处理"**变长输入 → 变长输出**"问题的经典架构，是机器翻译、文本摘要、对话系统的基础，也是现代 Transformer 的前身。

## 1. 什么是 Seq2Seq？

Seq2Seq（Sequence to Sequence）的本质：将一个**输入序列**映射为**另一个输出序列**，两者长度可以不同。

由 Sutskever 等人在 2014 年 Google 论文 ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215) 中提出。

**典型应用**：

| 任务 | 输入 | 输出 |
|---|---|---|
| **机器翻译** | "I love you" | "我爱你" |
| **文本摘要** | 500 字新闻 | 50 字摘要 |
| **对话系统** | 用户提问 | AI 回答 |
| **语音识别** | 音频帧序列 | 文字序列 |
| **代码生成** | 自然语言描述 | Python 代码 |

## 2. 架构组成

Seq2Seq 由三个核心组件构成：

```
输入序列 [I, love, you]
    ↓
┌─────────────────┐
│    Encoder      │  → 压缩为 Context Vector (固定维度)
│  (LSTM/GRU)     │
└─────────────────┘
    ↓ Context Vector c
┌─────────────────┐
│    Decoder      │  → 逐步生成输出序列
│  (LSTM/GRU)     │
└─────────────────┘
    ↓
输出序列 [我, 爱, 你]
```

### Encoder（编码器）

- 通常是 LSTM 或 GRU
- 逐步读入输入序列，**将全部信息压缩为一个固定维度的隐藏状态向量**（Context Vector）
- $c = h_T$（编码器最后一个时间步的隐藏状态）

### Context Vector（上下文向量）

- **信息瓶颈**：整个输入序列的语义被压缩进一个向量
- 维度通常是 $[1, \text{batch}, \text{hidden\_dim}]$

### Decoder（解码器）

- 以 Context Vector 作为初始隐藏状态
- 第一个输入是特殊标记 `<BOS>`（句子开始）
- 每步预测一个词，将上一步预测结果作为下一步输入
- 直到生成 `<EOS>`（句子结束）标记

## 3. 翻译示例（逐步展开）

输入：**I love you** → 输出：**我爱你**

```
编码阶段：
  t=1: 读入 "I"    → h₁
  t=2: 读入 "love" → h₂
  t=3: 读入 "you"  → h₃ = Context Vector c

解码阶段（初始状态 = c）：
  t=1: 输入 <BOS>  → 预测 "我"
  t=2: 输入 "我"   → 预测 "爱"
  t=3: 输入 "爱"   → 预测 "你"
  t=4: 输入 "你"   → 预测 <EOS>  → 停止
```

## 4. PyTorch 实现

### 编码器

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))   # [batch, src_len, embed]
        outputs, (hidden, cell) = self.lstm(embedded)
        # hidden: [n_layers, batch, hidden_dim] → 作为 Decoder 的初始状态
        return outputs, hidden, cell
```

### 解码器

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt_token, hidden, cell):
        # tgt_token: [batch]（单个时间步的词索引）
        tgt_token = tgt_token.unsqueeze(1)             # [batch, 1]
        embedded = self.dropout(self.embedding(tgt_token))  # [batch, 1, embed]
        
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: [batch, 1, hidden]
        
        pred = self.fc(output.squeeze(1))  # [batch, vocab_size]
        return pred, hidden, cell
```

### 完整 Seq2Seq 模型

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        teacher_forcing_ratio: 训练时使用真实标签的概率（防止误差累积）
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc.out_features
        
        # 存储解码器输出
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # 编码
        enc_outputs, hidden, cell = self.encoder(src)
        
        # 解码（逐步生成）
        # 第一个输入是 <BOS>
        dec_input = tgt[:, 0]  # [batch]
        
        for t in range(1, tgt_len):
            pred, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[:, t] = pred
            
            # Teacher Forcing：随机决定下一步用真实词还是预测词
            use_teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_force:
                dec_input = tgt[:, t]           # 真实词（训练时加速收敛）
            else:
                dec_input = pred.argmax(dim=1)  # 模型预测词
        
        return outputs

# 初始化
SRC_VOCAB = 10000
TGT_VOCAB = 8000
HIDDEN_DIM = 256
EMBED_DIM = 128
N_LAYERS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(SRC_VOCAB, EMBED_DIM, HIDDEN_DIM, N_LAYERS, dropout=0.3)
decoder = Decoder(TGT_VOCAB, EMBED_DIM, HIDDEN_DIM, N_LAYERS, dropout=0.3)
model = Seq2Seq(encoder, decoder, device).to(device)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 5. Teacher Forcing（教师强制）

训练时，Decoder 应该用什么作为下一步的输入？

| 策略 | 做法 | 优缺点 |
|---|---|---|
| **纯自回归** | 始终用上一步预测词 | 推理一致，但错误会累积（曝光偏差） |
| **纯教师强制** | 始终用真实标签词 | 收敛快，但训练/测试不一致 |
| **混合（推荐）** | 按概率随机切换 | 平衡收敛速度和泛化能力 |

## 6. Beam Search 解码

推理时，Decoder 不再贪心地选最高概率词，而是保留 $k$ 个候选路径：

```python
def beam_search(decoder, init_hidden, init_cell, bos_idx, eos_idx, 
                beam_width=5, max_len=50):
    """
    beam_width: 同时维护的候选序列数
    """
    # 初始化：beam 内容、对数概率、完成状态
    beams = [([bos_idx], 0.0, init_hidden, init_cell)]
    completed = []
    
    for _ in range(max_len):
        new_beams = []
        for sequence, score, hidden, cell in beams:
            last_token = torch.tensor([sequence[-1]])
            
            pred, new_hidden, new_cell = decoder(last_token, hidden, cell)
            log_probs = torch.log_softmax(pred, dim=-1).squeeze(0)
            
            # 取 top-k 候选
            topk_probs, topk_ids = log_probs.topk(beam_width)
            
            for prob, token_id in zip(topk_probs, topk_ids):
                new_seq = sequence + [token_id.item()]
                new_score = score + prob.item()
                
                if token_id.item() == eos_idx:
                    completed.append((new_seq, new_score))
                else:
                    new_beams.append((new_seq, new_score, new_hidden, new_cell))
        
        # 按分数排序，保留 top-k
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        if not beams:
            break
    
    # 返回最高分序列
    all_candidates = completed + [(b[0], b[1]) for b in beams]
    return max(all_candidates, key=lambda x: x[1])[0]
```

## 7. Seq2Seq 的核心局限：信息瓶颈

```
长输入序列（100个词）
    ↓ Encoder
固定维度向量 c（256维）← 所有100个词的信息被压缩进来
    ↓ Decoder
生成输出序列
```

**问题**：句子越长，信息损失越严重！

| 输入长度 | 性能 |
|---|---|
| 短句（<20词） | 翻译质量较好 |
| 长句（>50词） | 质量明显下降，信息丢失严重 |

**解决方案：Attention 机制（2015年）**

Decoder 在生成每个词时，不仅使用 Context Vector，还可以**动态查询 Encoder 的所有隐藏状态**，选择性地关注输入的不同部分。

## 8. 技术演进路线

```
2014  Seq2Seq（纯 LSTM 编码-解码）
  问题：信息瓶颈，长句丢失信息
    ↓
2015  Seq2Seq + Attention（Bahdanau）
  解决：Decoder 能动态关注输入各位置
    ↓
2017  Transformer（Attention Is All You Need）
  革命：完全用 Attention 替代 RNN，全并行
    ↓
2018+ BERT / GPT / T5 等预训练大模型
  范式：大规模预训练 + 下游任务微调
```

## 总结

| 特性 | Seq2Seq |
|---|---|
| **提出时间** | 2014（Sutskever，Google）|
| **核心架构** | Encoder（LSTM）+ Context Vector + Decoder（LSTM）|
| **输入/输出** | 变长序列 → 变长序列 |
| **核心局限** | 信息瓶颈，长句性能差 |
| **后续改进** | Attention 机制 → Transformer |
| **历史地位** | 现代 NLP 编码-解码框架的鼻祖 |
