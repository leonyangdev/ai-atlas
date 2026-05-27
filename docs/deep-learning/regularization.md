# 参数初始化与正则化

> 初始化决定了模型在"损失地形"中的起点；正则化决定了它能否找到**泛化能力强**的最优解，而非死记硬背训练集。

## 第一部分：参数初始化

### 为什么初始化如此关键？

正确的初始化保证信号能在深层网络中传播而不消失/爆炸。

### 1. 零初始化（禁忌）

```python
# ❌ 绝对不能做
nn.Linear(128, 64)
nn.init.zeros_(layer.weight)  # 致命错误！
```

**为什么不行？对称性失效（Symmetry Breaking Failure）**

- 所有神经元完全相同 → 前向传播输出相同 → 接收到完全相同的梯度
- 所有神经元同进同退 → 1000 个神经元的作用退化为 1 个
- **网络永远无法学习**

> 注：偏置 $b$ 可以初始化为 0，因为权重 $W$ 的随机性已经打破对称性。

### 2. 为什么神经网络极度依赖"方差"？

把每一层想象成**信号中继器**：

$$\text{Var}(Z_{\text{out}}) = n \times \text{Var}(W) \times \text{Var}(X_{\text{in}})$$

其中 $n$ 是输入神经元数量。

**方差失控的后果（以 50 层网络为例）**：

| 情况 | 方差倍数 | 50 层后 | 结果 |
|---|---|---|---|
| $W$ 稍小 | $0.9$ | $0.9^{50} \approx 0.005$ | 所有激活值趋近 0 → **梯度消失** |
| $W$ 稍大 | $1.1$ | $1.1^{50} \approx 117$ | 数据方差爆炸 → **梯度爆炸**，出现 `NaN` |
| $W$ 恰好 | $1.0$ | $1.0^{50} = 1.0$ | 信号稳定传播 ✅ |

**目标**：让 $n \times \text{Var}(W) = 1$，即 $\text{Var}(W) = \frac{1}{n}$

### 3. Xavier 初始化（适用于 Tanh/Sigmoid）

**解方程**：$1 = n_{\text{in}} \times \text{Var}(W)$，得 $\text{Var}(W) = \frac{1}{n_{\text{in}}}$

**实际公式**（考虑输入输出的平衡）：

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

```python
import torch.nn as nn

layer = nn.Linear(128, 64)
nn.init.xavier_uniform_(layer.weight)   # 均匀分布版本
# 或
nn.init.xavier_normal_(layer.weight)    # 正态分布版本
```

**适用范围**：Sigmoid、Tanh 激活函数（负半区不为 0）。

### 4. He 初始化（适用于 ReLU）

**ReLU 的问题**：会把负数强制置零，相当于每次"砍掉一半"的方差。

**何恺明的解法**：初始化时给出**两倍的方差**来补偿 ReLU 的截断：

$$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$

```python
layer = nn.Linear(128, 64)
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
# 或
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

**PyTorch `nn.Linear` 默认使用 Kaiming Uniform（适合 ReLU）。**

### 初始化选型速查

| 激活函数 | 推荐初始化 |
|---|---|
| Sigmoid / Tanh | **Xavier** |
| ReLU / Leaky ReLU | **He（Kaiming）** |
| GELU / SiLU | He |
| 无激活函数（回归输出层） | Xavier 或 Lecun |

```python
import torch.nn as nn

# PyTorch 的完整初始化示例
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)   # 偏置初始化为 0 ✅
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10)
)
model.apply(init_weights)
```

---

## 第二部分：过拟合与正则化

### 什么是过拟合？

```
                        过拟合模型
Loss                    ┌─────────────────
│    训练集 ↓           │ 训练集 Loss 极低
│                       │ 测试集 Loss 极高（死记硬背训练集噪声）
│    验证集 ↑           │
└──────────── epoch     └─────────────────
```

**识别特征**：训练集 Loss 远低于验证集 Loss。

**过拟合的权重特征**：某些权重变得**极大**（如 $W = 1000$），模型对输入微小变化极其敏感。

---

### 正则化方法一：L2 权值衰减（Weight Decay）

**核心思想**：枪打出头鸟，任何权重都不能变得过大。

**在损失函数上加惩罚项**：

$$\mathcal{L}_{总} = \mathcal{L}_{原始} + \lambda \sum W^2$$

**反向传播时的影响**（$\lambda W^2$ 求导得 $2\lambda W$）：

$$w_{\text{new}} = w - \alpha(\nabla_w \mathcal{L} + 2\lambda w) = w(1 - 2\alpha\lambda) - \alpha\nabla_w \mathcal{L}$$

**效果**：每次更新，$w$ 都被按比例缩小（向 0 靠拢） → **权值衰减（Weight Decay）**

```python
# 在 AdamW 中正确使用 weight_decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-3, 
    weight_decay=0.01   # λ = 0.01
)
```

**为什么压小权重能防过拟合**：

权重小 → 模型对输入任何维度的剧烈变化不再敏感 → 输出曲线变得平滑 → 学到了数据的大趋势，而不是训练集的局部噪声。

---

### 正则化方法二：Dropout

**核心思想**：通过"随机让部分神经元休假"，迫使网络学习更鲁棒的特征。

**训练时**：按概率 $p$（丢弃率），随机将部分神经元输出置为 0。

**关键澄清**：丢弃不是"固定配额"，而是**每个神经元独立抛硬币**。

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),      # 50% 概率随机丢弃神经元
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 10)
)
```

**Inverted Dropout（逆向缩放）**：

训练时丢弃 $p=0.5$，只有 50% 神经元活跃；测试时 100% 激活，总输出变为原来的 2 倍——会导致训练/测试分布不一致！

**解决方案**（PyTorch 默认实现）：训练时把存活神经元的输出**除以 $(1-p)$** 放大，测试时不做任何调整。

```python
# 训练时与测试时的模式切换（必须！）
model.train()   # 启用 Dropout，神经元随机失活
model.eval()    # 关闭 Dropout，所有神经元激活
```

**Dropout 的正则化本质**：

- 每次训练相当于在**随机子网络**上优化
- 神经元 A 无法确定 B 是否还在，被迫独立学习有用特征
- 测试时等价于多个子网络的集成（Bagging 效果）

---

### 正则化方法三：Batch Normalization（BN）

**痛点**：即使初始化正确，训练中随着参数更新，**内部协变量偏移（Internal Covariate Shift）** 会让深层网络接收的数据分布持续漂移，训练极慢。

**BN 做法**：在每层的线性变换和激活函数之间插入**强制归一化关卡**。

对一个 Batch 中的 $B$ 个样本，在当前激活位置：

$$\mu_B = \frac{1}{B}\sum_{i=1}^B z_i, \quad \sigma_B^2 = \frac{1}{B}\sum_{i=1}^B (z_i - \mu_B)^2$$

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{z}_i + \beta \quad \text{（可学习的缩放和平移）}$$

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),    # 在激活函数之前
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

**BN 为什么有正则化效果？**

- 图片 A 的中间层值不只依赖自己，还依赖**同 Batch 内其他图片**
- 不同 Batch 的均值/方差不同 → 同一张图片每次处理结果略有抖动
- 这种**噪声**迫使网络学习"即使数值轻微变化依然成立的核心特征"
- 效果类似 Dropout，但更稳定

**变体对比**：

| 归一化 | 归一化维度 | 适用场景 |
|---|---|---|
| **BatchNorm（BN）** | 同一特征的 Batch 维度 | CNN 图像分类 |
| **LayerNorm（LN）** | 同一样本的特征维度 | Transformer/NLP |
| **GroupNorm** | 同一样本的特征子组 | 小 batch 的 CV |
| **InstanceNorm** | 每个样本单独归一化 | 图像风格迁移 |

```python
# LayerNorm（Transformer 中使用）
layer_norm = nn.LayerNorm(hidden_dim)
# x shape: (batch, seq_len, hidden_dim) → 沿最后一维归一化
```

---

## 现代实践建议

| 场景 | 推荐组合 |
|---|---|
| **CNN 图像模型** | He 初始化 + BatchNorm + Weight Decay（通常不加 Dropout） |
| **Transformer/NLP** | Xavier/He 初始化 + LayerNorm + AdamW Weight Decay + Dropout |
| **小规模 MLP** | Xavier + Dropout(0.3~0.5) + L2 |
| **大语言模型** | He 初始化 + LayerNorm + AdamW + 梯度裁剪 |

> **工程要点**：在现代卷积网络（ResNet 等）中，BN 的正则化效果往往已经足够强，通常只用 BN + Weight Decay，不再额外加 Dropout。在 Transformer 架构中，一般用 LayerNorm + Dropout + AdamW 的组合。
