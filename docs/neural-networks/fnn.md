# 前馈神经网络 FNN

> 前馈神经网络（Feedforward Neural Network，FNN）也叫**多层感知机（MLP）**，是最基础的深度学习模型，是 CNN、RNN、Transformer 的共同基石。

## 1. 什么是 FNN？

FNN 的本质是一个"**输入经过一系列矩阵变换 + 非线性函数，最后输出结果**"的结构。

- 数据**单向流动**（从输入层 → 隐藏层 → 输出层），没有循环
- 每个神经元与下一层**全连接**（Full Connection），所以也叫"全连接网络"

**应用场景**：
- 表格型数据分类/回归
- LSTM、Transformer 中的子模块（FFN 层）
- 图像分类（展平成向量后）
- 特征型数据处理

## 2. 结构组成

### 输入层
接收**定长向量**输入：
- 灰度图 28×28 → 展平为 784 维向量
- TF-IDF 文本特征 → 300 维向量

### 隐藏层（核心）

每层 = **线性变换 + 激活函数**：

$$z = xW + b \quad a = \text{ReLU}(z)$$

**数据流维度示例**（batch=64，输入10维，隐藏32维，输出2维）：

```
x:      [64, 10]
W1:     [10, 32]
b1:     [32]
h = x @ W1 + b1:     [64, 32]
h_relu = ReLU(h):    [64, 32]
W2:     [32, 2]
out = h_relu @ W2:   [64, 2]
```

### 输出层
- 分类任务：接 Softmax → 概率分布
- 二分类：接 Sigmoid → 单个概率
- 回归任务：线性输出 → 直接数值

## 3. 为什么要加激活函数？

若无非线性激活，多层线性变换可以合并为一层：

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$$

**结论**：没有激活函数，100 层网络 = 1 层线性模型，完全无法处理非线性问题。

## 4. PyTorch 实现

### 方式一：nn.Sequential（推荐简单场景）

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),   # 输入层 → 隐藏层1
    nn.ReLU(),
    nn.Dropout(0.3),       # 正则化
    nn.Linear(256, 128),   # 隐藏层1 → 隐藏层2
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10)     # 隐藏层2 → 输出层（10分类）
)

x = torch.randn(64, 784)   # 64 张展平图片
out = model(x)
print(out.shape)           # torch.Size([64, 10])
```

### 方式二：继承 nn.Module（推荐复杂场景）

```python
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),     # 批量归一化
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 使用示例
net = FNN(input_dim=784, hidden_dim=256, output_dim=10)
print(net)
```

## 5. 完整训练示例（MNIST 手写数字）

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64)

# 模型
model = nn.Sequential(
    nn.Flatten(),                    # 28x28 → 784
    nn.Linear(784, 256), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 训练
for epoch in range(10):
    model.train()
    total_loss, correct = 0, 0
    
    for X, y in train_loader:
        optimizer.zero_grad()
        logits = model(X)                # 前向传播
        loss = criterion(logits, y)      # 计算损失
        loss.backward()                  # 反向传播
        optimizer.step()                 # 更新参数
        
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
    
    # 验证
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            test_correct += (model(X).argmax(1) == y).sum().item()
    
    print(f"Epoch {epoch+1:2d}: "
          f"训练准确率={correct/len(train_data):.4f}, "
          f"测试准确率={test_correct/len(test_data):.4f}")
```

## 6. FNN vs RNN

| 特征 | FNN（前馈） | RNN（循环） |
|---|---|---|
| **输入结构** | 定长向量 | 任意长序列 |
| **参数共享** | 否 | 是（所有时间步共享） |
| **顺序建模** | 否 | 是（有时间感知） |
| **记忆能力** | 无 | 有（短期） |
| **并行性** | 完全并行 | 串行（依赖上一步） |

## 7. FNN 在现代深度学习中的角色

FNN 已不再单独用于复杂任务，但它是各类架构的**核心子模块**：

```
Transformer 架构
├── Multi-Head Attention（注意力层）
├── Feed-Forward Network（FNN！每个 token 独立通过）
│       └── Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model)
└── Layer Norm
```

**理解 FNN，是理解所有深度学习架构的起点。**

## 总结

| 特性 | FNN |
|---|---|
| **别名** | MLP、全连接网络 |
| **基本单元** | Linear + 激活函数 |
| **训练方式** | 前向传播 + 反向传播（BP） |
| **适用场景** | 表格数据、作为其他架构子模块 |
| **局限性** | 无法处理变长序列，无时间感知 |
