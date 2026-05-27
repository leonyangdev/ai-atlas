# 前向传播与反向传播

> 前向传播：数据流向"输出"，得到预测值。反向传播：误差流向"输入"，计算梯度、更新参数。这是神经网络学习的完整闭环。

## 一、符号定义与网络结构

以 **2-3-3 网络**（2 个输入、3 个隐藏层神经元、3 个输出类别）为例，进行完整推导。

```
输入 x (2,1)
    ↓  W[1](3,2)  b[1](3,1)
隐藏层 z[1]=W[1]x+b[1]  → a[1]=σ[1](z[1])  (3,1)
    ↓  W[2](3,3)  b[2](3,1)
输出层 z[2]=W[2]a[1]+b[2] → a[2]=σ[2](z[2])  (3,1) = ŷ
    ↓
损失 L = Loss(a[2], y)
```

**完整变量表**：

| 符号 | 含义 | 维度 |
|---|---|---|
| $x$ | 输入特征向量 | $(2, 1)$ |
| $W^{[1]}$ | 隐藏层权重矩阵 | $(3, 2)$ |
| $b^{[1]}$ | 隐藏层偏置向量 | $(3, 1)$ |
| $z^{[1]}$ | 隐藏层线性组合 | $(3, 1)$ |
| $a^{[1]}$ | 隐藏层激活输出 | $(3, 1)$ |
| $W^{[2]}$ | 输出层权重矩阵 | $(3, 3)$ |
| $b^{[2]}$ | 输出层偏置向量 | $(3, 1)$ |
| $z^{[2]}$ | 输出层线性组合 | $(3, 1)$ |
| $a^{[2]} = \hat{y}$ | 最终预测值 | $(3, 1)$ |
| $y$ | 真实标签 | $(3, 1)$ |
| $\mathcal{L}$ | 损失函数值 | 标量 |

---

## 二、前向传播（Forward Propagation）

数据从输入流向输出，产生预测值。

### 步骤 1：隐藏层计算

**线性组合**（维度校验：$(3,2) \times (2,1) + (3,1) \to (3,1)$）：

$$z^{[1]} = W^{[1]}x + b^{[1]}$$

**非线性激活**（逐元素操作，维度不变）：

$$a^{[1]} = \sigma^{[1]}(z^{[1]})$$

### 步骤 2：输出层计算

**线性组合**（维度校验：$(3,3) \times (3,1) + (3,1) \to (3,1)$）：

$$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$$

**输出激活**（Softmax 或 Sigmoid）：

$$a^{[2]} = \hat{y} = \sigma^{[2]}(z^{[2]})$$

### 步骤 3：计算损失

$$\text{Loss} = \mathcal{L}(a^{[2]}, y)$$

---

## 三、反向传播（Backward Propagation）

反向传播的目标：计算损失对每个参数的**偏导数（梯度）**，从输出层向输入层逆向传播。

**核心工具：链式法则**

$$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{\partial \mathcal{L}}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}} \cdot \frac{\partial a^{[1]}}{\partial z^{[1]}} \cdot \frac{\partial z^{[1]}}{\partial W^{[1]}}$$

### 第一阶段：输出层（Layer 2）梯度

**Step 1：损失对输出激活值的梯度**

$$da^{[2]} = \frac{\partial \mathcal{L}}{\partial a^{[2]}} \quad \text{维度: } (3,1)$$

**Step 2：穿过激活函数（链式法则）**

$$dz^{[2]} = da^{[2]} \odot {\sigma^{[2]}}'(z^{[2]}) \quad \text{维度: } (3,1) \odot (3,1) \to (3,1)$$

$\odot$ 表示逐元素相乘（Hadamard 乘积）。

**Step 3：输出层权重和偏置的梯度**

$$dW^{[2]} = dz^{[2]} (a^{[1]})^T \quad \text{维度: } (3,1) \times (1,3) \to (3,3) \checkmark$$

$$db^{[2]} = dz^{[2]} \quad \text{维度: } (3,1) \checkmark$$

### 第二阶段：隐藏层（Layer 1）梯度

**Step 4：误差跨越 $W^{[2]}$ 传回隐藏层**

$$da^{[1]} = (W^{[2]})^T dz^{[2]} \quad \text{维度: } (3,3)^T \times (3,1) \to (3,1) \checkmark$$

**Step 5：穿过隐藏层激活函数**

$$dz^{[1]} = da^{[1]} \odot {\sigma^{[1]}}'(z^{[1]}) \quad \text{维度: } (3,1) \checkmark$$

**Step 6：隐藏层权重和偏置的梯度**

$$dW^{[1]} = dz^{[1]} x^T \quad \text{维度: } (3,1) \times (1,2) \to (3,2) \checkmark$$

$$db^{[1]} = dz^{[1]} \quad \text{维度: } (3,1) \checkmark$$

---

## 四、参数更新

用梯度下降更新参数（学习率 $\alpha$）：

$$W^{[2]} \leftarrow W^{[2]} - \alpha \cdot dW^{[2]}$$
$$b^{[2]} \leftarrow b^{[2]} - \alpha \cdot db^{[2]}$$
$$W^{[1]} \leftarrow W^{[1]} - \alpha \cdot dW^{[1]}$$
$$b^{[1]} \leftarrow b^{[1]} - \alpha \cdot db^{[1]}$$

---

## 五、PyTorch 自动微分实现

```python
import torch
import torch.nn as nn

# 构建 2-3-3 网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)   # W[1](3,2), b[1](3,)
        self.layer2 = nn.Linear(3, 3)   # W[2](3,3), b[2](3,)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # 前向传播
        z1 = self.layer1(x)      # (batch, 3)
        a1 = self.relu(z1)       # (batch, 3)
        z2 = self.layer2(a1)     # (batch, 3)
        return z2  # 返回 logits，让 CrossEntropyLoss 处理 softmax

# 数据（单样本，batch_size=1）
x = torch.randn(1, 2, requires_grad=False)
y = torch.tensor([1])  # 真实类别 = 1

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# 训练一步
optimizer.zero_grad()     # 清空梯度缓存

logits = net(x)           # 前向传播
loss = criterion(logits, y)  # 计算损失

loss.backward()           # 反向传播（自动计算所有梯度）

# 查看梯度
print(f"W[1] 梯度形状: {net.layer1.weight.grad.shape}")  # (3,2)
print(f"W[2] 梯度形状: {net.layer2.weight.grad.shape}")  # (3,3)

optimizer.step()          # 参数更新
print(f"Loss: {loss.item():.4f}")
```

---

## 六、完整训练循环

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 生成数据
torch.manual_seed(42)
X = torch.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).long()  # 二分类

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型
net = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(),
    nn.Linear(16, 2)
)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(50):
    total_loss = 0
    for X_batch, y_batch in loader:
        # ① 前向传播
        logits = net(X_batch)
        loss = criterion(logits, y_batch)
        
        # ② 反向传播
        optimizer.zero_grad()   # 清空梯度（不能漏！）
        loss.backward()
        
        # ③ 参数更新
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")
```

---

## 七、梯度消失与梯度爆炸

### 梯度消失（Vanishing Gradient）

**成因**：多个 Sigmoid 导数（< 0.25）连乘，深层梯度趋近于 0

**表现**：底层权重不更新，网络停止学习

**解决方案**：
- 使用 ReLU 替换 Sigmoid
- 批量归一化（BatchNorm）
- 残差连接（ResNet）
- LSTM/GRU（对 RNN）

### 梯度爆炸（Exploding Gradient）

**成因**：权重过大，梯度连乘指数级增大

**表现**：Loss 出现 `NaN`，参数更新疯狂

**解决方案**：
- 梯度裁剪（Gradient Clipping）
- 权重初始化（Xavier/He）
- 批量归一化

```python
# 梯度裁剪
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # 限制梯度范数
optimizer.step()
```

---

## 八、Dying ReLU 问题

当神经元在前向传播时输出 $z \le 0$，ReLU 导数为 0，该神经元梯度为 0，参数永不更新。

**触发条件**：
1. 学习率过大，权重被过大幅度更新
2. 偏置初始化过负

**检测方法**：

```python
# 统计 ReLU 后输出为 0 的比例
def check_dead_neurons(model, x):
    activations = []
    def hook(module, input, output):
        activations.append((output == 0).float().mean().item())
    
    hooks = [layer.register_forward_hook(hook) 
             for layer in model.modules() if isinstance(layer, nn.ReLU)]
    
    with torch.no_grad():
        model(x)
    
    for h in hooks:
        h.remove()
    
    for i, ratio in enumerate(activations):
        print(f"ReLU 层 {i+1} 死亡神经元比例: {ratio:.2%}")
```

## 总结

| 阶段 | 方向 | 计算目的 |
|---|---|---|
| **前向传播** | 输入 → 输出 | 得到预测值 $\hat{y}$，计算损失 |
| **反向传播** | 输出 → 输入 | 计算损失对每个参数的梯度 |
| **参数更新** | 梯度 → 权重 | 沿负梯度方向更新，降低损失 |
