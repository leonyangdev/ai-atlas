# 向量化计算

> 向量化是让代码快 100 倍的秘诀。深度学习框架（PyTorch/TensorFlow）的核心就是向量化矩阵运算。

## 一、为什么需要向量化？

看一个简单例子：对 100 万个数求和

```python
import numpy as np
import time

n = 1_000_000
a = np.random.randn(n)
b = np.random.randn(n)

# ❌ Python 循环（慢）
start = time.time()
result = 0
for i in range(n):
    result += a[i] * b[i]
print(f"Python 循环: {time.time() - start:.3f}s")

# ✅ NumPy 向量化（快）
start = time.time()
result = np.dot(a, b)
print(f"NumPy 向量化: {time.time() - start:.3f}s")

# 输出示例:
# Python 循环: 0.312s
# NumPy 向量化: 0.002s  ← 快 150 倍！
```

**原因**：NumPy 内部使用 C/Fortran 实现，底层调用 BLAS 库，充分利用 CPU 的 SIMD 指令集并行计算。GPU 上的加速更是成千上万倍。

## 二、向量化思维

### 2.1 消除显式循环

将"对每个元素做操作"替换为"对整个数组做操作"：

```python
import numpy as np

# ❌ 循环版本
def sigmoid_loop(z):
    result = []
    for x in z:
        result.append(1 / (1 + np.exp(-x)))
    return np.array(result)

# ✅ 向量化版本
def sigmoid_vectorized(z):
    return 1 / (1 + np.exp(-z))

z = np.random.randn(1000)
# 两者结果相同，但向量化版本快很多
```

### 2.2 神经网络向量化

考虑有 $m$ 个样本，每个样本 $n$ 个特征，神经网络第一层有 $h$ 个神经元：

```python
import numpy as np

# 假设参数
m = 1000    # 样本数
n = 784     # 输入特征数 (如 28x28 图片)
h = 256     # 隐藏层神经元数

X = np.random.randn(m, n)  # 输入矩阵 [m, n]
W = np.random.randn(n, h)  # 权重矩阵 [n, h]
b = np.zeros(h)            # 偏置 [h,]

# ❌ 循环版本 (非常慢)
Z_loop = np.zeros((m, h))
for i in range(m):
    for j in range(h):
        for k in range(n):
            Z_loop[i, j] += X[i, k] * W[k, j]
        Z_loop[i, j] += b[j]

# ✅ 向量化版本 (一行搞定！)
Z = X @ W + b  # shape: [m, h]

print(f"输入 X: {X.shape}")  # (1000, 784)
print(f"权重 W: {W.shape}")  # (784, 256)
print(f"输出 Z: {Z.shape}")  # (1000, 256)
```

## 三、广播机制 (Broadcasting)

广播允许不同形状的数组进行运算，NumPy 会自动扩展较小的数组：

```python
import numpy as np

# 规则：从右对齐，逐维比较
# 维度为 1 的可以广播扩展

# 示例 1: 矩阵加向量
A = np.ones((3, 4))    # shape: (3, 4)
b = np.array([1, 2, 3, 4])  # shape: (4,) → 广播为 (3, 4)
print((A + b).shape)   # (3, 4)

# 示例 2: 批量归一化
X = np.random.randn(32, 128)  # 32个样本, 128维特征
mean = X.mean(axis=0)          # (128,)
std = X.std(axis=0)            # (128,)

X_normalized = (X - mean) / (std + 1e-8)  # 广播减均值和除标准差
print(X_normalized.shape)  # (32, 128)

# 示例 3: 三维张量（深度学习中更常见）
# 批次数据: [batch_size, seq_len, hidden_dim]
attention_scores = np.random.randn(4, 10, 10)  # 4批次, 10序列, 10注意力
mask = np.zeros((1, 1, 10))                    # 广播到 (4, 10, 10)
masked_scores = attention_scores + mask
```

### 广播规则

1. 如果两个数组维度数不同，**在较小维度的数组左边补 1**
2. 维度大小为 1 的维度可以扩展到对方的大小
3. 任何维度大小不匹配且都不为 1，则报错

```
(3, 4) + (4,)     → (3, 4) + (1, 4) → (3, 4) ✅
(4, 3) + (4, 1)   → (4, 3) ✅
(1, 3, 4) + (3, 4) → (1, 3, 4) + (1, 3, 4) → (1, 3, 4) ✅
(3, 4) + (3,)     → ❌ 形状不兼容
```

## 四、NumPy 核心操作

### 4.1 数组创建

```python
import numpy as np

# 全零/全一
np.zeros((3, 4))
np.ones((3, 4))
np.eye(4)         # 4x4 单位矩阵

# 随机初始化（神经网络参数初始化用）
np.random.randn(3, 4)   # 标准正态分布
np.random.uniform(-0.1, 0.1, (3, 4))  # 均匀分布

# 序列
np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)   # [0, 0.25, 0.5, 0.75, 1]
```

### 4.2 形状操作（深度学习中极其重要！）

```python
import numpy as np

x = np.random.randn(2, 3, 4)

# reshape：改变形状，总元素数不变
x_flat = x.reshape(-1)           # shape: (24,)
x_2d = x.reshape(6, 4)           # shape: (6, 4)
x_same = x.reshape(2, -1)        # shape: (2, 12), -1 自动推断

# transpose：转置/交换维度
x_T = x.transpose(0, 2, 1)       # shape: (2, 4, 3)
x_T2 = x.swapaxes(1, 2)          # 等价于上面

# squeeze/unsqueeze：删除/添加维度为1的轴
x_sq = x_2d[np.newaxis, :]       # shape: (1, 6, 4) 添加批次维度
x_squeeze = x_sq.squeeze(0)      # shape: (6, 4) 删除批次维度

print(f"原始: {x.shape}")
print(f"扁平化: {x_flat.shape}")
print(f"重塑: {x_2d.shape}")
print(f"添加维度: {x_sq.shape}")
```

### 4.3 聚合操作

```python
import numpy as np

X = np.random.randn(32, 128)  # 32个样本, 128维

# 全局聚合
print(X.sum())    # 所有元素之和
print(X.mean())   # 所有元素平均
print(X.max())    # 最大值

# 按轴聚合
print(X.mean(axis=0).shape)  # (128,) 每个特征的均值
print(X.mean(axis=1).shape)  # (32,)  每个样本的均值
print(X.sum(axis=0, keepdims=True).shape)  # (1, 128) 保持维度
```

### 4.4 矩阵运算

```python
import numpy as np

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# 矩阵乘法
C1 = np.dot(A, B)      # shape: (3, 5)
C2 = A @ B             # 等价，更推荐用 @

# 元素级运算（Hadamard 乘积）
A2 = np.random.randn(3, 4)
D = A * A2             # 逐元素相乘，shape: (3, 4)

# 批量矩阵乘法（深度学习中的张量运算）
batch_A = np.random.randn(32, 3, 4)  # 32 个 3x4 矩阵
batch_B = np.random.randn(32, 4, 5)  # 32 个 4x5 矩阵
batch_C = batch_A @ batch_B          # shape: (32, 3, 5)
```

## 五、实际应用：线性回归向量化

```python
import numpy as np

class LinearRegressionVectorized:
    """向量化线性回归"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, lr=0.01, epochs=1000):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for epoch in range(epochs):
            # 前向传播 (向量化)
            y_pred = X @ self.weights + self.bias   # (m,)
            
            # 计算损失
            loss = np.mean((y_pred - y) ** 2)
            
            # 计算梯度 (向量化)
            dw = (2/m) * X.T @ (y_pred - y)        # (n,)
            db = (2/m) * np.sum(y_pred - y)         # 标量
            
            # 更新参数
            self.weights -= lr * dw
            self.bias -= lr * db
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    def predict(self, X):
        return X @ self.weights + self.bias

# 使用
X = np.random.randn(100, 3)
y = X @ np.array([1, 2, 3]) + 0.5 + np.random.randn(100) * 0.1

model = LinearRegressionVectorized()
model.fit(X, y, lr=0.01, epochs=500)
print(f"学到的权重: {model.weights}")  # 应接近 [1, 2, 3]
```

## 总结

| 操作 | Python 循环 | NumPy 向量化 | 加速比 |
|------|-----------|-------------|--------|
| 向量点积 | `sum(a[i]*b[i])` | `np.dot(a, b)` | 50-200x |
| 矩阵乘法 | 三重循环 | `A @ B` | 1000x+ |
| 批量操作 | for 循环 | 广播操作 | 10-100x |
| 激活函数 | 循环逐元素 | `np.exp(x)` | 50-100x |

> 💡 **规则**：在深度学习中，**消灭所有 Python 循环**，改用矩阵/张量操作！
