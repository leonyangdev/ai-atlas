# 线性代数基础

> 线性代数是深度学习的核心语言。理解向量和矩阵，就理解了神经网络的计算本质。

## 一、向量 (Vector)

### 1.1 什么是向量？

向量是一个有序的数字列表，用来表示方向和大小。在机器学习中，**每一个样本都可以用一个向量来表示**。

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

**直觉理解**：一个房子的特征向量可能是：
$$\mathbf{x} = \begin{bmatrix} 120 \text{（面积）} \\ 3 \text{（卧室数）} \\ 2 \text{（楼层）} \\ 5 \text{（建成年份）} \end{bmatrix}$$

### 1.2 向量运算

**加法**：对应元素相加

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \end{bmatrix}$$

**数量乘法**：每个元素乘以标量

$$\lambda \mathbf{a} = \begin{bmatrix} \lambda a_1 \\ \lambda a_2 \\ \vdots \end{bmatrix}$$

**点积（内积）**：

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

点积的几何意义：
$$\mathbf{a} \cdot \mathbf{b} = ||\mathbf{a}|| \cdot ||\mathbf{b}|| \cdot \cos\theta$$

- 点积 > 0：两向量夹角 < 90°（方向相近）
- 点积 = 0：两向量正交（相互独立）
- 点积 < 0：两向量夹角 > 90°（方向相反）

> 💡 **在 Attention 机制中**：Query 和 Key 的点积就是衡量"相关程度"，点积越大代表这两个词越相关！

## 二、向量范数 (Vector Norm)

**范数**是衡量向量"大小"或"长度"的方式。

### 2.1 L1 范数（曼哈顿距离）

$$||\mathbf{x}||_1 = \sum_{i=1}^{n} |x_i| = |x_1| + |x_2| + \cdots + |x_n|$$

- 几何意义：在城市街道中从一点到另一点的距离
- 应用：L1 正则化（Lasso），产生稀疏解（很多参数为 0）

### 2.2 L2 范数（欧氏距离）

$$||\mathbf{x}||_2 = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$$

- 几何意义：两点之间的直线距离
- 应用：L2 正则化（Ridge），使参数趋向于小但不为 0

### 2.3 Lp 范数（通用形式）

$$||\mathbf{x}||_p = \left(\sum_{i=1}^{n} |x_i|^p \right)^{1/p}$$

### 2.4 代码实现

```python
import numpy as np

x = np.array([3, 4])

# L1 范数
l1 = np.linalg.norm(x, ord=1)  # 3 + 4 = 7
print(f"L1 范数: {l1}")

# L2 范数
l2 = np.linalg.norm(x, ord=2)  # sqrt(9 + 16) = 5
print(f"L2 范数: {l2}")

# 向量归一化（单位向量）
x_normalized = x / l2
print(f"归一化向量: {x_normalized}")  # [0.6, 0.8]
```

## 三、矩阵 (Matrix)

### 3.1 矩阵的形状

矩阵是一个二维数字数组，形状为 $m \times n$（$m$ 行，$n$ 列）：

$$\mathbf{W} = \begin{bmatrix} w_{11} & w_{12} & \cdots & w_{1n} \\ w_{21} & w_{22} & \cdots & w_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{m1} & w_{m2} & \cdots & w_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

### 3.2 矩阵乘法

$$\mathbf{C} = \mathbf{A} \cdot \mathbf{B}, \quad A \in \mathbb{R}^{m \times k},\ B \in \mathbb{R}^{k \times n} \Rightarrow C \in \mathbb{R}^{m \times n}$$

$$c_{ij} = \sum_{l=1}^{k} a_{il} \cdot b_{lj}$$

> 💡 **神经网络的本质就是矩阵乘法**：每一层 `输出 = 激活函数(W × 输入 + b)`，其中 W 是权重矩阵，b 是偏置向量。

```python
import numpy as np

# 神经网络一层的前向传播
X = np.array([[1, 2, 3]])      # 输入：shape (1, 3)
W = np.random.randn(3, 4)      # 权重矩阵：shape (3, 4)
b = np.zeros(4)                # 偏置：shape (4,)

output = np.dot(X, W) + b      # 输出：shape (1, 4)
print(output.shape)            # (1, 4)
```

### 3.3 转置

$$(\mathbf{A}^T)_{ij} = A_{ji}$$

性质：$(\mathbf{AB})^T = \mathbf{B}^T \mathbf{A}^T$

### 3.4 重要矩阵类型

| 类型 | 定义 | 应用 |
|------|------|------|
| 单位矩阵 $\mathbf{I}$ | 对角线为 1，其余为 0 | $\mathbf{AI} = \mathbf{A}$ |
| 对称矩阵 | $\mathbf{A} = \mathbf{A}^T$ | 协方差矩阵，海森矩阵 |
| 正交矩阵 | $\mathbf{A}^T \mathbf{A} = \mathbf{I}$ | PCA 的旋转矩阵 |

## 四、特征值与特征向量

### 4.1 定义

对于方阵 $\mathbf{A}$，若存在非零向量 $\mathbf{v}$ 和标量 $\lambda$ 使得：

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

则称 $\lambda$ 为**特征值（eigenvalue）**，$\mathbf{v}$ 为对应的**特征向量（eigenvector）**。

**直觉理解**：特征向量是矩阵变换不改变方向的特殊向量，特征值是该方向上的伸缩比例。

### 4.2 应用：主成分分析（PCA）

PCA 通过计算数据协方差矩阵的特征值和特征向量，找到数据方差最大的方向，从而实现降维：

1. 对数据中心化（减去均值）
2. 计算协方差矩阵 $\mathbf{C} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$
3. 对 $\mathbf{C}$ 进行特征值分解，取前 $k$ 个最大特征值对应的特征向量
4. 将数据投影到这 $k$ 个方向

```python
import numpy as np
from sklearn.decomposition import PCA

# 生成高维数据
X = np.random.randn(100, 10)  # 100 个样本，10 个特征

# PCA 降到 2 维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"原始形状: {X.shape}")           # (100, 10)
print(f"降维后形状: {X_reduced.shape}") # (100, 2)
print(f"各成分解释的方差比: {pca.explained_variance_ratio_}")
```

## 五、广播机制 (Broadcasting)

NumPy 的广播机制允许不同形状的数组进行运算，这在深度学习中非常常用：

```python
import numpy as np

# 矩阵加向量（广播）
A = np.ones((3, 4))   # shape: (3, 4)
b = np.array([1, 2, 3, 4])  # shape: (4,)

C = A + b  # b 自动广播到 (3, 4)
print(C.shape)  # (3, 4)

# 批量归一化（神经网络中常用）
X = np.random.randn(32, 128)  # batch_size=32, hidden_dim=128
mu = X.mean(axis=0)           # shape: (128,)
X_normalized = X - mu         # 广播减均值
```

## 总结

| 概念 | 核心公式 | AI 中的应用 |
|------|---------|------------|
| 向量点积 | $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i$ | Attention 相似度 |
| L2 范数 | $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$ | 权重正则化，梯度裁剪 |
| 矩阵乘法 | $c_{ij} = \sum_l a_{il} b_{lj}$ | 神经网络线性层 |
| 特征分解 | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ | PCA 降维 |
