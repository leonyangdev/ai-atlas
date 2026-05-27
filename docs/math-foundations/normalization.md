# 归一化与标准化

> 数据缩放是机器学习中最容易被忽视却至关重要的预处理步骤。

## 一、为什么需要数据缩放？

考虑房价预测问题：

| 特征 | 范围 | 问题 |
|------|------|------|
| 房屋面积（㎡） | 50 ~ 500 | 数值大 |
| 卧室数量 | 1 ~ 6 | 数值小 |
| 建成年份 | 1950 ~ 2024 | 数值很大 |

如果不进行缩放，梯度下降会出现问题：
- 面积特征的梯度很大，学习率需要设很小
- 卧室数量的梯度很小，收敛很慢
- 整体训练不稳定，收敛缓慢

## 二、Min-Max 归一化

### 2.1 公式

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

归一化后的值范围在 $[0, 1]$ 之间。

### 2.2 使用场景

- **图像数据**：像素值 [0, 255] 归一化到 [0, 1]
- **神经网络输入**：保证所有特征在相同量级
- 数据范围已知且有界的情况

### 2.3 缺点

对**异常值**极其敏感！如果有一个极端值，会压缩大部分数据到很小的范围。

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 手动实现
def min_max_normalize(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8)  # 加小值防除零

# sklearn 实现
X = np.array([[1, 100], [2, 200], [3, 300], [4, 400]])
scaler = MinMaxScaler()

# ⚠️ 关键：fit 在训练集，transform 在训练集和测试集
X_train = X[:3]
X_test = X[3:]

scaler.fit(X_train)               # 学习训练集的最小/最大值
X_train_scaled = scaler.transform(X_train)   # 使用训练集参数归一化
X_test_scaled = scaler.transform(X_test)     # 使用相同参数归一化测试集

print("训练集缩放后:")
print(X_train_scaled)
# [[0.  0. ]
#  [0.5 0.5]
#  [1.  1. ]]
```

## 三、Z-Score 标准化

### 3.1 公式

$$x_{std} = \frac{x - \mu}{\sigma}$$

其中 $\mu$ 是均值，$\sigma$ 是标准差。

标准化后：均值为 0，标准差为 1（服从**标准正态分布**）。

### 3.2 使用场景

- **大多数机器学习算法的首选**
- 数据分布接近正态分布时效果最好
- 有异常值时比 Min-Max 更鲁棒

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# 手动实现
def z_score_normalize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return (X - mu) / (sigma + 1e-8)

# sklearn 实现
X = np.array([[1, 100, 2010],
              [2, 200, 2015],
              [3, 300, 2020],
              [4, 400, 2022]])

X_train, X_test = X[:3], X[3:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled = scaler.transform(X_test)        # 只 transform！

print("训练集均值（应接近 0）:", X_train_scaled.mean(axis=0))
print("训练集标准差（应接近 1）:", X_train_scaled.std(axis=0))
```

## 四、fit_transform 和 transform 的区别

这是初学者最容易犯的错误之一！

### 4.1 核心区别

| 方法 | 功能 | 使用时机 |
|------|------|---------|
| `fit(X)` | 学习参数（均值、标准差等） | **只在训练集上调用** |
| `transform(X)` | 使用已学参数变换数据 | 训练集和测试集都用 |
| `fit_transform(X)` | fit + transform 的组合 | **只在训练集上调用** |

### 4.2 为什么不能在测试集上 fit？

**数据泄露问题**！

如果在测试集上 `fit`，测试集的统计信息（均值、标准差、最大/最小值）就被"泄露"给了模型。这样模型在测试集上表现会虚高，但在真实生产环境中会表现很差。

```python
# ❌ 错误做法（数据泄露）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)  # ❌ 不应在测试集 fit！

# ✅ 正确做法
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 在训练集 fit 并 transform
X_test_scaled = scaler.transform(X_test)         # 仅 transform，使用训练集参数
```

### 4.3 完整流水线

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pipeline 自动处理 fit/transform 的时序问题
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)   # 只在训练集 fit
score = pipeline.score(X_test, y_test)  # 自动对测试集只 transform
```

## 五、其他归一化方法

### 5.1 最大绝对值缩放 (MaxAbsScaler)

$$x_{scaled} = \frac{x}{|x_{max}|}$$

适合已经中心化（均值为0）的稀疏数据。

### 5.2 鲁棒缩放 (RobustScaler)

$$x_{scaled} = \frac{x - Q_2}{Q_3 - Q_1}$$

其中 $Q_1$, $Q_2$, $Q_3$ 分别是第 25、50、75 百分位数。

**优点**：对异常值不敏感，适合含有噪声的数据。

### 5.3 深度学习中的 Layer Normalization

在 Transformer 架构中使用 Layer Normalization，对每个样本的特征维度进行归一化：

$$LN(x) = \frac{x - \mu_x}{\sigma_x} \cdot \gamma + \beta$$

其中 $\gamma$ 和 $\beta$ 是可学习参数。

```python
import torch
import torch.nn as nn

# PyTorch 内置 LayerNorm
layer_norm = nn.LayerNorm(normalized_shape=512)
x = torch.randn(32, 10, 512)  # batch=32, seq_len=10, hidden=512
x_normalized = layer_norm(x)
print(x_normalized.shape)  # (32, 10, 512)
```

### 5.4 Batch Normalization

在每个 mini-batch 中对每个特征维度进行归一化：

$$BN(x_i) = \gamma \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

```python
import torch.nn as nn

# CNN 中使用 BatchNorm
batch_norm = nn.BatchNorm2d(64)  # 64 个 channel
x = torch.randn(32, 64, 28, 28)  # batch, channel, H, W
x_normalized = batch_norm(x)
```

## 六、选择建议

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 一般机器学习 | Z-Score 标准化 | 通用，鲁棒 |
| 图像神经网络 | Min-Max [0,1] 归一化 | 像素值范围明确 |
| 含异常值 | RobustScaler | 不受异常值影响 |
| Transformer | Layer Normalization | 适合 NLP 序列 |
| CNN | Batch Normalization | 适合图像卷积 |

## 总结

```python
# 完整的特征预处理流程
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np

# 1. 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 创建预处理流水线
preprocessor = StandardScaler()

# 3. 在训练集上 fit，训练集和测试集都 transform
X_train_scaled = preprocessor.fit_transform(X_train)  # 训练集
X_test_scaled = preprocessor.transform(X_test)         # 测试集

# 4. 使用缩放后的数据训练模型
# ...
```
