# 降维算法

> 降维的本质：在**保留关键信息**的前提下，将数据从高维映射到低维——去噪、加速、可视化，三合一。

## 1. 为什么需要降维？

### 维度诅咒（Curse of Dimensionality）

| 问题 | 描述 |
|---|---|
| **空间稀疏性** | 维度增加，空间体积指数级增大，有限数据极度稀疏 |
| **距离失效** | 高维空间任意两点距离趋于相等，K-Means/KNN 等算法失效 |
| **计算量爆炸** | 特征维度过高，模型训练变慢 |
| **过拟合风险** | 特征越多，模型越容易"死记硬背"训练集噪声 |

降维是解以上问题的良药。

### 降维的两大路线

```
高维数据
    ├── 线性降维（旋转坐标轴）
    │       └── PCA（主成分分析）
    │
    └── 非线性降维（扭曲空间）
            ├── t-SNE（局部邻里保持）
            └── UMAP（全局+局部）
```

## 2. PCA（主成分分析）

### 2.1 核心哲学

**方差越大，信息量越多。** PCA 找一组新的正交坐标轴（主成分），使数据在这些轴上的**投影方差最大**。

直觉：把一朵有立体感的云（3D），从正确角度投射，得到最具区分度的影子（2D）。

### 2.2 数学推导

**步骤一：数据中心化**

$$X_{\text{centered}} = X - \mu$$

将数据平移使中心对准原点，简化后续计算。

**步骤二：计算协方差矩阵**

$$\Sigma = \frac{1}{m-1} X_{\text{centered}}^T X_{\text{centered}}$$

- 对角线：各特征的**方差**
- 非对角线：特征间的**协方差**（越接近0特征越独立）

**步骤三：特征值分解**

$$\Sigma v = \lambda v$$

- $v$（特征向量）：新坐标轴的**方向**（主成分）
- $\lambda$（特征值）：数据在该方向上的**方差大小**（信息量）

**步骤四：投影降维**

$$Y = X_{\text{centered}} W$$

$W$ 由前 $k$ 个最大特征值对应的特征向量组成，得到 $k$ 维降维结果。

### 2.3 代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 加载数据
data = load_iris()
X, y = data.data, data.target  # (150, 4)

# 标准化（PCA 前必须！否则方差大的特征会主导）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 降至 2 维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 查看方差解释比例
print(f"各主成分方差解释比例: {pca.explained_variance_ratio_}")
print(f"累计方差解释比例: {np.cumsum(pca.explained_variance_ratio_)}")

# 可视化
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i, (color, label) in enumerate(zip(colors, data.target_names)):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('鸢尾花数据集 PCA 降维可视化')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### 选择降维维度：碎石图（Scree Plot）

```python
# 先 fit 全部主成分
pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumsum = np.cumsum(explained_variance)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance)+1), explained_variance)
plt.xlabel('主成分编号'); plt.ylabel('解释方差比例')
plt.title('碎石图（Scree Plot）')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumsum)+1), cumsum, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% 阈值')
plt.xlabel('主成分数量'); plt.ylabel('累计解释方差')
plt.title('累计方差解释比例')
plt.legend()
plt.tight_layout(); plt.show()

# 自动选择解释 95% 方差所需的维度数
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"解释 95% 方差需要 {n_components_95} 个主成分")
```

### 2.4 PCA 的应用场景

| 场景 | 说明 |
|---|---|
| **模型预处理** | 去除冗余特征，缓解多重共线性 |
| **数据可视化** | 降至 2/3 维后可视化探索 |
| **数据去噪** | 舍弃方差极小的维度（通常是噪声） |
| **压缩存储** | 将成千上万维降至几十维 |
| **与 t-SNE 配合** | 先用 PCA 降至 50 维，再用 t-SNE 降至 2 维 |

## 3. t-SNE

### 3.1 核心哲学

**保持局部邻里关系**：高维中是"邻居"的点，在低维中也要紧靠在一起。

t-SNE 是非线性的——它不是"旋转坐标轴"，而是**扭曲空间**，像用橡皮膜包裹数据后摊开。

### 3.2 数学原理

**高维空间相似度（高斯分布）**

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

距离越远，$p_{j|i}$ 呈指数级衰减，只关注局部邻居。

**低维空间相似度（t 分布）**

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

使用 **t 分布**（长尾）而非高斯分布：让非邻居在低维空间被推得更远，解决"拥挤问题"。

**优化目标（最小化 KL 散度）**

$$C = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

通过梯度下降移动低维坐标，使 $Q$ 逼近 $P$。

### 3.3 代码实现

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# 手写数字数据集（8x8 = 64 维）
digits = load_digits()
X_digits, y_digits = digits.data, digits.target  # (1797, 64)

# t-SNE 降至 2 维（注意：t-SNE 不用于预处理，只用于可视化）
tsne = TSNE(
    n_components=2,
    perplexity=30,       # 控制局部/全局平衡，通常 5~50
    n_iter=1000,
    random_state=42,
    learning_rate='auto',
    init='pca'           # 用 PCA 初始化，更稳定
)
X_tsne = tsne.fit_transform(X_digits)

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                      c=y_digits, cmap='tab10', alpha=0.7, s=20)
plt.colorbar(scatter, label='数字类别')
plt.title('手写数字 t-SNE 可视化（64维→2维）')
plt.grid(alpha=0.3)
plt.show()
```

### 3.4 perplexity 参数的影响

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, perp in zip(axes, [5, 30, 100]):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_t = tsne.fit_transform(X_digits[:500])  # 取前500个加速
    ax.scatter(X_t[:, 0], X_t[:, 1], c=y_digits[:500], cmap='tab10', s=15, alpha=0.7)
    ax.set_title(f'perplexity={perp}')
plt.suptitle('perplexity 对 t-SNE 的影响')
plt.tight_layout(); plt.show()
```

### 3.5 注意事项

- **t-SNE 仅用于可视化**，不能用于预处理后喂给下游模型
- **不可重复性**：不同运行结果可能不同（加 `random_state` 固定）
- **计算慢**：$O(N^2)$，超过 1 万样本先用 PCA 压缩
- **距离无绝对意义**：簇间距离不代表真实相似度

## 4. PCA + t-SNE 最佳实践

大数据量时，单独运行 t-SNE 极慢。工程上的"组合拳"：

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

# 模拟高维数据（784维，如 MNIST）
# Step 1: PCA 降至 50 维（去噪 + 大幅提速）
pca_50 = PCA(n_components=50, random_state=42)
X_pca_50 = pca_50.fit_transform(X_high_dim)
print(f"PCA 50维解释方差: {pca_50.explained_variance_ratio_.sum():.2%}")

# Step 2: t-SNE 降至 2 维（精细展开）
tsne_2 = TSNE(n_components=2, perplexity=30, n_iter=1000, 
              random_state=42, init='pca')
X_2d = tsne_2.fit_transform(X_pca_50)

plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
plt.title('PCA(50维) + t-SNE(2维) 组合降维可视化')
plt.colorbar(label='类别'); plt.show()
```

## 5. 算法对比

| 维度 | PCA | t-SNE |
|---|---|---|
| **映射方式** | 线性（旋转坐标轴） | 非线性（扭曲空间） |
| **保留重点** | 全局方差（大轮廓） | 局部结构（近邻关系） |
| **计算复杂度** | 低 $O(n^3)$（特征数相关） | 高 $O(N^2)$（样本数相关） |
| **可用于预处理** | ✅ 可以 | ❌ 不能 |
| **可视化效果** | 中等（线性假设可能不够） | **优秀**（非线性结构清晰） |
| **可解释性** | 好（主成分有意义） | 差（坐标无绝对意义） |
| **大规模数据** | ✅ 适合 | ❌ 需先 PCA 预处理 |

## 6. 现代降维方法补充

### UMAP（2018）

UMAP（Uniform Manifold Approximation and Projection）比 t-SNE 更快、保持更好的全局结构，是目前最常用的可视化降维工具：

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
```

### 自编码器（Autoencoder）

用神经网络学习非线性降维，适合复杂数据（图像、文本）：

```python
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)   # 瓶颈层
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
```

## 总结

| 算法 | 最适合 | 注意事项 |
|---|---|---|
| **PCA** | 预处理降维、去噪、加速 | 先标准化，选合适维度数 |
| **t-SNE** | 高维数据可视化 | 只用于可视化，不能预处理 |
| **UMAP** | 可视化 + 保留全局结构 | 比 t-SNE 快，结构更完整 |
| **自编码器** | 复杂非线性数据 | 需要更多数据和计算资源 |
