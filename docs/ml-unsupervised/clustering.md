# 聚类算法

> 聚类是无监督学习的核心任务：让算法自动发现数据中的自然群组，使得**同组内相似度高，不同组间差异大**。

## 1. 什么是聚类？

聚类（Clustering）将数据集划分为若干"簇"，无需任何标签。核心应用：

- **客户细分**：按行为/购买记录将用户分群，精准营销
- **图像分割**：将像素按颜色/纹理归组
- **异常检测**：不属于任何正常群组的点即为异常
- **文本分析**：自动对新闻/评论分主题归类
- **生物信息学**：基因表达数据聚类，发现疾病类别

## 2. K-Means 聚类

### 算法步骤

```
1. 随机选择 K 个初始聚类中心
2. 计算每个数据点到各中心的距离，分配到最近中心
3. 重新计算每个簇的均值作为新中心
4. 重复 2-3 步，直到中心不再变化（收敛）
```

### 目标函数（最小化簇内平方距离之和）

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

其中 $\mu_k$ 是第 $k$ 个簇的质心。

### 如何选择 K：肘部法则

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K 值')
plt.ylabel('簇内误差平方和（SSE）')
plt.title('肘部法则选择最优 K')
plt.xticks(K_range)
plt.grid(alpha=0.3)
plt.show()
# 找 SSE 曲线的"拐点"，拐点处即为最优 K
```

### 完整示例

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 训练 K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30)
plt.scatter(centers[:, 0], centers[:, 1], 
            s=300, c='red', marker='X', zorder=5, label='中心点')
plt.title('K-Means 聚类结果')
plt.legend()
plt.show()

# 评估
score = silhouette_score(X, labels)
print(f"轮廓系数: {score:.4f}")  # 越接近 1 越好
```

### 优缺点

| 优点 | 缺点 |
|---|---|
| 简单高效，适合大规模数据 | 必须预先指定 K 值 |
| 收敛速度快 | 对初始中心敏感（可用 K-Means++ 改进） |
| 易于实现和理解 | 只适合**凸形（球状）**簇 |
| | 对异常值和噪声敏感 |

### K-Means++ 初始化

为解决对初始中心敏感的问题，K-Means++ 以**距离概率正比**的方式选择初始中心：

```python
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
```

## 3. DBSCAN（密度聚类）

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）通过**密度**定义簇，不需要指定 K 值，且能处理任意形状的簇。

### 核心概念

| 概念 | 定义 |
|---|---|
| **核心点** | 以该点为圆心、半径 ε 的邻域内至少有 MinPts 个点 |
| **边界点** | 不是核心点，但在某核心点的 ε 邻域内 |
| **噪声点** | 既不是核心点也不是边界点 |

```
ε = 0.5, MinPts = 4
    
    ●●●        ← 核心点构成簇
   ●●●●●
    ●●●
         ○     ← 边界点（可达但不是核心点）
               ×  ← 噪声点（孤立点）
```

### 代码示例

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# 月牙形数据（K-Means 无法处理）
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# DBSCAN 可以处理非凸形状
db = DBSCAN(eps=0.3, min_samples=5)
labels_db = db.fit_predict(X_moons)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
km_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X_moons)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=km_labels, cmap='bwr', alpha=0.7)
plt.title('K-Means（失败）')

plt.subplot(1, 2, 2)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_db, cmap='bwr', alpha=0.7)
plt.title('DBSCAN（成功）')

plt.tight_layout()
plt.show()

n_noise = (labels_db == -1).sum()
print(f"噪声点数量: {n_noise}")
```

### 参数调优

```python
# 使用 k-distance 图辅助选择 ε
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5).fit(X_moons)
distances, _ = nbrs.kneighbors(X_moons)
# 排序第5近邻的距离，找"拐点"即为合适的 ε
k_distances = np.sort(distances[:, 4])[::-1]
plt.plot(k_distances)
plt.axhline(y=0.3, color='r', linestyle='--', label='ε=0.3')
plt.xlabel('点的索引'); plt.ylabel('第5近邻距离')
plt.title('k-距离图（辅助选择 ε）')
plt.legend(); plt.show()
```

### 优缺点

| 优点 | 缺点 |
|---|---|
| **无需指定 K 值** | 参数 ε 和 MinPts 难以确定 |
| **适合任意形状的簇** | 对高维数据表现较差（维度诅咒） |
| **能检测噪声/异常点** | 密度差异大时效果不稳定 |
| 对初始值不敏感 | |

## 4. 层次聚类

层次聚类（Hierarchical Clustering）不需要指定 K，产生**树状图（Dendrogram）**，展示所有可能的聚类层次。

### 两种方向

- **自底向上（凝聚型）**：每个点先各自一组，逐步合并最近的两组
- **自顶向下（分裂型）**：所有点先一组，逐步分裂

### 代码示例

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 绘制树状图
linked = linkage(X[:50], method='ward')  # 取前50个点可视化
plt.figure(figsize=(12, 5))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('层次聚类树状图（Dendrogram）')
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.show()

# 截取 4 个簇
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agg = agg.fit_predict(X)
print(f"轮廓系数: {silhouette_score(X, labels_agg):.4f}")
```

## 5. 高斯混合模型（GMM）

GMM 假设数据是**多个高斯分布的混合**，使用 EM 算法优化参数，可以处理椭圆形簇（比 K-Means 更灵活）。

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X)
labels_gmm = gmm.predict(X)
proba = gmm.predict_proba(X)  # 每个点属于各簇的概率

print(f"轮廓系数: {silhouette_score(X, labels_gmm):.4f}")
print(f"BIC（越小越好）: {gmm.bic(X):.2f}")
```

## 6. 算法对比与选择

| 算法 | 需指定K | 任意形状 | 处理噪声 | 速度 | 适用场景 |
|---|---|---|---|---|---|
| **K-Means** | ✓ | ✗ | 差 | 快 | 大规模数值数据，凸形簇 |
| **K-Means++** | ✓ | ✗ | 差 | 快 | 同上，更稳定初始化 |
| **DBSCAN** | ✗ | ✓ | **好** | 中 | 任意形状，异常检测 |
| **层次聚类** | ✗ | 部分 | 差 | 慢 | 小数据集，需要层次结构 |
| **GMM** | ✓ | 椭圆 | 一般 | 中 | 概率聚类，软分配 |

**选择建议**：
- 快速实验 → K-Means++
- 形状复杂 + 有噪声 → DBSCAN
- 需要树状关系 → 层次聚类
- 软概率归属 → GMM

## 7. 聚类评估指标

### 内部评估（无标签）

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 轮廓系数：-1 ~ 1，越接近1越好
sil = silhouette_score(X, labels)

# Calinski-Harabasz 指数：越大越好
ch = calinski_harabasz_score(X, labels)

# Davies-Bouldin 指数：越小越好
db = davies_bouldin_score(X, labels)

print(f"轮廓系数: {sil:.4f}")
print(f"CH 指数: {ch:.2f}")
print(f"DB 指数: {db:.4f}")
```

### 外部评估（有参考标签时）

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(true_labels, pred_labels)      # 调整兰德系数
nmi = normalized_mutual_info_score(true_labels, pred_labels)  # 归一化互信息
```

## 总结

| 算法 | 核心机制 | 使用场景 |
|---|---|---|
| K-Means | 最小化簇内距离平方和，迭代更新质心 | 大规模数据，凸形簇 |
| DBSCAN | 密度可达性，噪声容忍 | 任意形状，异常检测 |
| 层次聚类 | 逐步合并/分裂，树状图 | 小数据集，层次关系 |
| GMM | EM 算法，概率软分配 | 椭圆形簇，概率估计 |
