# 支持向量机 (SVM)

> SVM 是一位"精英主义者"：决策边界只由少数几个关键数据点（支持向量）决定，其余数据对边界没有影响。

## 1. 核心思想

**SVM 的哲学**：找到一条分割线，让两类数据之间的"间隔"（Margin）最大化。

**三个核心概念**：

- **超平面（Hyperplane）**：用于分割两类数据的 $n-1$ 维面（二维中是直线，三维中是平面）
- **最大间隔（Maximum Margin）**：超平面距离两边最近数据点的距离尽可能大
- **支持向量（Support Vectors）**：恰好落在间隔边界上的那几个数据点

> 💡 **直觉**：想象你要在两个班同学中间画一条分界线，SVM 会选择让两边都"最宽松"的那条线，而不是随便画一条。

## 2. 数学推导

### 2.1 超平面方程

$$w^T x + b = 0$$

- $w$：权重向量（超平面的法向量，决定方向）
- $b$：偏置（决定超平面到原点的距离）
- $x$：输入数据点

### 2.2 最大化间隔

间隔（两个异类支持向量到超平面的总距离）为 $\frac{2}{||w||}$。

**最大化间隔 等价于 最小化 $||w||$**（最小化权重的范数）：

$$\min_{w, b} \frac{1}{2} ||w||^2$$

约束条件：

$$y_i(w^T x_i + b) \ge 1, \quad \forall i$$

### 2.3 对偶问题

通过拉格朗日乘子法，问题变为对偶形式，关键发现：

**优化过程完全依赖于数据点之间的点积 $x_i \cdot x_j$！**

这为核技巧奠定了基础。

## 3. 软间隔：处理噪声数据

现实数据很少完全线性可分。引入**软间隔**：

### 3.1 松弛变量

允许部分数据点落在间隔内或被误分类：

$$\min_{w, b, \xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i$$

约束：$y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$

### 3.2 惩罚参数 C

$$\begin{cases} C \text{ 大} \rightarrow \text{间隔窄，不容许错误，易过拟合} \\ C \text{ 小} \rightarrow \text{间隔宽，容许更多错误，更泛化} \end{cases}$$

## 4. 核技巧：处理非线性问题

**核心逻辑**：在低维空间无法线性分割时，将数据映射到高维空间，在高维中线性可分。

**核函数**：$K(x_i, x_j)$ 直接等于高维空间中两点的内积，无需显式计算高维坐标。

### 4.1 多项式核

$$K(x, y) = (\gamma(x \cdot y) + c)^d$$

引入特征之间的多次交叉组合。

### 4.2 高斯核（RBF 核）⭐ 最常用

$$K(x, y) = e^{-\gamma ||x - y||^2}$$

- 将数据映射到**无限维**空间（因为 $e^x$ 的泰勒展开是无穷多项！）
- $\gamma$ 控制单个数据点的影响范围：
  - $\gamma$ 大 → 影响范围小，边界复杂，易过拟合
  - $\gamma$ 小 → 影响范围大，边界平滑

> 💡 **为什么 RBF 核能映射到无限维？**
> 
> 根据泰勒展开：$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$
> 
> 这意味着高斯核隐式地包含了从一次方到无穷次方所有的特征组合！

### 4.3 线性核

$$K(x, y) = x \cdot y$$

即原始点积，适合高维稀疏数据（如文本分类）。

## 5. 代码实现

### 5.1 基本使用

```python
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# 线性可分数据
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                            n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ⚠️ SVM 必须标准化！
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 线性 SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_s, y_train)
print(f"线性 SVM 准确率: {svm_linear.score(X_test_s, y_test):.4f}")
print(f"支持向量数量: {svm_linear.n_support_}")
```

### 5.2 非线性数据（RBF 核）

```python
# 同心圆数据（线性不可分）
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_circles, y_circles, test_size=0.2)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# RBF 核 SVM
svm_rbf = SVC(kernel='rbf', C=10.0, gamma=1.0)
svm_rbf.fit(X_train_s, y_train)
print(f"RBF SVM 准确率: {svm_rbf.score(X_test_s, y_test):.4f}")

# 可视化决策边界
def plot_decision_boundary(clf, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k', s=30)
    plt.title(title)
    plt.show()

plot_decision_boundary(svm_rbf, X_test_s, y_test, "RBF SVM 决策边界")
```

### 5.3 超参数调优

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# C 和 gamma 的网格搜索
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'svm__kernel': ['rbf', 'linear']
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
])

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳验证准确率: {grid.best_score_:.4f}")
print(f"测试集准确率: {grid.score(X_test, y_test):.4f}")
```

## 6. SVM 业务场景案例

**场景：领导力培训平台预测学员是否结业**

数据特征：
- $x_1$：视频完播率（0-100%）
- $x_2$：讨论区互动频次

**异常情况**：有位高管完播率和互动都很低，但凭经验通过了考核。

**软间隔 C 的作用**：
- **高 C**：不容许这个异常点被误分类，决策边界扭曲，过拟合
- **低 C**：把这个高管视为可接受的例外，保持平滑边界

**RBF 核 γ 的作用**：
- 当结业学员分布为两个"岛屿"（理论型 + 社交型），线性不可分
- RBF 核可以学到复杂的非线性边界

## 7. 优缺点

### 优点
- ✅ **高维效果好**：文本、基因数据等高维场景表现优秀
- ✅ **全局最优**：凸优化问题，没有局部最优
- ✅ **内存效率高**：只需存储支持向量
- ✅ **核技巧强大**：可以处理非常复杂的非线性边界

### 缺点
- ❌ **大数据集慢**：训练复杂度 $O(n^2)$ 到 $O(n^3)$
- ❌ **必须标准化**：对特征尺度敏感
- ❌ **超参数调优难**：C 和 γ 需要精细调整
- ❌ **不直接输出概率**：需要额外的 Platt Scaling

## 8. SVM vs 逻辑回归

| 对比 | SVM | 逻辑回归 |
|------|-----|---------|
| 目标 | 最大化间隔 | 最大化似然 |
| 核技巧 | 支持 | 有限支持 |
| 输出 | 类别标签 | 概率 |
| 数据量 | 中小型数据集 | 大型数据集 |
| 高维 | 很好 | 可以 |

## 总结

| 核函数 | 适用场景 |
|-------|---------|
| 线性核 | 高维稀疏数据（文本分类）、线性可分 |
| RBF 核 | 通用场景，非线性数据的默认选择 |
| 多项式核 | 需要特征交叉的场景 |
