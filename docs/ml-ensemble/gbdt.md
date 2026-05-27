# 梯度提升决策树 GBDT

> GBDT（Gradient Boosting Decision Tree）是在表格数据领域"统治级"的经典算法——每一棵新树专注于纠正前面所有树的错误，沿着损失函数的负梯度方向逐步逼近最优解。

## 1. 核心思想

GBDT 是一种**加法模型**，名字中藏着两个关键词：

- **提升（Boosting — 接力纠错）**：模型串行训练，每一棵新树专门弥补前面所有树累积的**误差**
- **梯度（Gradient — 指引方向）**：新树沿着损失函数的**负梯度**方向拟合，是"广义残差"

### 学习率与树数量的权衡

| 学习率 $\nu$ | 单棵树补偿力度 | 所需树数 | 过拟合风险 |
|---|---|---|---|
| 大（0.3~1.0） | 强，每步大幅修正 | 少 | 高 |
| 小（0.01~0.1） | 弱，每步小步慢移 | 多 | 低 |

> 工业调参铁律：**小学习率（0.01~0.05）+ 足够多的树 + 限制树深（3~8）**

## 2. 数学原理

### 变量字典

| 符号 | 含义 |
|---|---|
| $N$ | 训练样本数 |
| $y_i$ | 第 $i$ 个样本真实值 |
| $F_m(x)$ | 前 $m$ 棵树的累积预测 |
| $r_{im}$ | 第 $m$ 轮第 $i$ 个样本的**负梯度（伪残差）** |
| $\nu$ | 学习率 $(0, 1]$ |
| $J$ | 叶子节点数 |
| $\gamma_{jm}$ | 第 $m$ 棵树第 $j$ 个叶子的输出值 |

### 完整工作流（以 MSE 回归为例）

**Step 0：初始化**

找常数 $c$ 使总损失最小：

$$F_0(x) = \arg\min_{c} \sum_{i=1}^{N} \frac{1}{2}(y_i - c)^2$$

令导数为 0，解得 $c = \bar{y}$（样本均值）。**在 MSE 下，初始预测值就是所有样本的平均数。**

**Step 1：计算负梯度（每轮迭代）**

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

代入 MSE 损失 $L = \frac{1}{2}(y_i - F)^2$，得到：

$$r_{im} = y_i - F_{m-1}(x_i)$$

**结论**：回归任务中，负梯度 = **残差（真实值 - 预测值）**。

**Step 2：训练新树**

以 $(x_i, r_{im})$ 为训练数据，建一棵回归树。叶子节点 $j$ 的最优输出值 $\gamma_{jm}$ 即落在该叶子的样本残差的均值。

**Step 3：更新总模型**

$$F_m(x) = F_{m-1}(x) + \nu \sum_{j=1}^{J} \gamma_{jm} \cdot \mathbb{1}[x \in R_{jm}]$$

### 数值示例（房价预测）

预测 3 套房价：$y = [100, 120, 140]$，学习率 $\nu = 0.1$

| 轮次 | 初始/当前预测 | 残差 | 更新后预测 |
|---|---|---|---|
| $F_0$ | [120, 120, 120] | [-20, 0, 20] | — |
| $F_1$ | — | — | [118, 120, 122] |
| $F_2$ | — | [-18, 0, 18] | [116.2, 120, 123.8] |
| ... | 积少成多，逐步逼近真实值 | | |

## 3. 分类任务：一个优美的数学巧合

GBDT 做分类时，通过 Sigmoid 将连续输出转为概率 $p = \sigma(F(x))$，损失函数变为**对数损失**：

$$L(y, F(x)) = -yF(x) + \log(1 + e^{F(x)})$$

对 $F(x)$ 求偏导并取负，得到负梯度：

$$r_{im} = y_i - p_i$$

| 场景 | 真实标签 | 预测概率 | 负梯度 | 物理含义 |
|---|---|---|---|---|
| 明显误判 | 1（正类） | 0.2 | +0.8 | 方向对了，但力度还不够，需要正向推力 |
| 方向错误 | 0（负类） | 0.8 | -0.8 | 方向反了，需要负向拉力修正 |
| 完美预测 | 1（正类） | 1.0 | 0.0 | 无需修正 |

## 4. 代码实现

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# ===== 分类示例 =====
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = GradientBoostingClassifier(
    n_estimators=100,    # 树的数量
    learning_rate=0.1,   # 学习率（步长）
    max_depth=5,         # 每棵树最大深度
    subsample=0.8,       # 样本随机采样比例（防过拟合）
    min_samples_split=5, # 分裂最小样本数
    random_state=42
)
clf.fit(X_train, y_train)
print(f"分类准确率: {accuracy_score(y_test, clf.predict(X_test)):.4f}")

# 追踪每轮迭代的测试集性能
train_scores = [accuracy_score(y_train, p) for p in clf.staged_predict(X_train)]
test_scores  = [accuracy_score(y_test,  p) for p in clf.staged_predict(X_test)]

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(train_scores, label='训练集')
plt.plot(test_scores,  label='测试集')
plt.xlabel('迭代轮数'); plt.ylabel('准确率')
plt.title('GBDT：随迭代增加的性能变化')
plt.legend(); plt.grid(alpha=0.3); plt.show()

# ===== 回归示例 =====
Xr, yr = make_regression(n_samples=300, n_features=10, noise=15, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2)

reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    loss='squared_error',   # MSE 损失
    random_state=42
)
reg.fit(Xr_train, yr_train)
print(f"回归 R²: {r2_score(yr_test, reg.predict(Xr_test)):.4f}")
```

## 5. 超参数调优

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':  [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth':     [3, 5, 7],
    'subsample':     [0.7, 0.8, 1.0],
}

search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
search.fit(X_train, y_train)
print(f"最佳参数: {search.best_params_}")
print(f"最佳 CV 得分: {search.best_score_:.4f}")
```

## 6. GBDT vs 随机森林 vs XGBoost

| 维度 | GBDT | 随机森林 | XGBoost |
|---|---|---|---|
| 训练方式 | 串行 | **并行** | 串行（特征并行） |
| 梯度信息 | 一阶 | 无 | **一阶 + 二阶** |
| 正则化 | 弱（学习率） | 中 | **强（L1/L2 + 叶子数）** |
| 过拟合控制 | 中 | **好** | **好** |
| 训练速度 | 慢 | 快 | 中 |
| 处理缺失值 | 需预处理 | 需预处理 | **自动** |

## 7. 优缺点

### 优点
- ✅ **高精度**：在表格数据上有"统治级"的表现
- ✅ **鲁棒性强**：换用 Huber Loss 可抵抗异常值
- ✅ **特征容忍度高**：不需要归一化

### 缺点
- ❌ **无法并行（致命伤）**：串行训练，大数据集训练极慢
- ❌ **高维稀疏特征表现差**：不如神经网络高效
- ❌ **超参数多**：学习率、树深、子采样需要细调

## 8. 与 XGBoost / LightGBM 的关系

```
GBDT（原型）
  ↓
XGBoost（二阶导 + 工程并行 + 正则化 + 缺失值处理）
  ↓
LightGBM（直方图加速 + Leaf-wise生长 + GOSS采样 + EFB特征捆绑）
```

工业界几乎不再用"裸 GBDT"，而是直接上 XGBoost 或 LightGBM。

## 总结

| 特性 | GBDT |
|---|---|
| **提出时间** | 2001（Friedman） |
| **核心机制** | 拟合负梯度（伪残差） |
| **弱分类器** | 回归决策树 |
| **训练方式** | 串行 Boosting |
| **历史意义** | XGBoost / LightGBM 的理论基础 |
