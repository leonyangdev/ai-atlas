# XGBoost

> XGBoost（eXtreme Gradient Boosting）是由陈天奇开发的极端梯度提升算法，是 Kaggle 竞赛中最受欢迎的"杀器"之一。

## 1. 什么是 XGBoost？

XGBoost 是对梯度提升（Gradient Boosting）算法的工程优化和数学增强：

- **集成学习**：训练多棵简单的决策树，串行地组合它们
- **梯度提升**：每棵新树专门**修正前面所有树累积的错误**
- **极端**：引入二阶导数信息、正则化项、特征并行计算等改进

## 2. 核心原理

### 2.1 目标函数

训练第 $t$ 棵树时，目标函数包括**损失项**和**正则化项**：

$$Obj^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

- $l(...)$：损失函数（MSE、对数损失等）
- $\hat{y}_i^{(t-1)}$：前 $t-1$ 棵树的累积预测（已知常数）
- $f_t(x_i)$：第 $t$ 棵树对样本的预测值（优化目标）
- $\Omega(f_t)$：正则化项，惩罚树的复杂度

### 2.2 正则化项

$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

- $T$：叶子节点数量（树的复杂度）
- $\gamma$：叶子节点惩罚系数（门槛越高，树越难分裂）
- $w_j$：第 $j$ 个叶子节点的预测值
- $\lambda$：L2 正则化系数

### 2.3 二阶泰勒展开

对损失函数进行**二阶泰勒展开**，利用一阶导 $g_i$ 和二阶导 $h_i$：

- **一阶导** $g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$：误差方向
- **二阶导** $h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$：误差变化的曲率

### 2.4 最优叶子权重

对第 $j$ 个叶子节点，最优预测值为：

$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

其中 $G_j = \sum_{i \in I_j} g_i$，$H_j = \sum_{i \in I_j} h_i$（叶子中所有样本的导数之和）。

### 2.5 分裂增益

评估是否值得分裂一个节点：

$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

- 左子节点得分 + 右子节点得分 - 父节点得分 - 新叶子惩罚
- **如果 Gain ≤ 0，则不分裂**（预剪枝）

### 2.6 数值示例

假设一个节点有 5 个样本，一阶导 $g_i = [-0.2, -0.5, 0.4, 0.8, 0.5]$，$h_i = 1$（MSE 时）：

- 父节点：$G = 1.0$，$H = 5.0$，设 $\lambda=1$，$\gamma=0.5$
- 按"年龄=25"分裂：左 $G_L=-0.7$，$H_L=2$；右 $G_R=1.7$，$H_R=3$

$$Gain = \frac{1}{2}\left[\frac{0.49}{3} + \frac{2.89}{4} - \frac{1.0}{6}\right] - 0.5 = -0.141$$

**结论**：Gain < 0，XGBoost 拒绝此分裂！

## 3. 主要特性

| 特性 | 说明 |
|------|------|
| **二阶导数** | 比 GBDT 收敛更快、更精准 |
| **内置正则化** | 防过拟合，L1/L2 + 叶子数量惩罚 |
| **处理缺失值** | 自动学习缺失值应进左子树还是右子树 |
| **特征并行** | 在寻找最优分裂点时特征级别并行 |
| **列子采样** | 类似随机森林的特征随机，防过拟合 |

## 4. 代码实现

### 4.1 分类示例

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 训练 XGBoost 分类器
clf = xgb.XGBClassifier(
    n_estimators=100,        # 树的数量
    max_depth=5,             # 最大深度
    learning_rate=0.1,       # 学习率（缩减因子）
    subsample=0.8,           # 行采样比例
    colsample_bytree=0.8,    # 列采样比例
    gamma=0,                 # 分裂最小增益阈值
    reg_alpha=0,             # L1 正则化
    reg_lambda=1,            # L2 正则化
    objective='binary:logistic',
    random_state=42,
    verbosity=0
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = clf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, 
                            target_names=['恶性', '良性']))

# 特征重要性
importances = clf.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
top_features = [data.feature_names[i] for i in top_idx]

plt.figure(figsize=(10, 6))
plt.barh(top_features, importances[top_idx], color='steelblue')
plt.xlabel('特征重要性')
plt.title('XGBoost Top 10 特征重要性')
plt.tight_layout()
plt.show()
```

### 4.2 回归示例

```python
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_regression(n_samples=300, n_features=20, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    objective='reg:squarederror',
    random_state=42,
    verbosity=0
)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred)**0.5:.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### 4.3 超参数调优

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 1),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0.5, 2)
}

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(verbosity=0),
    param_dist, n_iter=50, cv=5,
    scoring='accuracy', random_state=42
)
random_search.fit(X_train, y_train)
print(f"最佳参数: {random_search.best_params_}")
```

## 5. XGBoost vs GBDT vs LightGBM

| 对比 | GBDT | XGBoost | LightGBM |
|------|------|---------|---------|
| 分裂策略 | 按层分裂 | 按层分裂 | 按叶子分裂（更准确）|
| 内存占用 | 低 | 高（预排序） | 低（直方图）|
| 训练速度 | 慢 | 中等 | 快 |
| 准确率 | 好 | 很好 | 通常最好 |
| 缺失值处理 | 需预处理 | 自动处理 | 自动处理 |
| 类别特征 | 需编码 | 需编码 | 原生支持 |

## 6. 优缺点

### 优点
- ✅ **准确率极高**：二阶导数加持，逼近最优解精准
- ✅ **自带防过拟合**：内置 L1/L2 正则化，列采样
- ✅ **优雅处理缺失值**：无需手动填充缺失值
- ✅ **工程计算优化**：特征并行、缓存友好
- ✅ **灵活的目标函数**：支持自定义损失函数

### 缺点
- ❌ **内存占用高**：预排序需要存储数据 Block
- ❌ **不适合非结构化数据**：图像、音频、文本效果差
- ❌ **超参数多**：调参相对复杂

## 总结

XGBoost 是表格数据领域（结构化数据）最强大的算法之一，在各类 Kaggle 竞赛和工业场景中广泛应用。

**适用场景**：
- 金融风控（欺诈检测、信贷评分）
- 电商推荐（CTR 预估）
- 医疗诊断
- 任何结构化数据的分类/回归任务
