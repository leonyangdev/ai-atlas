# 随机森林

> 随机森林 = 多棵决策树 + 随机化（样本随机 + 特征随机）+ 集成投票

## 1. 核心思想

随机森林（Random Forest）是最成功的 Bagging 算法之一，由 Leo Breiman 于 2001 年提出。

### 两重随机性

1. **样本随机（Bootstrap 采样）**：每棵树使用有放回随机采样的训练子集
2. **特征随机**：每次分裂时，只从随机选择的 $m$ 个特征中找最优分裂

这两重随机性确保了树与树之间的多样性（差异性），是集成效果好的关键。

## 2. 算法步骤

```
FOR i = 1 to n_trees:
  1. Bootstrap 采样：有放回地采样 N 个样本 → 训练子集 Dᵢ
  2. 训练决策树（在 Dᵢ 上）：
     - 每次分裂时，随机选 m 个特征（m << 全部特征数）
     - 在这 m 个特征中选最优分裂点
     - 不剪枝（让每棵树充分生长）

预测：
  - 分类：所有树投票，多数类获胜
  - 回归：所有树预测均值
```

## 3. 袋外误差 (Out-of-Bag Error)

Bootstrap 采样时，约 37% 的样本不会被某棵树用到（袋外样本 OOB）。

可以用这些 OOB 样本来评估模型，无需单独的验证集：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X, y)

print(f"OOB 准确率: {rf.oob_score_:.4f}")  # 接近测试集准确率
```

## 4. 特征重要性

随机森林可以计算每个特征对分类的重要性（通过不纯度减少量）：

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 特征重要性
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
plt.xlabel('Feature Importance')
plt.title('Random Forest - Top 15 Features')
plt.tight_layout()
plt.show()
```

## 5. 完整代码示例

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=15, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林（不需要特征标准化）
rf = RandomForestClassifier(
    n_estimators=100,    # 树的数量
    max_depth=None,      # None = 不限深度
    min_samples_split=2, # 分裂的最小样本数
    min_samples_leaf=1,  # 叶子节点最小样本数
    max_features='sqrt', # 每次分裂考虑的特征数（√n）
    bootstrap=True,      # 是否 Bootstrap 采样
    oob_score=True,      # 计算 OOB 误差
    n_jobs=-1,           # 使用所有 CPU 核心
    random_state=42
)

rf.fit(X_train, y_train)

# 评估
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"OOB 准确率: {rf.oob_score_:.4f}")
print(f"测试集准确率: {rf.score(X_test, y_test):.4f}")

# 交叉验证
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"\n5折交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

## 6. 超参数调优

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# 随机搜索（推荐：搜索空间大时更高效）
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', 0.5]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    rf, param_dist, n_iter=50, cv=5, 
    scoring='accuracy', random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)

print(f"最佳参数: {random_search.best_params_}")
print(f"最佳得分: {random_search.best_score_:.4f}")
```

## 7. 优缺点

### 优点
- ✅ **通常不需要调参**：默认参数就能表现很好
- ✅ **不需要特征标准化**：对特征尺度不敏感
- ✅ **内置特征重要性**：自动进行特征选择
- ✅ **OOB 评估**：无需单独验证集
- ✅ **并行训练**：树与树之间互相独立
- ✅ **对异常值和噪声鲁棒**

### 缺点
- ❌ **模型不可解释**：黑盒，无法像单棵决策树那样直观理解
- ❌ **内存占用大**：存储所有树
- ❌ **预测相对慢**：需要对所有树预测并聚合
- ❌ **对非结构化数据弱**：不适合图像、文本

## 8. Random Forest vs GBDT

| 对比 | 随机森林 | GBDT/XGBoost |
|------|---------|-------------|
| 训练方式 | 并行 | 串行 |
| 目标 | 降低方差 | 降低偏差 |
| 调参难度 | 容易 | 较难 |
| 准确率 | 好 | 更好（通常）|
| 速度 | 快 | 慢（串行训练）|
| 过拟合风险 | 低 | 较高（需调参）|

**实践建议**：
- 快速获得不错的 Baseline → 随机森林
- 追求最高准确率 → XGBoost/LightGBM

## 总结

| 特性 | 值 |
|------|------|
| **训练方式** | Bootstrap 采样 + 特征随机 |
| **预测方式** | 多数投票（分类）/ 均值（回归）|
| **关键参数** | `n_estimators`、`max_depth`、`max_features` |
| **适用场景** | 通用，表格数据的强大 Baseline |
