# 决策树

> 决策树就像一个"是/否"问题游戏，通过一系列有序的判断，最终给出预测结果。

## 1. 什么是决策树？

决策树是一种树形结构的机器学习算法，适用于**分类**和**回归**任务。

**直觉理解**：判断"今天是否适合出去玩"：

```
天气晴朗？
├── 否 → 不适合出去玩 ❌
└── 是 → 气温合适？
          ├── 否 → 不适合出去玩 ❌
          └── 是 → 今天有空？
                    ├── 是 → 适合出去玩 ✅
                    └── 否 → 不适合 ❌
```

## 2. 如何选择分裂特征？

**核心问题**：每次分裂时，选哪个特征、哪个阈值？

**目标**：让分裂后的子集尽量"纯净"——同一组数据的类别尽可能一致。

### 2.1 信息熵

**信息熵**衡量数据集的混乱程度（不纯度）：

$$H(S) = -\sum_{c=1}^{C} p_c \log_2 p_c$$

- 熵越小 → 数据越纯（越好）
- 若全部是同一类：$H = 0$（最纯）
- 若各类均等：$H = \log_2 C$（最混乱）

### 2.2 信息增益 (ID3)

$$\text{IG}(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

**选择信息增益最大的特征分裂**。

### 2.3 信息增益比 (C4.5)

改进 ID3，防止偏向取值多的特征：

$$\text{GainRatio}(S, A) = \frac{\text{IG}(S, A)}{H_A(S)}$$

### 2.4 基尼指数 (CART)

$$\text{Gini}(S) = 1 - \sum_{c=1}^{C} p_c^2$$

- 基尼指数越小 → 数据越纯
- CART 使用基尼指数选择最优分裂

```python
import numpy as np

def gini_impurity(y):
    """计算基尼指数"""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def information_entropy(y):
    """计算信息熵"""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-15))

# 纯净数据
y_pure = np.array([1, 1, 1, 1, 1])
print(f"纯净: 基尼={gini_impurity(y_pure):.3f}, 熵={information_entropy(y_pure):.3f}")

# 混合数据
y_mixed = np.array([1, 1, 0, 0])
print(f"混合: 基尼={gini_impurity(y_mixed):.3f}, 熵={information_entropy(y_mixed):.3f}")
```

## 3. 为什么希望分裂时数据尽量"干净"？

这是决策树的核心哲学。

**例子**：预测包裹是否准时送达

| 特征 | 分裂1（交通情况） | 分裂2（快递员经验） |
|-----|----------------|-----------------|
| 左组 | [准时, 准时, 准时] | [准时, 延迟, 准时] |
| 右组 | [延迟, 延迟] | [延迟, 准时, 延迟, 准时] |

**分裂1 更好**：两个子组都更纯净，可以更准确地预测。

如果数据不"干净"，意味着知道某个特征值后，依然无法确定类别，这个特征对预测没有帮助。

## 4. 决策树算法

| 算法 | 特征选择标准 | 适用数据 | 特点 |
|------|------------|---------|------|
| ID3 | 信息增益 | 离散特征 | 偏向多取值特征 |
| C4.5 | 信息增益比 | 离散+连续 | 改进了 ID3 |
| CART | 基尼指数 (分类) / MSE (回归) | 通用 | 生成二叉树，sklearn 默认 |

## 5. 剪枝

决策树很容易**过拟合**（把训练数据"记住"了），需要剪枝：

### 5.1 预剪枝

在构建过程中**提前停止**：

```python
from sklearn.tree import DecisionTreeClassifier

# 通过超参数控制剪枝
clf = DecisionTreeClassifier(
    max_depth=5,          # 最大深度（最重要！）
    min_samples_split=10, # 分裂所需最少样本数
    min_samples_leaf=5,   # 叶节点最少样本数
    min_impurity_decrease=0.01  # 分裂所需最小不纯度减少量
)
```

### 5.2 后剪枝

先长成完整树，再通过代价-复杂度剪枝（CCP）：

```python
# 查找最优 ccp_alpha
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# 对不同 alpha 值进行交叉验证
from sklearn.model_selection import cross_val_score

scores = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    scores.append(cv_score.mean())

best_alpha = ccp_alphas[np.argmax(scores)]
```

## 6. 代码实现

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练（使用基尼指数，最大深度3）
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 可视化决策树
plt.figure(figsize=(15, 8))
plot_tree(clf, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,  # 颜色填充
          rounded=True,
          fontsize=12)
plt.title("鸢尾花决策树（max_depth=3）")
plt.show()
```

## 7. 快递送达预测示例

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# 创建数据集
data = {
    '交通情况': ['拥堵', '一般', '畅通', '拥堵', '一般', '畅通', '拥堵', '一般', '畅通', '拥堵'],
    '天气': ['雨天', '雨天', '晴天', '晴天', '雨天', '晴天', '雨天', '晴天', '晴天', '雨天'],
    '快递员经验': ['低', '中', '高', '中', '低', '高', '中', '高', '低', '高'],
    '准时到达': ['否', '否', '是', '是', '否', '是', '否', '是', '是', '否']
}
df = pd.DataFrame(data)

# 编码
le = LabelEncoder()
for col in ['交通情况', '天气', '快递员经验', '准时到达']:
    df[col] = le.fit_transform(df[col])

X = df.drop('准时到达', axis=1)
y = df['准时到达']

# 训练
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X, y)

# 可视化
plt.figure(figsize=(12, 8))
plot_tree(model, 
          feature_names=X.columns,
          class_names=['延迟', '准时'],
          filled=True, 
          rounded=True)
plt.title("快递送达预测决策树")
plt.show()
```

## 8. 决策树的优缺点

### 优点
- ✅ **可解释性强**：可以直接可视化决策过程
- ✅ **无需特征归一化**：对特征尺度不敏感
- ✅ **自动特征选择**：不重要的特征不会被选择
- ✅ **处理非线性**：可以学习非线性决策边界
- ✅ **适用离散和连续特征**

### 缺点
- ❌ **容易过拟合**：特别是不剪枝的情况
- ❌ **不稳定性**：数据的小变化可能导致完全不同的树
- ❌ **偏向多取值特征**（ID3 算法）
- ❌ **无法外推**：不能预测超出训练范围的值

## 9. 应用场景

| 场景 | 说明 |
|------|------|
| 信用评分 | 规则清晰，银行可向客户解释 |
| 医疗诊断 | 医生可理解决策逻辑 |
| 故障诊断 | 工程师可追溯决策路径 |
| 推荐系统 | 初级产品推荐（更复杂时用树集成）|

## 10. 总结

| 特性 | 说明 |
|------|------|
| **分裂标准** | 基尼（CART）、信息增益（ID3）、信息增益比（C4.5） |
| **防过拟合** | 预剪枝（max_depth）、后剪枝（ccp_alpha）|
| **特征要求** | 无需标准化 |
| **可解释性** | 极强 |
| **升级版** | 随机森林（Bagging）、GBDT（Boosting） |
