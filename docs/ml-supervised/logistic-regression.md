# 逻辑回归

> 逻辑回归虽然名字里有"回归"，但本质上是一个分类算法。它通过 Sigmoid 函数将线性输出转化为概率。

## 1. 什么是逻辑回归？

逻辑回归（Logistic Regression）是用于解决**二分类（0 或 1）问题**的统计学习方法。

**核心思想**：不直接预测类别，而是预测**属于某一类别的概率**。

| 对比维度 | 线性回归 | 逻辑回归 |
|---------|---------|---------|
| 任务类型 | 回归（连续值预测） | 分类（概率预测） |
| 输出范围 | $(-\infty, +\infty)$ | $(0, 1)$ |
| 损失函数 | MSE | 对数损失（交叉熵） |

**应用场景**：
- **金融风控**：预测交易是否欺诈
- **医疗诊断**：预测肿瘤是恶性（1）还是良性（0）
- **精准营销**：预测用户是否点击广告（CTR 预估）

## 2. 核心架构

### 2.1 第一步：线性部分

$$z = w^T x + b = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b$$

### 2.2 第二步：Sigmoid 激活

为了将任意实数 $z$ 映射到 $(0, 1)$，引入 Sigmoid 函数：

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

- $z \to +\infty$：$\hat{y} \to 1$
- $z \to -\infty$：$\hat{y} \to 0$
- $z = 0$：$\hat{y} = 0.5$（最不确定的状态）

### 2.3 第三步：决策边界

以 0.5 为阈值：

$$\hat{label} = \begin{cases} 1 & \hat{y} \ge 0.5 \\ 0 & \hat{y} < 0.5 \end{cases}$$

也可以调整阈值来控制精确率/召回率。

## 3. 为什么不用 MSE？

如果逻辑回归使用 MSE：$L = \frac{1}{2}(\sigma(wx+b) - y)^2$

会导致两个问题：

1. **非凸性**：损失函数充满局部最小值，梯度下降无法找到全局最优
2. **梯度消失**：当 $\hat{y}$ 接近 0 或 1 时，Sigmoid 导数趋近于 0，参数几乎不更新

## 4. 对数损失函数（交叉熵）

基于**最大似然估计（MLE）** 推导：

$$L(\hat{y}, y) = -[y \ln(\hat{y}) + (1-y) \ln(1-\hat{y})]$$

- **保证凸性**：只有一个全局最低点
- **惩罚自信的错误**：预测错误且非常自信时，损失趋向无穷大

```python
import numpy as np

def log_loss(y_true, y_pred):
    """二分类交叉熵损失"""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 正确且自信：损失很小
print(log_loss([1], [0.99]))   # ≈ 0.01
# 错误且自信：损失很大
print(log_loss([1], [0.01]))   # ≈ 4.6
```

## 5. 梯度推导（完整过程）

求梯度 $\frac{\partial L}{\partial w}$，利用链式法则拆解为三步：

$$\frac{\partial L}{\partial w} = \underbrace{\frac{\partial L}{\partial \hat{y}}}_{\text{Step 1}} \cdot \underbrace{\frac{\partial \hat{y}}{\partial z}}_{\text{Step 2}} \cdot \underbrace{\frac{\partial z}{\partial w}}_{\text{Step 3}}$$

### Step 1：损失函数求导

$$L = -[y \ln(\hat{y}) + (1-y) \ln(1-\hat{y})]$$

$$\frac{\partial L}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

### Step 2：Sigmoid 求导

$$\hat{y} = \frac{1}{1 + e^{-z}}$$

$$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y}) \quad \text{（优美的自洽性质！）}$$

### Step 3：线性部分求导

$$z = wx + b \Rightarrow \frac{\partial z}{\partial w} = x$$

### 最终结果（完美的抵消！）

$$\frac{\partial L}{\partial w} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) \cdot x = (\hat{y} - y) \cdot x$$

> 💡 **惊奇之处**：分子分母中的 $\hat{y}(1-\hat{y})$ 完全抵消，梯度公式极其简洁！

## 6. 参数更新

$$w \leftarrow w - \alpha \cdot \frac{\partial L}{\partial w} = w - \alpha \cdot (\hat{y} - y) \cdot x$$

$$b \leftarrow b - \alpha \cdot (\hat{y} - y)$$

## 7. 代码实现

### 7.1 从零实现

```python
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_epochs=1000):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for epoch in range(self.n_epochs):
            # 前向传播
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            
            # 计算梯度（向量化！）
            dw = (1/m) * X.T @ (y_pred - y)
            db = (1/m) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# 示例
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

X, y = make_classification(n_samples=500, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(lr=0.1, n_epochs=500)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
```

### 7.2 使用 scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 加载乳腺癌数据集
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 训练
clf = LogisticRegression(max_iter=1000, C=1.0)  # C = 1/λ，正则化强度的倒数
clf.fit(X_train_s, y_train)

# 评估
y_pred = clf.predict(X_test_s)
print(classification_report(y_test, y_pred, target_names=['恶性', '良性']))

# 查看概率
y_proba = clf.predict_proba(X_test_s)[:5]
print("前5个样本的预测概率:", y_proba)
```

## 8. 正则化

```python
from sklearn.linear_model import LogisticRegression

# L1 正则化（稀疏解，自动特征选择）
clf_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)

# L2 正则化（默认，防过拟合）
clf_l2 = LogisticRegression(penalty='l2', C=1.0)

# 弹性网络（需要 saga solver）
clf_elastic = LogisticRegression(penalty='elasticnet', solver='saga', 
                                  l1_ratio=0.5, C=1.0)
```

## 9. 多分类

逻辑回归可以通过两种策略处理多分类：

1. **One-vs-Rest (OvR)**：训练 C 个二分类器（每个类对其余所有类）
2. **Softmax（多项式逻辑回归）**：直接输出每个类的概率分布

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)  # 3 分类

# multi_class='auto' 自动选择策略
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
clf.fit(X, y)

print("分类结果:", clf.predict(X[:5]))
print("各类概率:", clf.predict_proba(X[:5]).round(3))
```

## 10. 总结

| 特性 | 说明 |
|------|------|
| **任务类型** | 二分类 / 多分类 |
| **核心组件** | 线性函数 + Sigmoid 激活 |
| **损失函数** | 交叉熵（对数损失） |
| **参数更新** | $(\hat{y} - y) \cdot x$（简洁优雅）|
| **优点** | 可解释性强，输出概率，训练快 |
| **缺点** | 只能学习线性决策边界 |
| **扩展** | 使用多项式特征可处理非线性 |
