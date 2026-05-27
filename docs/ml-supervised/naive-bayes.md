# 朴素贝叶斯

## 1. 贝叶斯定理

朴素贝叶斯基于**贝叶斯定理**：

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

在分类问题中：

$$P(y|x) = \frac{P(x|y) \cdot P(y)}{P(x)}$$

- $P(y|x)$：后验概率（看到特征 x 后，y 的概率）—— **我们想要的**
- $P(x|y)$：似然（y 类别下，特征 x 出现的概率）
- $P(y)$：先验概率（不看特征时，y 的概率）
- $P(x)$：边缘概率（特征 x 出现的概率，与分类无关）

## 2. "朴素"在哪里？

**朴素假设（Naive Assumption）**：各特征之间**条件独立**。

$$P(x_1, x_2, \ldots, x_n | y) = \prod_{i=1}^{n} P(x_i | y)$$

这个假设在现实中往往不成立（比如"天气"和"温度"是相关的），但实践证明效果仍然很好。

## 3. 分类决策

对于新样本 $x$，选择后验概率最大的类别：

$$\hat{y} = \arg\max_y P(y|x) = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i|y)$$

由于 $P(x)$ 与 y 无关，可以忽略。

## 4. 朴素贝叶斯变体

### 4.1 高斯朴素贝叶斯

假设每个特征服从正态分布：

$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} e^{-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}}$$

适合：**连续特征**（如身高、体重）

### 4.2 多项式朴素贝叶斯

$$P(x_i|y) = \frac{N_{yi} + \alpha}{N_y + \alpha n}$$

适合：**文本分类**（词频统计，$\alpha$ 是拉普拉斯平滑）

### 4.3 伯努利朴素贝叶斯

每个特征是 0/1 的二值变量。

适合：**文档分类**（单词是否出现）

## 5. 代码实现

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# --- 高斯朴素贝叶斯（连续特征）---
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(f"高斯朴素贝叶斯准确率: {accuracy_score(y_test, y_pred):.4f}")

# --- 多项式朴素贝叶斯（文本分类）---
categories = ['sci.tech', 'rec.sport.baseball', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = vectorizer.fit_transform(newsgroups.data)
y_text = newsgroups.target

X_train, X_test, y_train, y_test = train_test_split(X_text, y_text, test_size=0.2)

mnb = MultinomialNB(alpha=1.0)  # alpha: 拉普拉斯平滑
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print(f"多项式朴素贝叶斯准确率: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=categories))
```

## 6. 从零实现高斯朴素贝叶斯

```python
import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}      # P(y)
        self.means = {}       # μ_y
        self.variances = {}   # σ²_y
        
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(y)
            self.means[c] = X_c.mean(axis=0)
            self.variances[c] = X_c.var(axis=0) + 1e-9  # 防止方差为0
        
        return self
    
    def _gaussian_likelihood(self, x, mean, var):
        """高斯概率密度"""
        exponent = -((x - mean) ** 2) / (2 * var)
        return np.exp(exponent) / np.sqrt(2 * np.pi * var)
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                # log 后验概率（避免数值下溢）
                log_prior = np.log(self.priors[c])
                log_likelihood = np.sum(np.log(
                    self._gaussian_likelihood(x, self.means[c], self.variances[c])
                ))
                posteriors[c] = log_prior + log_likelihood
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# 测试
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
print(f"准确率: {np.mean(gnb.predict(X_test) == y_test):.4f}")
```

## 7. 优缺点

### 优点
- ✅ **极快的训练和预测**：只需统计均值和方差
- ✅ **小数据集表现好**：参数少，不易过拟合
- ✅ **自然支持多分类**
- ✅ **对缺失值有一定鲁棒性**
- ✅ **增量学习**：新数据只需更新统计量

### 缺点
- ❌ **独立性假设不现实**：特征往往相关
- ❌ **连续特征需假设分布**（通常假设高斯）
- ❌ **零概率问题**：如果某特征在训练集中未出现，概率为 0（需拉普拉斯平滑）

## 8. 应用场景

| 场景 | 适用变体 |
|------|---------|
| 文本/邮件分类 | 多项式朴素贝叶斯 |
| 情感分析 | 多项式/伯努利朴素贝叶斯 |
| 医疗诊断 | 高斯朴素贝叶斯 |
| 实时在线学习 | 任意变体（增量更新） |

## 总结

| 特性 | 值 |
|------|------|
| **类型** | 概率生成模型 |
| **核心假设** | 特征条件独立 |
| **训练速度** | 非常快 $O(nd)$ |
| **优势场景** | 文本分类、小数据 |
| **关键技巧** | 拉普拉斯平滑、log 概率（防下溢）|
