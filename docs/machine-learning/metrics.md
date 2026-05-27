# 模型评估指标

> 选错评估指标，可能训练出一个"数字上很好看，实际上一塌糊涂"的模型。

## 一、分类任务指标

### 1.1 混淆矩阵 (Confusion Matrix)

对于二分类问题，所有预测结果分为四类：

|  | **实际正例 (P)** | **实际负例 (N)** |
|--|--------------|--------------|
| **预测正例** | **TP** (真正例) | **FP** (假正例，误报) |
| **预测负例** | **FN** (假负例，漏报) | **TN** (真负例) |

**以垃圾邮件检测为例**：
- **TP**：垃圾邮件被正确识别为垃圾邮件 ✅
- **FP**：正常邮件被误判为垃圾邮件 ❌（影响用户体验）
- **FN**：垃圾邮件没有被识别出来 ❌（核心失败）
- **TN**：正常邮件被正确识别为正常邮件 ✅

### 1.2 准确率 (Accuracy)

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**含义**：所有预测中正确的比例

**缺陷**：在类别不平衡时失去意义！

> 📌 例：假设 99% 的邮件是正常邮件，1% 是垃圾邮件。一个"无脑"模型把所有邮件都判为正常，准确率 = 99%，但它完全没用！

### 1.3 精确率 (Precision)

$$\text{Precision} = \frac{TP}{TP + FP}$$

**含义**：模型预测为正例的样本中，真正是正例的比例

**直觉**："我说是的，有多少真的是？"——**宁可错杀，不可放过**的对立面

**高精确率重要的场景**：法院判决（不能错杀好人）、垃圾邮件检测（不能把重要邮件判为垃圾）

### 1.4 召回率 (Recall / Sensitivity / TPR)

$$\text{Recall} = \frac{TP}{TP + FN}$$

**含义**：所有真正的正例中，被模型正确识别的比例

**直觉**："真正的病人中，有多少被我找出来了？"

**高召回率重要的场景**：癌症检测（不能漏诊）、欺诈检测（不能漏报欺诈）

### 1.5 精确率与召回率的权衡

精确率和召回率往往相互对立：

```
提高阈值 → 精确率↑，召回率↓（更保守，只在很有把握时才判正）
降低阈值 → 精确率↓，召回率↑（更激进，尽量多找正例）
```

**选择标准**：
- 误报代价高 → 优先提高精确率（如：过滤重要邮件）
- 漏报代价高 → 优先提高召回率（如：癌症筛查）

### 1.6 F1 分数 (F1 Score)

精确率和召回率的调和平均数：

$$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

**含义**：综合衡量精确率和召回率的指标，取值范围 [0, 1]

**推广**：$F_\beta$ 分数，可以调整精确率和召回率的权重：

$$F_\beta = (1 + \beta^2) \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}$$

- $\beta > 1$：更看重召回率
- $\beta < 1$：更看重精确率

### 1.7 假正例率 (FPR)

$$\text{FPR} = \frac{FP}{FP + TN}$$

**含义**：所有真正的负例中，被误判为正例的比例（误报率）

### 1.8 ROC 曲线与 AUC

**ROC 曲线**：通过改变分类阈值，绘制 TPR（召回率）vs FPR 的曲线

**AUC（Area Under the Curve）**：ROC 曲线下的面积

- AUC = 1.0：完美分类器
- AUC = 0.5：随机猜测（对角线）
- AUC < 0.5：比随机还差

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 加载数据
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 计算各指标
print("=" * 50)
print(f"准确率:  {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率:  {precision_score(y_test, y_pred):.4f}")
print(f"召回率:  {recall_score(y_test, y_pred):.4f}")
print(f"F1 分数: {f1_score(y_test, y_pred):.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(cm)

# 详细分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, 
                            target_names=['恶性', '良性']))

# ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR / 召回率)')
plt.title('ROC 曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## 二、回归任务指标

### 2.1 均方误差 (MSE)

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**特点**：对大误差惩罚更重（因为平方），单位是原始单位的平方

### 2.2 均方根误差 (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}}$$

**特点**：与原始数据单位相同，更直观

### 2.3 平均绝对误差 (MAE)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

**特点**：对异常值不敏感

### 2.4 决定系数 R²

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**含义**：模型解释了因变量方差的多大比例

- $R^2 = 1$：完美拟合
- $R^2 = 0$：与均值模型一样
- $R^2 < 0$：比均值模型还差

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_true = np.array([3, 2.5, 5.1, 8, 7.2])
y_pred = np.array([2.8, 2.6, 5, 8.3, 7.5])

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
```

## 三、多分类任务

多分类时，精确率、召回率、F1 需要指定平均方式：

```python
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
# 省略分割和训练步骤...

# macro：每个类别权重相同
# micro：全局计算（受样本量影响）
# weighted：按类别样本数加权

print(classification_report(y_test, y_pred, 
                            target_names=['Setosa', 'Versicolor', 'Virginica'],
                            zero_division=0))
```

## 四、指标选择指南

| 场景 | 推荐指标 | 原因 |
|------|---------|------|
| 类别平衡 | Accuracy | 直观简单 |
| 类别不平衡 | F1-Score / AUC-ROC | 不受类别比例影响 |
| 漏报代价高（癌症检测） | Recall（召回率） | 不能漏诊 |
| 误报代价高（垃圾邮件） | Precision（精确率） | 不能误判正常邮件 |
| 综合评估二分类 | AUC-ROC | 阈值无关，综合评估 |
| 需要概率输出 | Log Loss | 衡量概率预测质量 |
| 回归 | RMSE（可解释）或 MAE（鲁棒） | 根据是否需要对大误差更敏感 |

## 总结

```
关键公式速查表：

准确率 = (TP + TN) / 全部样本
精确率 = TP / (TP + FP)        ← 预测为正中真正是正的比例
召回率 = TP / (TP + FN)        ← 真正正例中被预测出的比例
F1     = 2×P×R / (P+R)         ← 精确率和召回率的调和平均
FPR    = FP / (FP + TN)        ← 负例中误判为正的比例（ROC 横轴）
```
