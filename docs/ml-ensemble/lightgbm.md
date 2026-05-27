# LightGBM

> LightGBM（Light Gradient Boosting Machine）是微软推出的面向大规模数据的高效梯度提升框架。在 GBDT 框架不变的前提下，用**直方图算法 + Leaf-wise 生长 + GOSS + EFB**实现了数倍到数十倍的提速，同时保持甚至超越 XGBoost 的精度。

## 1. 什么是 LightGBM？

LightGBM 的核心不是"换了一个模型"，而是在 **GBDT 框架**基础上进行工程与算法优化：

| 优化方向 | 具体技术 | 效果 |
|---|---|---|
| 加速分裂点搜索 | **Histogram 直方图算法** | 内存减少 8× 以上 |
| 更高效的树生长 | **Leaf-wise（按叶子生长）** | 同等叶子数精度更高 |
| 大数据集采样 | **GOSS（梯度单边采样）** | 保留关键样本加速 |
| 高维稀疏特征 | **EFB（互斥特征捆绑）** | 特征数减少 |

**最常用于**：
- 表格数据的分类/回归（工业界首选）
- 排序问题（Learning to Rank）
- 大规模/高维稀疏特征（广告 CTR/CVR）

## 2. 数学基础：GBDT 加法框架

LightGBM 属于 GBDT，模型是**加法模型**：

$$F_M(x) = \sum_{m=1}^{M} f_m(x)$$

训练目标：

$$\mathcal{L} = \sum_{i=1}^{N} \ell(y_i, F(x_i)) + \sum_{m=1}^{M} \Omega(f_m)$$

使用**二阶泰勒展开**，对第 $t$ 轮加入新树 $f_t$：

$$g_i = \frac{\partial \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i}, \quad h_i = \frac{\partial^2 \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$$

近似目标函数：

$$\tilde{\mathcal{L}}_t \approx \sum_{i=1}^{N}\left(g_i f_t(x_i) + \frac{1}{2}h_i f_t(x_i)^2\right) + \Omega(f_t)$$

### 最优叶子权重

假设叶子 $j$ 中的样本集合为 $I_j$，加入 L2 正则 $\frac{\lambda}{2}\sum_j w_j^2$，叶子的最优输出值：

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

### 分裂增益（Gain）

父节点分裂为左叶 $L$ 和右叶 $R$，增益：

$$\text{Gain} = \frac{1}{2}\left(\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right) - \gamma$$

- $\text{Gain} > 0$：值得分裂
- $\text{Gain} \le 0$：拒绝分裂（预剪枝）

**示例**：父节点 $(G=-10, H=5)$，按某阈值分裂后左 $(G_L=-6, H_L=3)$，右 $(G_R=-4, H_R=2)$，取 $\lambda=1, \gamma=0$：

$$\text{Gain} = \frac{1}{2}\left(\frac{36}{4} + \frac{16}{3} - \frac{100}{6}\right) = \frac{1}{2}(-2.33) \approx -1.17$$

Gain 为负，拒绝此分裂。

### 常用损失函数梯度

| 任务 | 损失函数 | $g_i$ | $h_i$ |
|---|---|---|---|
| 回归（MSE） | $\frac{1}{2}(y-\hat{y})^2$ | $\hat{y}-y$（残差） | $1$ |
| 二分类（Logloss） | $-[y\log p + (1-y)\log(1-p)]$ | $p - y$ | $p(1-p)$ |

## 3. 四大核心优化技术

### 3.1 Histogram 直方图算法

传统 GBDT（XGBoost 预排序）每次搜索所有可能分裂阈值，复杂度高。

LightGBM 把连续特征**离散化为 $K$ 个桶（bins）**，只在桶边界搜索：

```
连续值：[0.1, 0.5, 0.3, 0.9, 0.7, 0.2]
↓ 离散化为 4 个桶
桶号：  [  0,   2,   1,   3,   3,   0]
```

**效果**：
- 搜索候选从"样本级"降至"桶级"，速度提升数倍
- 用 uint8/uint16 存储桶号，内存减少 8× 以上
- 直方图可复用（父节点 = 左子树 + 右子树，可差分计算）

调参建议：`max_bin` 越大越精确但越慢，通常 255 即可。

### 3.2 Leaf-wise（按叶子生长）vs Level-wise

```
Level-wise（XGBoost / 传统 GBDT）   Leaf-wise（LightGBM）
┌──────────────────┐                 ┌──────────────────┐
│    层1：2叶子    │                 │    根节点        │
│    层2：4叶子    │ 均匀生长        │    ↓             │ 每次选
│    层3：8叶子    │                 │  Gain最大的叶子  │ Gain最大的叶子分裂
└──────────────────┘                 └──────────────────┘
```

**Leaf-wise 优点**：同等叶子数限制下，损失下降更快，精度更高  
**Leaf-wise 风险**：容易长出很深的树，过拟合风险高

> 防过拟合关键参数：`num_leaves`（叶子数上限）、`min_data_in_leaf`（叶子最小样本数）、`max_depth`

### 3.3 GOSS（梯度单边采样）

**直觉**：梯度大的样本对信息增益贡献更多，应优先保留。

**做法**：
1. 保留**大梯度样本**全部（如 top 20%）
2. 从**小梯度样本**随机抽取一部分（如 10%）
3. 对小梯度抽样部分做权重放大校正

**效果**：样本量减少一半甚至更多，但保留了关键信息。

### 3.4 EFB（互斥特征捆绑）

**直觉**：高维稀疏特征（如 one-hot 编码）中，很多特征几乎不会同时非零（互斥）。

**做法**：把"互斥"的稀疏特征捆绑成一个"bundle"，减少特征维度。

**典型场景**：one-hot 类别特征极多时（广告/推荐系统），可大幅减少搜索开销。

## 4. 代码实现

### 4.1 安装

```bash
pip install lightgbm
```

### 4.2 分类示例

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# LightGBM Dataset 格式
train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_test,  label=y_test, reference=train_data)

params = {
    'objective':       'binary',     # 二分类
    'metric':          'binary_logloss',
    'num_leaves':      31,           # 叶子数（核心容量参数）
    'max_depth':       -1,           # -1 表示不限制
    'learning_rate':   0.05,
    'n_estimators':    200,
    'min_data_in_leaf': 20,          # 防过拟合
    'feature_fraction': 0.8,         # 列采样
    'bagging_fraction': 0.8,         # 行采样
    'bagging_freq':    5,
    'lambda_l2':       0.1,          # L2 正则
    'verbose':        -1,
}

callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False),
             lgb.log_evaluation(period=50)]

model = lgb.train(
    params, train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=callbacks
)

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['恶性', '良性']))
```

### 4.3 Sklearn API（更简洁）

```python
from lightgbm import LGBMClassifier

clf = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(20, verbose=False)]
)
print(f"准确率: {accuracy_score(y_test, clf.predict(X_test)):.4f}")
```

### 4.4 超参数调优

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators':       randint(100, 500),
    'learning_rate':      uniform(0.01, 0.2),
    'num_leaves':         randint(20, 150),
    'min_child_samples':  randint(10, 50),
    'subsample':          uniform(0.6, 0.4),
    'colsample_bytree':   uniform(0.6, 0.4),
    'reg_lambda':         uniform(0, 2),
    'reg_alpha':          uniform(0, 1),
}

search = RandomizedSearchCV(
    LGBMClassifier(verbose=-1, random_state=42),
    param_dist, n_iter=50, cv=5,
    scoring='accuracy', random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)
print(f"最佳参数: {search.best_params_}")
```

## 5. 关键参数速查

| 参数 | 说明 | 推荐范围 |
|---|---|---|
| `num_leaves` | 叶子数上限（**最核心**） | 31~255 |
| `max_depth` | 最大深度（配合 num_leaves 防过拟合） | 6~12 |
| `min_data_in_leaf` | 叶子最小样本数（防过拟合） | 20~100 |
| `learning_rate` | 学习率 | 0.01~0.1 |
| `n_estimators` | 树的数量（配合 early stopping） | 100~3000 |
| `feature_fraction` | 列采样比例 | 0.6~0.9 |
| `bagging_fraction` | 行采样比例 | 0.6~0.9 |
| `lambda_l2` | L2 正则 | 0~2 |
| `max_bin` | 直方图桶数（精度 vs 速度） | 255 |

## 6. LightGBM vs XGBoost vs GBDT

| 对比 | GBDT | XGBoost | LightGBM |
|---|---|---|---|
| 分裂策略 | Level-wise | Level-wise | **Leaf-wise** |
| 内存占用 | 低 | 高（预排序） | **低（直方图）** |
| 训练速度 | 慢 | 中等 | **快** |
| 准确率 | 好 | 很好 | **通常最好** |
| 缺失值处理 | 需预处理 | 自动处理 | **自动处理** |
| 类别特征 | 需编码 | 需编码 | **原生支持** |
| 大数据扩展性 | 差 | 中 | **好** |

## 7. 优缺点

### 优点
- ✅ **训练速度快**：Histogram + GOSS + EFB 大幅提速
- ✅ **内存占用低**：离散化桶存储，内存友好
- ✅ **高精度**：Leaf-wise 能更高效降低损失
- ✅ **原生支持类别特征**：省去 one-hot 编码
- ✅ **工程功能完善**：early stopping、单调约束、SHAP 值

### 缺点
- ❌ **Leaf-wise 容易过拟合**：`num_leaves` 过大会失控
- ❌ **对特征泄露敏感**：容易出现"训练高分、线上崩"
- ❌ **小数据集不稳定**：数据量很小时结果抖动大
- ❌ **近似误差**：直方图分桶带来精度损失

## 8. 适用场景

**强烈推荐**：
- 结构化表格数据（金融风控、营销转化、用户流失预测）
- 高维稀疏特征（广告 CTR/CVR、推荐系统重排）
- 训练时延敏感、数据规模大的场景

**不建议**：
- 纯文本/图像/音频端到端任务（深度模型更合适）
- 强时序依赖需要序列建模（可配合特征工程使用）
- 极小数据集 + 强噪声（容易过拟合）

## 总结

| 特性 | LightGBM |
|---|---|
| **提出时间** | 2017（微软） |
| **核心创新** | Histogram + Leaf-wise + GOSS + EFB |
| **关键参数** | `num_leaves`、`min_data_in_leaf`、`learning_rate` |
| **适用场景** | 大规模结构化数据，工业界首选 |
| **历史地位** | 与 XGBoost 并列"竞赛双雄" |
