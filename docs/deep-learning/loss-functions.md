# 损失函数

> 损失函数是神经网络训练的"指挥棒"——它衡量预测值与真实值的差距，训练的本质就是通过梯度下降让这个数值**降到最低**。

## 理解损失函数的两个视角

### 微观视角：建造山谷的形状

- **坐标系**：X 轴 = 预测误差 $(y - \hat{y})$，Y 轴 = 损失值
- **意义**：选择 MSE 还是 MAE，是在决定建一个 **U 型抛物线**（碗）还是 **V 型折线**（漏斗）

### 宏观视角：小球寻找最低点

- **坐标系**：X 轴 = 模型参数 $w$，Y 轴 = 总损失值
- **意义**：把参数想象成一颗小球，梯度下降让它顺着坡度找最低点
- **学习率**：决定小球的步幅——太大飞出山谷（梯度爆炸），太小卡在半山腰

---

## 一、回归任务损失函数

### 1. MSE（均方误差 / L2 Loss）

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**形状**：U 型平滑抛物线

| 优点 | 缺点 |
|---|---|
| 收敛丝滑（越近谷底坡度越小，自动减速） | **对异常值极其敏感**（误差被平方放大） |
| 数学性质好（连续可微，梯度好算） | 一个极端异常值会主导整体损失 |

**使用场景**：数据质量好、异常值少的回归任务。

### 2. MAE（平均绝对误差 / L1 Loss）

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**形状**：V 型折线

| 优点 | 缺点 |
|---|---|
| **对异常值鲁棒**（类似中位数，不放大极端误差） | 梯度在任何位置**恒定**，谷底附近仍高速震荡 |
| 更能反映"典型误差" | 在 0 处不可导，需要特殊处理 |

**使用场景**：数据中存在较多异常值、噪声的回归任务。

### 3. Huber Loss（融合两者的终极形态）

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

**精髓**：引入阈值 $\delta$，分段处理：

```
误差大（远离谷底 / 遇到异常值）→ MAE 模式：匀速下降，不被异常值拖垮
误差小（快到谷底）           → MSE 模式：平滑减速，精准停车
```

**使用场景**：
- 目标检测（YOLO、Faster R-CNN 的边框回归）
- 强化学习中的 Q 值估计
- 一般回归任务的首选替代方案

```python
import torch
import torch.nn as nn

mse = nn.MSELoss()
mae = nn.L1Loss()
huber = nn.HuberLoss(delta=1.0)   # 或 nn.SmoothL1Loss()

y_true = torch.tensor([1.0, 2.0, 3.0, 100.0])  # 包含一个异常值
y_pred = torch.tensor([1.1, 2.1, 3.1, 3.1])

print(f"MSE:   {mse(y_pred, y_true):.4f}")    # 受 100 影响巨大
print(f"MAE:   {mae(y_pred, y_true):.4f}")    # 对 100 更鲁棒
print(f"Huber: {huber(y_pred, y_true):.4f}")  # 折中方案
```

---

## 二、分类任务损失函数

分类任务需要两个核心组件配合：

1. **One-Hot 编码**：提供"标准答案"（如猫 = $[1,0,0]$）
2. **Softmax 函数**：将 Logits 转为概率分布

### 1. 二元交叉熵（BCE / Binary Cross Entropy）

用于**二分类**任务（0 或 1），节省一个输出神经元：

$$L = -[y \ln(\hat{y}) + (1-y) \ln(1-\hat{y})]$$

**直觉**：
- 如果 $y=1$，$\hat{y}$ 越接近 1，$-\ln(\hat{y})$ 越接近 0（损失越小）
- 如果 $y=0$，$\hat{y}$ 越接近 0，$-\ln(1-\hat{y})$ 越接近 0（损失越小）

```python
bce = nn.BCEWithLogitsLoss()  # 内置 Sigmoid + 数值稳定处理

logits = torch.tensor([2.0, -1.0, 0.5])   # 未经 Sigmoid 的原始输出
labels = torch.tensor([1.0, 0.0, 1.0])    # 二值标签

loss = bce(logits, labels)
print(f"BCE Loss: {loss:.4f}")
```

### 2. 多分类交叉熵（CCE / Categorical Cross Entropy）

BCE 的推广，将类别从 2 扩展到 $C$：

$$L = -\sum_{i=1}^{C} y_i \ln(\hat{y}_i)$$

由于 $y$ 是 One-Hot 编码，只有真实类别那一项 $y_i=1$，其余为 0，实际等价于：

$$L = -\ln(\hat{y}_{真实类别})$$

**意义**：最大化模型对正确类别的概率预测。

```python
ce = nn.CrossEntropyLoss()  # PyTorch 内置 Softmax + 数值稳定

logits = torch.tensor([[2.0, 1.0, 0.5],   # 样本1的3类 logits
                        [0.5, 2.5, 0.1]])   # 样本2的3类 logits
labels = torch.tensor([0, 1])              # 样本1真实类=0，样本2真实类=1

loss = ce(logits, labels)
print(f"CE Loss: {loss:.4f}")
```

---

## 三、数值稳定性：避坑指南

### 问题一：指数爆炸（Overflow）

如果 Logits 有极大值（如 1000），$e^{1000}$ = `inf`。

**框架解决方案**：自动减去最大值 $C$（不影响结果，消除溢出）：

$$\text{Softmax}(x_i) = \frac{e^{x_i - C}}{\sum_j e^{x_j - C}}$$

### 问题二：对数陷阱（Underflow）

如果先算 Softmax 再取 log，极小概率被精度限制抹为 0，$\ln(0) = -\infty$（`NaN`）。

**框架解决方案**：LogSoftmax 合并运算，绕过微小概率：

$$\ln\left(\frac{e^{z_i}}{\sum e^{z_j}}\right) = z_i - \ln\left(\sum e^{z_j}\right)$$

**工程黄金法则**：
- **永远直接传 Logits（未经 Softmax/Sigmoid 的原始输出）给框架的损失函数**
- PyTorch：`CrossEntropyLoss`（内置 Softmax）、`BCEWithLogitsLoss`（内置 Sigmoid）
- **不要手动先接 Softmax 再传给 CrossEntropyLoss**，会导致数值不稳定

```python
# ❌ 错误做法
softmax_output = torch.softmax(logits, dim=-1)
loss = nn.NLLLoss()(torch.log(softmax_output), labels)

# ✅ 正确做法（框架内部处理数值稳定性）
loss = nn.CrossEntropyLoss()(logits, labels)
```

---

## 四、选型总结

| 任务类型 | 损失函数 | 输出层 | 关键特点 |
|---|---|---|---|
| **回归**（无异常值） | **MSE** | Linear | 收敛平滑，但怕异常值 |
| **回归**（有异常值） | **MAE** | Linear | 对异常值鲁棒 |
| **回归**（工业首选） | **Huber** | Linear | 大误差 MAE + 小误差 MSE |
| **二分类** | **BCE** | Sigmoid | 输出概率，两类问题 |
| **多分类** | **CCE** | Softmax | 输出概率分布，N 类问题 |
| **多标签分类** | **BCE（多输出）** | 多个 Sigmoid | 每个标签独立二分类 |
