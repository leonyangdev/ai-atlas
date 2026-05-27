# 深度学习优化器

> 如果损失函数定义了山谷的地形，那么**优化器**就是决定参数这颗"小球"如何安全、快速地滚到谷底的导航系统。

## 优化器演进主线

```
盲目下山          → 增加惯性           → 感知地形         → 修复数学 Bug
Vanilla SGD    →  SGD + Momentum   →  RMSprop + Adam  →  AdamW
```

## 核心变量字典

| 符号 | 含义 |
|---|---|
| $w_t$ | 第 $t$ 步的模型参数（小球当前位置） |
| $g_t$ | 第 $t$ 步的梯度（当前脚下的坡度） |
| $\alpha$ | 基础学习率（系统基础步幅） |
| $m_t$ | 一阶矩（历史梯度加权平均 → **动量/惯性**） |
| $v_t$ | 二阶矩（历史梯度平方加权平均 → **地形崎岖度**） |
| $\lambda$ | 权重衰减系数（防过拟合的"财产税率"） |
| $\beta_1, \beta_2$ | 指数衰减率（通常 0.9 和 0.999） |
| $\epsilon$ | 数值稳定项（通常 $10^{-8}$，防止除以 0） |

---

## 阶段一：SGD（随机梯度下降）

**比喻**：蒙着眼的盲人，每步只探当前坡度，哪边坡就往哪边迈固定步幅。

### 公式

$$w_t = w_{t-1} - \alpha \cdot g_t$$

### 优缺点

| 优点 | 缺点 |
|---|---|
| 容易从"尖锐局部最小值（窄坑）"弹出 | 遇到鞍点（梯度为 0 的平地）直接停滞 |
| **泛化能力极强**（倾向于找宽广平缓盆地） | 在 V 型山谷中两壁横跳，收敛极慢 |

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)
```

---

## 阶段二：SGD + Momentum（带动量）

**比喻**：给盲人装上了"铁球"。下山不仅看当前坡度，还有历史积累的惯性。

### 公式

历史动量（指数移动平均 EMA）：

$$m_t = \beta \cdot m_{t-1} + (1 - \beta) \cdot g_t \quad (\beta = 0.9)$$

参数更新：

$$w_t = w_{t-1} - \alpha \cdot m_t$$

### 工作原理

| 场景 | 效果 |
|---|---|
| 平地（$g_t = 0$） | 凭借 90% 历史惯性继续前进，冲过鞍点 |
| V 型谷横跳 | 左右梯度正负抵消，强行把小球按在谷底向前滚 |

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 阶段三：RMSprop（自适应学习率）

**比喻**：给车装上了**独立悬挂系统**——坡陡的维度减速，坡平的维度加速。

### 公式

跟踪梯度平方的移动平均（地形崎岖度）：

$$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t^2$$

自适应更新：

$$w_t = w_{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon} \cdot g_t$$

### 工作原理

| 地形 | $v_t$ | 步幅 $\alpha / \sqrt{v_t}$ | 效果 |
|---|---|---|---|
| 坡度剧烈（震荡维度） | 大 | 小 | 自动踩刹车，防梯度爆炸 |
| 坡度平缓（停滞维度） | 小 | 大 | 自动踩油门，冲出平原 |

```python
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
```

---

## 阶段四：Adam（万能之王）

**比喻**：**动量（方向盘） + RMSprop（智能悬挂）** 的终极合体。

### 公式

计算方向与惯性（动量）：

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

计算地形与阻力（二阶矩）：

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

偏差修正（初始步骤一阶矩接近 0，需要修正）：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

终极更新：

$$w_t = w_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

**默认超参数**：$\alpha=0.001$，$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$

### 优缺点

| 优点 | 缺点 |
|---|---|
| 开箱即用，极少需要调参 | 容易钻进"尖锐极小值"，导致过拟合 |
| 初期收敛极快 | **权重衰减实现有数学 Bug**（见下文） |

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
```

### Adam 的权重衰减 Bug

传统 Adam 在损失函数中加 L2 正则：

$$\mathcal{L}' = \mathcal{L} + \frac{\lambda}{2}\|w\|^2$$

梯度变为 $g_t + \lambda w_t$，代入 Adam 更新：

$$w_t = w_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot (g_t + \lambda w_{t-1})$$

**问题**：惩罚项 $\lambda w$ 也被 $\sqrt{\hat{v}_t}$ 除！梯度震荡剧烈的参数（分母大）**少交税**，老实的参数（分母小）**多交税**——这完全颠倒了惩罚的初衷。

---

## 阶段五：AdamW（大模型时代真神）

**核心改进**：将"防过拟合惩罚"与"下山步幅"**彻底解耦**。

### 公式

$$w_t = w_{t-1} - \underbrace{\frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}_{\text{Adam 负责下山}} - \underbrace{\alpha \lambda \cdot w_{t-1}}_{\text{独立负责权重衰减（公平收税）}}$$

**区别**：权重衰减项 $\alpha \lambda w_{t-1}$ 不再被 $\sqrt{\hat{v}_t}$ 除，所有参数无论振荡大小，都按相同比例衰减。

```python
# ❌ Adam + L2 正则（有 Bug）
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

# ✅ AdamW（解耦权重衰减，正确实现）
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

**AdamW 保留了 Adam 的极速收敛，同时找回了 SGD 的优异泛化能力。它是当今所有大语言模型（LLM）和 Transformer 架构的绝对标配。**

---

## 学习率调度

固定学习率不够好——训练初期需要大步幅探索，后期需要小步幅精调。

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 余弦退火（LLM 训练常用）
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 或者 OneCycleLR（超收敛）
scheduler = OneCycleLR(optimizer, max_lr=1e-3, 
                       steps_per_epoch=len(train_loader), epochs=50)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()   # 每步/每轮更新学习率
```

---

## 完整对比可视化

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# 对比不同优化器在 Rosenbrock 函数上的轨迹
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# 绘制损失等高线
x = np.linspace(-2, 2, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, np.log(Z + 1), levels=50, cmap='viridis')
plt.colorbar(label='log(Loss + 1)')
plt.scatter([1], [1], c='red', s=200, zorder=5, label='全局最优(1,1)')
plt.title('Rosenbrock 函数（优化器测试场）')
plt.xlabel('w1'); plt.ylabel('w2')
plt.legend(); plt.show()
```

---

## 选型指南

| 优化器 | 特点 | 最适合场景 |
|---|---|---|
| **SGD + Momentum** | 慢但泛化极强，倾向于宽广平缓最优点 | 计算机视觉（CV），图像分类 |
| **Adam** | 快速收敛，自适应，但可能过拟合 | 快速跑通基线，一般回归/分类任务 |
| **AdamW** | 修复 Adam 泛化 Bug，收敛快+泛化好 | **NLP、Transformer、LLM 首选** |
| **LARS/LAMB** | 超大批量训练优化 | 分布式训练，批量 > 8192 |
| **Lion** | Google 2023 新优化器，内存更少 | 资源受限的大模型训练 |
