# 深度学习

> 深度学习是以**多层神经网络**为核心的机器学习方法：让模型自动从海量数据中逐层学习特征表示，在视觉、语言、语音等领域超越人类专家级性能。

## 什么是深度学习？

深度学习（Deep Learning）的"深度"指网络中**隐藏层的数量**。

```
输入层 → 隐藏层1 → 隐藏层2 → ... → 隐藏层N → 输出层
(原始像素)  (边缘)    (纹理)           (语义)    (类别)
```

与传统机器学习的本质区别：

| 维度 | 传统机器学习 | 深度学习 |
|---|---|---|
| **特征提取** | 手工设计特征 | **自动学习特征** |
| **数据需求** | 中等（千级） | 大（百万级+） |
| **计算需求** | CPU 即可 | GPU/TPU 必要 |
| **可解释性** | 较好 | 差（黑盒） |
| **性能上限** | 有瓶颈 | 数据量越大越强 |

## 深度学习的多种学习范式

| 范式 | 说明 | 代表模型 |
|---|---|---|
| **监督深度学习** | 带标签数据，有"标准答案" | CNN、FNN |
| **无监督深度学习** | 无标签数据，自发现结构 | GAN、VAE、自编码器 |
| **半监督深度学习** | 少量标签+大量无标签数据 | SimCLR、GPT 预训练 |
| **深度强化学习** | 与环境交互，最大化奖励 | DQN、AlphaGo |

## 常见架构

| 架构 | 擅长 | 代表 |
|---|---|---|
| **FNN/MLP** | 表格数据、基础分类 | 全连接网络 |
| **CNN** | 图像、视频 | ResNet、EfficientNet |
| **RNN/LSTM/GRU** | 序列数据、时间序列 | 情感分析、机器翻译 |
| **Transformer** | 语言、多模态 | BERT、GPT、ViT |
| **GAN** | 图像生成、数据增强 | StyleGAN、Pix2Pix |
| **扩散模型** | 高质量生成 | Stable Diffusion、DALL-E |

## 本章目录

| 主题 | 内容 |
|---|---|
| [什么是深度学习](/deep-learning/introduction) | 定义、原理、历史沿革、学习范式 |
| [激活函数](/deep-learning/activation-functions) | 从 Step 到 SwiGLU 的演进史 |
| [损失函数](/deep-learning/loss-functions) | MSE、MAE、Huber、交叉熵的选择指南 |
| [前向与反向传播](/deep-learning/backpropagation) | 完整数学推导，链式法则，梯度流 |
| [优化器](/deep-learning/optimizers) | SGD → Adam → AdamW 演进全解 |
| [正则化与初始化](/deep-learning/regularization) | Xavier/He 初始化，Dropout，BatchNorm，L2 |

## 深度学习历史里程碑

| 时间 | 事件 |
|---|---|
| 1957 | 感知机（Perceptron）提出 |
| 1986 | 反向传播算法（Backpropagation）普及 |
| 2006 | Hinton 提出深度信念网络，深度学习复兴 |
| 2012 | AlexNet 赢得 ImageNet，深度学习进入实用时代 |
| 2014 | GAN（生成对抗网络）提出 |
| 2015 | ResNet（152层）突破梯度消失难题 |
| 2017 | Transformer 架构提出（"Attention Is All You Need"） |
| 2018 | BERT 预训练语言模型革新 NLP |
| 2020 | GPT-3（1750亿参数）刷新语言生成上限 |
| 2022 | ChatGPT 大语言模型走进大众视野 |
| 2023+ | 多模态大模型、扩散模型、AI Agent 时代 |

## 学习建议

1. **先打数学基础**：线性代数（矩阵）+ 微积分（链式法则）+ 概率论
2. **理解关键组件**：激活函数 → 损失函数 → 反向传播 → 优化器
3. **动手实践**：用 PyTorch 实现一个完整的训练循环
4. **从 FNN 到复杂模型**：FNN → CNN → RNN → LSTM → Transformer
