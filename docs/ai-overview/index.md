# AI 知识体系总览

## 什么是人工智能？

**人工智能（Artificial Intelligence, AI）** 是计算机科学的一个分支，致力于创造能够执行通常需要人类智慧才能完成的任务的系统。

AI 不是一种单一技术，而是一个由多个子领域组成的技术体系：

```
人工智能 (AI)
├── 机器学习 (Machine Learning)
│   ├── 监督学习（线性回归、SVM、决策树...）
│   ├── 无监督学习（聚类、降维...）
│   └── 强化学习（Q-Learning、PPO...）
├── 深度学习 (Deep Learning)
│   ├── 神经网络基础（FNN、CNN）
│   ├── 序列模型（RNN、LSTM、GRU）
│   └── Transformer 架构
└── 自然语言处理 (NLP)
    ├── 文本表示（Word2Vec、GloVe）
    ├── 预训练模型（BERT、GPT）
    └── 大语言模型（LLM）
```

## 知识体系结构

本知识库按照从基础到进阶的逻辑顺序组织，共分为以下几个部分：

### 🧮 第一篇：数学基础

AI 的根基是数学。掌握以下数学知识是理解所有算法的必要条件：

| 模块 | 核心内容 |
|------|---------|
| [线性代数](/math-foundations/linear-algebra) | 向量、矩阵、范数、特征值分解 |
| [微积分与优化](/math-foundations/calculus) | 导数、梯度、链式法则、泰勒展开 |
| [向量化计算](/math-foundations/vectorization) | NumPy 矩阵运算加速技巧 |
| [归一化与标准化](/math-foundations/normalization) | Min-Max 归一化、Z-Score 标准化 |

### 🤖 第二篇：机器学习

从传统机器学习算法入手，理解 AI 的基本工作方式：

| 模块 | 核心内容 |
|------|---------|
| [ML 概述](/machine-learning/) | 监督/无监督/强化学习范式 |
| [特征工程](/machine-learning/feature-engineering) | 特征选择、变换、降维 |
| [模型评估](/machine-learning/model-evaluation) | 损失函数、正则化、过拟合 |
| [评估指标](/machine-learning/metrics) | 准确率、精确率、召回率、F1、AUC |

### 📊 第三篇：监督学习算法

| 算法 | 适用场景 |
|------|---------|
| [线性回归](/ml-supervised/linear-regression) | 连续值预测 |
| [逻辑回归](/ml-supervised/logistic-regression) | 二分类问题 |
| [K 近邻 (KNN)](/ml-supervised/knn) | 小数据集分类/回归 |
| [决策树](/ml-supervised/decision-tree) | 可解释性分类 |
| [支持向量机 (SVM)](/ml-supervised/svm) | 高维小样本分类 |
| [朴素贝叶斯](/ml-supervised/naive-bayes) | 文本分类 |
| [感知机](/ml-supervised/perceptron) | 线性可分问题 |

### 🌲 第四篇：集成学习

| 算法 | 适用场景 |
|------|---------|
| [集成学习概述](/ml-ensemble/) | Bagging vs Boosting vs Stacking |
| [随机森林](/ml-ensemble/random-forest) | 通用分类/回归 |
| [梯度提升 GBDT](/ml-ensemble/gbdt) | 高精度预测 |
| [AdaBoost](/ml-ensemble/adaboost) | 自适应提升 |
| [XGBoost](/ml-ensemble/xgboost) | 竞赛神器，表格数据 |
| [LightGBM](/ml-ensemble/lightgbm) | 高效大数据处理 |

### 🔵 第五篇：无监督学习

| 模块 | 核心内容 |
|------|---------|
| [聚类算法](/ml-unsupervised/clustering) | K-Means、层次聚类、DBSCAN |
| [降维技术](/ml-unsupervised/dimensionality-reduction) | PCA、t-SNE、UMAP |

### 🧠 第六篇：深度学习基础

| 模块 | 核心内容 |
|------|---------|
| [深度学习介绍](/deep-learning/) | 什么是深度学习，与 ML 的区别 |
| [激活函数](/deep-learning/activation-functions) | Sigmoid、ReLU、Softmax 的演进史 |
| [损失函数](/deep-learning/loss-functions) | MSE、交叉熵、对比损失 |
| [前向/反向传播](/deep-learning/backpropagation) | 链式法则，计算图 |
| [优化器](/deep-learning/optimizers) | SGD → Adam → AdamW |
| [参数初始化与正则化](/deep-learning/regularization) | Dropout、BatchNorm、权重衰减 |

### 🏗️ 第七篇：神经网络架构

| 架构 | 核心内容 |
|------|---------|
| [前馈神经网络 FNN](/neural-networks/fnn) | 多层感知机基础 |
| [循环神经网络 RNN](/neural-networks/rnn) | 序列建模，时间步 |
| [长短期记忆 LSTM](/neural-networks/lstm) | 门控机制，长距离依赖 |
| [门控循环单元 GRU](/neural-networks/gru) | LSTM 的轻量版 |
| [Seq2Seq](/neural-networks/seq2seq) | 编码器-解码器框架 |

### ⚡ 第八篇：Attention 与 Transformer

| 模块 | 核心内容 |
|------|---------|
| [注意力机制](/attention-transformer/attention) | Q/K/V，加权求和，可视化 |
| [Transformer 架构](/attention-transformer/transformer) | 编码器-解码器，完整数据流 |
| [自注意力与多头注意力](/attention-transformer/self-attention) | 并行化的全局感知 |
| [位置编码](/attention-transformer/positional-encoding) | 为什么 Transformer 需要位置信息 |

### 📝 第九篇：NLP 基础

| 模块 | 核心内容 |
|------|---------|
| [NLP 导论](/nlp-introduction/what-is-nlp) | NLP 定义、任务和挑战 |
| [NLP 发展史](/nlp-introduction/history) | 规则 → 统计 → 深度学习 → 大模型 |
| [文本预处理](/text-preprocessing/tokenization) | 分词、子词、词表构建 |
| [文本表示](/text-representation/word2vec) | Word2Vec、GloVe、FastText |

### 🌟 第十篇：预训练语言模型

| 模型 | 核心内容 |
|------|---------|
| [GPT 系列](/pretrained-models/gpt) | 自回归语言模型，解码器架构 |
| [BERT](/pretrained-models/bert) | 双向编码，Masked LM |
| [BERT 变体](/pretrained-models/bert-variants) | RoBERTa、ALBERT、DistilBERT |

### 🚀 第十一篇：现代 AI 与大语言模型

| 模块 | 核心内容 |
|------|---------|
| [大语言模型 (LLM)](/modern-nlp/llm) | GPT-4、LLaMA、Claude 的技术原理 |
| [提示工程](/modern-nlp/prompt-engineering) | CoT、Few-shot、提示设计最佳实践 |
| [对齐技术](/modern-nlp/alignment) | RLHF、DPO、Constitutional AI |

### 🛠️ 第十二篇：工程实践

| 模块 | 核心内容 |
|------|---------|
| [HuggingFace 生态](/engineering/huggingface) | Transformers、Datasets、Tokenizers |
| [模型微调](/engineering/finetuning) | LoRA、QLoRA、全量微调 |
| [模型部署](/engineering/deployment) | vLLM、ONNX、量化、推理优化 |

---

## 学习路径建议

### 🎯 路径一：AI 入门（无基础）

```
数学基础 → 机器学习概述 → 线性回归 → 逻辑回归 → 决策树 → 
集成学习 → 深度学习基础 → Transformer → 大语言模型
```

**预计时间：** 3-6 个月

### 🎯 路径二：深度学习工程师

```
数学基础（快速回顾）→ 深度学习基础 → 神经网络架构 → 
Transformer → 预训练模型 → 微调 → 工程部署
```

**预计时间：** 2-4 个月

### 🎯 路径三：NLP 专项

```
文本预处理 → 文本表示 → 序列模型 → Attention → Transformer → 
BERT/GPT → 大语言模型 → 提示工程
```

**预计时间：** 2-3 个月

---

> 💡 **学习建议：** 每个章节都包含理论讲解、数学推导和代码实现。建议在阅读理论后，动手实现代码，加深理解。
