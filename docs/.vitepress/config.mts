import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI 知识体系',
  description: '从机器学习到大语言模型，系统化的人工智能学习笔记',
  lang: 'zh-CN',
  base: '/nlp-tutorial/',
  lastUpdated: true,
  cleanUrls: true,

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
  ],

  markdown: {
    math: true,
  },

  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => tag.startsWith('mjx-'),
      },
    },
  },

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'AI 知识体系',

    nav: [
      { text: '首页', link: '/' },
      {
        text: '数学基础',
        items: [
          { text: '总览', link: '/math-foundations/' },
          { text: '线性代数', link: '/math-foundations/linear-algebra' },
          { text: '微积分与梯度', link: '/math-foundations/calculus' },
          { text: '向量化编程', link: '/math-foundations/vectorization' },
          { text: '归一化与标准化', link: '/math-foundations/normalization' },
        ]
      },
      {
        text: '机器学习',
        items: [
          {
            text: '机器学习基础',
            items: [
              { text: '总览', link: '/machine-learning/' },
              { text: '特征工程', link: '/machine-learning/feature-engineering' },
              { text: '模型评估方法', link: '/machine-learning/model-evaluation' },
              { text: '评估指标', link: '/machine-learning/metrics' },
            ]
          },
          {
            text: '监督学习',
            items: [
              { text: '总览', link: '/ml-supervised/' },
              { text: '线性回归', link: '/ml-supervised/linear-regression' },
              { text: '逻辑回归', link: '/ml-supervised/logistic-regression' },
              { text: 'K 近邻', link: '/ml-supervised/knn' },
              { text: '决策树', link: '/ml-supervised/decision-tree' },
              { text: 'SVM', link: '/ml-supervised/svm' },
              { text: '朴素贝叶斯', link: '/ml-supervised/naive-bayes' },
              { text: '感知机', link: '/ml-supervised/perceptron' },
            ]
          },
          {
            text: '集成学习',
            items: [
              { text: '总览', link: '/ml-ensemble/' },
              { text: '随机森林', link: '/ml-ensemble/random-forest' },
              { text: 'GBDT', link: '/ml-ensemble/gbdt' },
              { text: 'AdaBoost', link: '/ml-ensemble/adaboost' },
              { text: 'XGBoost', link: '/ml-ensemble/xgboost' },
              { text: 'LightGBM', link: '/ml-ensemble/lightgbm' },
            ]
          },
          {
            text: '无监督学习',
            items: [
              { text: '总览', link: '/ml-unsupervised/' },
              { text: '聚类算法', link: '/ml-unsupervised/clustering' },
              { text: '降维算法', link: '/ml-unsupervised/dimensionality-reduction' },
            ]
          },
        ]
      },
      {
        text: '深度学习',
        items: [
          {
            text: '深度学习基础',
            items: [
              { text: '总览', link: '/deep-learning/' },
              { text: '什么是深度学习', link: '/deep-learning/introduction' },
              { text: '激活函数', link: '/deep-learning/activation-functions' },
              { text: '损失函数', link: '/deep-learning/loss-functions' },
              { text: '前向与反向传播', link: '/deep-learning/backpropagation' },
              { text: '优化器', link: '/deep-learning/optimizers' },
              { text: '参数初始化与正则化', link: '/deep-learning/regularization' },
            ]
          },
          {
            text: '神经网络架构',
            items: [
              { text: '总览', link: '/neural-networks/' },
              { text: 'FNN 前馈网络', link: '/neural-networks/fnn' },
              { text: 'RNN 循环网络', link: '/neural-networks/rnn' },
              { text: 'LSTM', link: '/neural-networks/lstm' },
              { text: 'GRU', link: '/neural-networks/gru' },
              { text: 'Seq2Seq', link: '/neural-networks/seq2seq' },
            ]
          },
        ]
      },
      {
        text: 'NLP',
        items: [
          {
            text: 'NLP 基础',
            items: [
              { text: 'NLP 导论', link: '/nlp-introduction/what-is-nlp' },
              { text: 'NLP 发展史', link: '/nlp-introduction/history' },
            ]
          },
          {
            text: '文本处理',
            items: [
              { text: '分词技术', link: '/text-preprocessing/tokenization' },
              { text: '子词分词算法', link: '/text-preprocessing/subword' },
              { text: '词表与编码', link: '/text-preprocessing/vocabulary' },
              { text: 'One-hot 与词袋', link: '/text-representation/one-hot-bow' },
              { text: 'TF-IDF 与 N-gram', link: '/text-representation/tfidf-ngram' },
              { text: 'Word2Vec', link: '/text-representation/word2vec' },
              { text: 'GloVe 与 FastText', link: '/text-representation/glove-fasttext' },
            ]
          },
          {
            text: '注意力与预训练',
            items: [
              { text: '注意力机制', link: '/attention-transformer/attention' },
              { text: 'Transformer', link: '/attention-transformer/transformer' },
              { text: 'BERT', link: '/pretrained-models/bert' },
              { text: 'GPT 系列', link: '/pretrained-models/gpt' },
              { text: '大语言模型', link: '/modern-nlp/llm' },
            ]
          },
        ]
      },
    ],

    sidebar: {
      '/ai-overview/': [
        {
          text: 'AI 知识体系总览',
          items: [
            { text: '学习路线图', link: '/ai-overview/' },
          ]
        }
      ],

      '/math-foundations/': [
        {
          text: '数学基础',
          collapsed: false,
          items: [
            { text: '总览', link: '/math-foundations/' },
            { text: '线性代数', link: '/math-foundations/linear-algebra' },
            { text: '微积分与梯度', link: '/math-foundations/calculus' },
            { text: '向量化编程', link: '/math-foundations/vectorization' },
            { text: '归一化与标准化', link: '/math-foundations/normalization' },
          ]
        }
      ],

      '/machine-learning/': [
        {
          text: '机器学习基础',
          collapsed: false,
          items: [
            { text: '总览', link: '/machine-learning/' },
            { text: '特征工程', link: '/machine-learning/feature-engineering' },
            { text: '模型评估方法', link: '/machine-learning/model-evaluation' },
            { text: '评估指标大全', link: '/machine-learning/metrics' },
          ]
        }
      ],

      '/ml-supervised/': [
        {
          text: '监督学习',
          collapsed: false,
          items: [
            { text: '总览', link: '/ml-supervised/' },
            { text: '线性回归', link: '/ml-supervised/linear-regression' },
            { text: '逻辑回归', link: '/ml-supervised/logistic-regression' },
            { text: 'K 近邻 KNN', link: '/ml-supervised/knn' },
            { text: '决策树', link: '/ml-supervised/decision-tree' },
            { text: '支持向量机 SVM', link: '/ml-supervised/svm' },
            { text: '朴素贝叶斯', link: '/ml-supervised/naive-bayes' },
            { text: '感知机', link: '/ml-supervised/perceptron' },
          ]
        }
      ],

      '/ml-ensemble/': [
        {
          text: '集成学习',
          collapsed: false,
          items: [
            { text: '总览', link: '/ml-ensemble/' },
            { text: '随机森林', link: '/ml-ensemble/random-forest' },
            { text: '梯度提升 GBDT', link: '/ml-ensemble/gbdt' },
            { text: 'AdaBoost', link: '/ml-ensemble/adaboost' },
            { text: 'XGBoost', link: '/ml-ensemble/xgboost' },
            { text: 'LightGBM', link: '/ml-ensemble/lightgbm' },
          ]
        }
      ],

      '/ml-unsupervised/': [
        {
          text: '无监督学习',
          collapsed: false,
          items: [
            { text: '总览', link: '/ml-unsupervised/' },
            { text: '聚类算法', link: '/ml-unsupervised/clustering' },
            { text: '降维算法 PCA & t-SNE', link: '/ml-unsupervised/dimensionality-reduction' },
          ]
        }
      ],

      '/deep-learning/': [
        {
          text: '深度学习基础',
          collapsed: false,
          items: [
            { text: '总览', link: '/deep-learning/' },
            { text: '什么是深度学习', link: '/deep-learning/introduction' },
            { text: '激活函数演进史', link: '/deep-learning/activation-functions' },
            { text: '损失函数选型', link: '/deep-learning/loss-functions' },
            { text: '前向与反向传播', link: '/deep-learning/backpropagation' },
            { text: '优化器 SGD→AdamW', link: '/deep-learning/optimizers' },
            { text: '参数初始化与正则化', link: '/deep-learning/regularization' },
          ]
        }
      ],

      '/neural-networks/': [
        {
          text: '神经网络架构',
          collapsed: false,
          items: [
            { text: '总览', link: '/neural-networks/' },
            { text: 'FNN 前馈网络', link: '/neural-networks/fnn' },
            { text: 'RNN 循环网络', link: '/neural-networks/rnn' },
            { text: 'LSTM 长短期记忆', link: '/neural-networks/lstm' },
            { text: 'GRU 门控循环单元', link: '/neural-networks/gru' },
            { text: 'Seq2Seq 序列到序列', link: '/neural-networks/seq2seq' },
          ]
        }
      ],

      '/nlp-introduction/': [
        {
          text: 'NLP 基础',
          collapsed: false,
          items: [
            { text: 'NLP 导论', link: '/nlp-introduction/what-is-nlp' },
            { text: 'NLP 发展史', link: '/nlp-introduction/history' },
            { text: '数学基础（旧）', link: '/nlp-introduction/math-foundations' },
          ]
        }
      ],

      '/text-preprocessing/': [
        {
          text: '文本预处理',
          collapsed: false,
          items: [
            { text: '分词技术', link: '/text-preprocessing/tokenization' },
            { text: '子词分词算法', link: '/text-preprocessing/subword' },
            { text: '词表与编码', link: '/text-preprocessing/vocabulary' },
          ]
        }
      ],

      '/text-representation/': [
        {
          text: '文本表示',
          collapsed: false,
          items: [
            { text: 'One-hot 与词袋', link: '/text-representation/one-hot-bow' },
            { text: 'TF-IDF 与 N-gram', link: '/text-representation/tfidf-ngram' },
            { text: 'Word2Vec', link: '/text-representation/word2vec' },
            { text: 'GloVe 与 FastText', link: '/text-representation/glove-fasttext' },
          ]
        }
      ],

      '/attention-transformer/': [
        {
          text: '注意力与 Transformer',
          collapsed: false,
          items: [
            { text: '注意力机制', link: '/attention-transformer/attention' },
            { text: 'Transformer 架构', link: '/attention-transformer/transformer' },
            { text: '自注意力与多头注意力', link: '/attention-transformer/self-attention' },
            { text: '位置编码', link: '/attention-transformer/positional-encoding' },
          ]
        }
      ],

      '/pretrained-models/': [
        {
          text: '预训练语言模型',
          collapsed: false,
          items: [
            { text: 'ELMo', link: '/pretrained-models/elmo' },
            { text: 'GPT 系列', link: '/pretrained-models/gpt' },
            { text: 'BERT', link: '/pretrained-models/bert' },
            { text: 'BERT 变体', link: '/pretrained-models/bert-variants' },
            { text: '其他预训练模型', link: '/pretrained-models/other-models' },
          ]
        }
      ],

      '/modern-nlp/': [
        {
          text: '现代 NLP 与 LLM',
          collapsed: false,
          items: [
            { text: '大语言模型原理', link: '/modern-nlp/llm' },
            { text: '提示工程', link: '/modern-nlp/prompt-engineering' },
            { text: '人类对齐 RLHF & DPO', link: '/modern-nlp/alignment' },
          ]
        }
      ],

      '/engineering/': [
        {
          text: '工程实践',
          collapsed: false,
          items: [
            { text: 'HuggingFace 生态', link: '/engineering/huggingface' },
            { text: '模型微调', link: '/engineering/finetuning' },
            { text: '模型部署', link: '/engineering/deployment' },
          ]
        }
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/leonyangdev/nlp-tutorial' }
    ],

    footer: {
      message: 'AI 知识体系 — 从机器学习到大语言模型',
      copyright: 'Copyright 2025'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },

    outline: {
      level: [2, 3],
      label: '页面导航'
    },

    lastUpdated: {
      text: '最后更新于'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    }
  }
})
