## 问题背景

本项目的目标是完成一个基于 ``CIFAR-10`` 数据集的图像分类任务。给定一张 $\large 32×32$ 的彩色图片，模型需要判断该图片属于以下 $\large 10$ 个类别中的哪一类：

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

>CIFAR-10 数据集包含 $\large 60,000$ 张 $\large 32×32$ 彩色图片，共 $\large 10$ 个类别，每类 $\large 6,000$ 张，其中 $\large 50,000$ 张用于训练，$\large 10,000$ 张用于测试。这个数据集规模适中、类别清晰、图片尺寸较小，因此非常适合作为 ``CNN`` 图像分类任务的入门练习。

形式化的模型：

1. 给模型输入一张图片：
$$
\displaystyle \large x \in \mathbb R ^{3\times 32 \times 32}
$$
2. 模型输出对相应类别的预测倾向：
$$
\displaystyle \large z = [z_0, z_1, \dots , z_9]
$$
3. 经过 ``softmax`` 后，可以将这些输出转换为概率：
$$
\displaystyle \large p_i = \frac {e^{z_i}} {\sum_{j=0}^{9}e^{z_j}}
$$
4. 模型选择概率最大的类别作为最终预测结果：
$$
\displaystyle \large \hat y = \arg \max_i p_i
$$
---
## 数据集

>[CIFAR-10 - Object Recognition in Images | Kaggle](https://www.kaggle.com/competitions/cifar-10/)，一共包含 $\large 60,000$ 张 RGB 彩色图片。
>
>其中每张图片的尺寸都是 $\large 32 \times 32$，每张图片有 $\large 3$ 个颜色通道，图片的原始形状为 $\large 32\times 32 \times 3$.

在 ``CIFAR-10`` 中，每个类别共有 $\large \dfrac {60000} {10} = 6000$ 张图片，其中 $\large 5,000$ 张用于训练，$\large 1,000$ 张用于测试。

在 ``PyTorch`` 中，图像张量通常采用 $(C, H, W)$ 的格式，所以当选择 ``batch_size=64`` 时，一个批次的图片进入模型的数据形状为 ``[64, 3, 32, 32]``.

---
```mermaid
flowchart TD
    A[CIFAR-10 数据集] --> B[下载并读取数据]
    B --> C[图像预处理]

    C --> C1[训练集数据增强]
    C --> C2[验证集与测试集基础预处理]

    C1 --> D[划分训练集和验证集]
    C2 --> E[测试集]

    D --> D1[训练集 45000 张]
    D --> D2[验证集 5000 张]

    D1 --> F1[Train DataLoader]
    D2 --> F2[Validation DataLoader]
    E --> F3[Test DataLoader]

    F1 --> G[LeNet 模型]
    G --> H[前向传播]
    H --> I[计算交叉熵损失]
    I --> J[反向传播]
    J --> K[优化器更新参数]
    K --> L{当前 Epoch 是否结束}

    L -- 否 --> F1
    L -- 是 --> M[在验证集上评估]

    F2 --> M
    M --> N{验证准确率是否提高}

    N -- 是 --> O[保存最佳模型权重]
    N -- 否 --> P[继续下一轮训练]

    O --> P
    P --> Q{是否达到最大 Epoch}

    Q -- 否 --> F1
    Q -- 是 --> R[加载最佳模型]

    R --> S[在测试集上评估]
    F3 --> S

    S --> T[计算测试准确率]
    T --> U[绘制混淆矩阵]
    U --> V[分析错误分类样本]
    V --> W[整理实验结果与学习笔记]
```
