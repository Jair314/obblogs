## 1. 题目要求

本项目的目标是完成一个基于 CIFAR-10 数据集的图像分类任务。给定一张 32×32 的彩色图片，模型需要判断该图片属于以下 10 个类别中的哪一类：

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

CIFAR-10 数据集包含 60,000 张 32×32 彩色图片，共 10 个类别，每类 6,000 张，其中 50,000 张用于训练，10,000 张用于测试。这个数据集规模适中、类别清晰、图片尺寸较小，因此非常适合作为 CNN 图像分类任务的入门练习。

本项目同时完成了 Kaggle 提交流程。Kaggle 比赛页面为：**CIFAR-10 - Object Recognition in Images**。该比赛版数据提供 50,000 张训练图片和 300,000 张测试图片，其中 10,000 张测试图片用于实际评分，另外 290,000 张图片主要用于防止人工标注测试集作弊。

本次项目最终完成了以下任务：

```text
1. 使用 PyTorch 加载 CIFAR-10 数据集；
2. 自定义一个 LeNet 风格的 CNN 模型；
3. 完成训练、验证、测试流程；
4. 保存最佳模型 checkpoint；
5. 输出 classification report、confusion matrix 和 ROC 曲线；
6. 使用训练好的模型对 Kaggle test 图片进行预测；
7. 生成 submission.csv 并提交到 Kaggle；
8. 获得 Kaggle 分数：0.63360。
```

---

## 2. 项目整体思路

图像分类任务的基本流程可以概括为：

```text
原始图片
↓
图像预处理
↓
CNN 提取特征
↓
全连接层分类
↓
输出 10 个类别的 logits
↓
CrossEntropyLoss 计算损失
↓
反向传播更新参数
```

在本项目中，输入图片大小为：

```text
3 × 32 × 32
```

其中：

```text
3 表示 RGB 三个颜色通道；
32 × 32 表示图像的高和宽。
```

模型最后输出：

```text
10 个类别对应的 logits
```

例如：

```text
[0.2, -1.3, 2.1, 0.4, ...]
```

这些数值不是概率，而是模型对每个类别的原始打分。训练时使用 `CrossEntropyLoss`，它适合处理多分类问题，可以直接接收 logits 和类别标签。PyTorch 官方文档也说明，`CrossEntropyLoss` 常用于 C 类分类任务，并计算输入 logits 与目标标签之间的交叉熵损失。

---

## 3. 数据预处理

本项目中使用的预处理代码如下：

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), 
                         (0.5, 0.5, 0.5))
])
```

这里包含两个步骤。

第一个是 `ToTensor()`。它会把图片从 PIL Image 或 NumPy 数组转换成 PyTorch Tensor，并把像素值从 0～255 缩放到 0～1。

第二个是 `Normalize()`。这里使用：

```python
mean = (0.5, 0.5, 0.5)
std  = (0.5, 0.5, 0.5)
```

归一化公式为：

```text
x' = (x - mean) / std
```

由于 `ToTensor()` 后像素值范围是 `[0, 1]`，所以：

```text
x' = (x - 0.5) / 0.5
```

会把像素大致变换到：

```text
[-1, 1]
```

这样做的好处是让输入数据分布更稳定，模型训练时梯度更新更平滑。对于第一版 baseline 来说，使用 `(0.5, 0.5, 0.5)` 是一个简单、直观、容易理解的选择。更严格的做法是使用 CIFAR-10 训练集本身的通道均值和标准差，但在入门实验中，当前设置已经足够跑通完整流程。

数据加载部分如下：

```python
train_dataset = datasets.CIFAR10(
    root='./dataset',
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./dataset',
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)
```

其中：

```text
batch_size = 64
```

表示每次送入模型 64 张图片。这个值不算太大，也不算太小，适合入门实验。batch size 太小会导致训练波动较大，太大又可能占用更多显存，并且不一定适合所有设备。

训练集使用：

```python
shuffle=True
```

这是因为训练时需要打乱数据顺序，避免模型按照固定类别或固定顺序学习。

测试集使用：

```python
shuffle=False
```

因为测试时不需要打乱顺序，保持确定性更方便后续分析和复现。

---

## 4. CNN 是如何实现的

本项目使用的是一个 LeNet 风格的简单 CNN。模型代码如下：

```python
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),

            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.net(x)
```

这个模型不是 PyTorch 内置的完整 CNN，而是自己定义的网络结构。PyTorch 提供了 `Conv2d`、`ReLU`、`MaxPool2d`、`Linear` 等基础模块，我们则负责把这些模块按照一定顺序组合起来。`Conv2d` 的作用是对输入图像进行二维卷积，从局部区域中提取特征。

模型整体结构可以写成：

```text
Input
↓
Conv2d(3 → 6, kernel=5)
↓
ReLU
↓
MaxPool2d
↓
Conv2d(6 → 16, kernel=5)
↓
ReLU
↓
MaxPool2d
↓
Flatten
↓
Linear(400 → 120)
↓
ReLU
↓
Linear(120 → 84)
↓
ReLU
↓
Linear(84 → 10)
↓
Output logits
```

---

## 5. 每一层的尺寸变化

输入图片尺寸为：

```text
[batch_size, 3, 32, 32]
```

第一个卷积层：

```python
nn.Conv2d(3, 6, 5)
```

含义是：

```text
输入通道数：3
输出通道数：6
卷积核大小：5×5
```

由于没有设置 padding，卷积后的特征图大小为：

```text
32 - 5 + 1 = 28
```

所以输出尺寸为：

```text
[batch_size, 6, 28, 28]
```

接着经过：

```python
nn.MaxPool2d(2, 2)
```

特征图尺寸减半：

```text
28 × 28 → 14 × 14
```

输出变成：

```text
[batch_size, 6, 14, 14]
```

第二个卷积层：

```python
nn.Conv2d(6, 16, 5)
```

输入特征图大小是 14×14，卷积核大小为 5×5，因此：

```text
14 - 5 + 1 = 10
```

输出为：

```text
[batch_size, 16, 10, 10]
```

再经过一次池化：

```text
10 × 10 → 5 × 5
```

输出为：

```text
[batch_size, 16, 5, 5]
```

随后使用：

```python
nn.Flatten()
```

把每张图片的特征展平成一维向量：

```text
16 × 5 × 5 = 400
```

所以第一个全连接层写成：

```python
nn.Linear(16 * 5 * 5, 120)
```

这也是为什么这里必须写 `16 * 5 * 5`，因为经过两次卷积和两次池化之后，每张图片最终对应 400 个特征值。

---

## 6. 为什么选择 LeNet 风格 CNN

选择 LeNet 风格 CNN 的主要原因不是为了追求最高分，而是为了适合入门理解。

LeNet-5 是早期经典卷积神经网络之一，原始 LeNet-5 架构包含卷积层、池化层和全连接层，是 CNN 发展史中非常基础、非常重要的结构。LeCun 等人的经典论文中详细描述了 LeNet-5，并将 CNN 用于文档识别任务。

本项目选择 LeNet 风格结构，有以下几个原因。

第一，结构简单，容易理解。它由少量卷积层、池化层和全连接层组成，能够清楚展示 CNN 的基本工作方式：

```text
卷积层负责提取局部特征；
池化层负责降低空间尺寸；
全连接层负责根据提取出的特征进行分类。
```

第二，参数量较小，训练速度快。本项目模型大约只有 62,006 个可训练参数。这样的模型很适合初学阶段快速验证流程，不容易因为模型太大而陷入训练时间长、显存不足、调参困难等问题。

第三，适合建立 baseline。第一次做 CIFAR-10 分类时，最重要的不是直接拿到很高准确率，而是先完整跑通：

```text
数据加载
↓
模型定义
↓
训练
↓
验证
↓
测试
↓
保存模型
↓
Kaggle 提交
```

LeNet 风格 CNN 虽然性能不是最强，但非常适合作为 baseline。后续可以在此基础上继续升级，比如加入 BatchNorm、Dropout、数据增强，或者换成 ResNet18、Vision Transformer 等更强模型。

第四，它能帮助理解现代 CNN 的基本思想。虽然 ResNet、DenseNet、EfficientNet 等模型远比 LeNet 复杂，但它们依然建立在卷积、非线性激活、特征提取、下采样和分类这些基本思想之上。因此，先理解 LeNet 风格模型，有助于后续理解更复杂的网络。

---

## 7. 主要参数选择说明

### 7.1 卷积核大小为什么选 5×5

模型中两个卷积层都使用：

```python
kernel_size = 5
```

5×5 卷积核来自经典 LeNet 风格设计。它比 3×3 卷积核感受野更大，每次可以看到更大范围的局部区域。对于 32×32 的小图片，5×5 卷积能够快速提取边缘、纹理、局部形状等基础特征。

不过，现代 CNN 中更常用 3×3 卷积。因为多个 3×3 卷积堆叠起来可以获得较大感受野，同时参数量更可控。因此，如果后续想提高准确率，可以把 5×5 卷积改成多个 3×3 卷积，并加入 BatchNorm。

### 7.2 输出通道为什么是 6 和 16

模型中两个卷积层分别是：

```python
nn.Conv2d(3, 6, 5)
nn.Conv2d(6, 16, 5)
```

第一个卷积层把 RGB 三通道图像映射成 6 个特征图。第二个卷积层再把 6 个特征图映射成 16 个更高级的特征图。

这个设置比较小，主要是为了保持模型轻量。对于 CIFAR-10 来说，6 和 16 的通道数并不强，但足以完成一个入门 baseline。后续如果要提升效果，可以改成：

```python
nn.Conv2d(3, 32, 3, padding=1)
nn.Conv2d(32, 64, 3, padding=1)
nn.Conv2d(64, 128, 3, padding=1)
```

这样模型容量更大，能学习更丰富的图像特征。

### 7.3 为什么使用 ReLU

每个卷积层和前两个全连接层后面都使用了：

```python
nn.ReLU()
```

ReLU 的作用是引入非线性。如果没有 ReLU，那么多层线性变换叠加起来本质上仍然接近一个线性模型，表达能力会很有限。

ReLU 会把负数变成 0，正数保持不变：

```text
ReLU(x) = max(0, x)
```

这样网络可以学习更复杂的非线性模式。

### 7.4 为什么使用 MaxPool2d

模型中使用了两次：

```python
nn.MaxPool2d(2, 2)
```

它的作用是将特征图尺寸减半。例如：

```text
28×28 → 14×14
10×10 → 5×5
```

池化层有两个主要作用：

```text
1. 降低特征图尺寸，减少后续计算量；
2. 保留局部区域中最显著的特征，提高一定的平移鲁棒性。
```

对于入门 CNN 来说，卷积层提取特征、池化层缩小尺寸，是非常经典的组合。

### 7.5 为什么最后输出 10 维

最后一层是：

```python
nn.Linear(84, 10)
```

因为 CIFAR-10 有 10 个类别，所以模型最后需要输出 10 个类别的分数。每个输出值对应一个类别的 logit，值越大表示模型越倾向于认为图片属于该类别。

预测时使用：

```python
_, predicted = torch.max(outputs, 1)
```

它会选择 logit 最大的类别作为最终预测结果。

---

## 8. 损失函数和优化器

本项目使用：

```python
criterion = nn.CrossEntropyLoss()
```

对于多分类任务，`CrossEntropyLoss` 是最常用的损失函数之一。它会比较模型输出的 logits 和真实类别标签，然后计算分类错误程度。PyTorch 文档也指出，它适用于 C 类分类问题。

优化器使用：

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

SGD，即随机梯度下降，是经典的神经网络优化方法。PyTorch 的 `SGD` 支持 momentum 参数，用于在更新时引入动量。

这里选择：

```text
learning rate = 0.01
momentum = 0.9
```

原因是：

```text
1. 0.01 是 CNN 入门实验中比较常见的初始学习率；
2. momentum 可以让参数更新方向更加稳定；
3. 对于小型 CNN，SGD + momentum 通常可以得到比较可靠的 baseline。
```

如果学习率太大，训练可能震荡甚至不收敛；如果学习率太小，训练速度会很慢。`momentum=0.9` 则可以让优化过程利用之前梯度方向的信息，减少来回抖动。

---

## 9. 训练与验证结果

训练过程中，模型会在每个 epoch 后计算验证集准确率，并保存验证集表现最好的模型：

```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    save_checkpoint(...)
```

本次实验中，模型在第 8 个 epoch 左右保存了最佳 checkpoint：

```text
best_model.pth
val_acc = 0.6358
```

随后使用该模型生成 Kaggle 提交文件，并获得：

```text
Kaggle Score: 0.63360
Private score: 0.63360
```

这个结果与本地验证集准确率非常接近，说明本地验证流程和 Kaggle 线上评分结果基本一致。

对于该 LeNet 风格小模型来说，约 63% 的 CIFAR-10 准确率是比较合理的。因为该模型结构较浅、通道数较少，表达能力有限。它适合作为 baseline，但不是高性能模型。

---

## 10. Kaggle 提交流程

本项目使用训练好的 `best_model.pth` 对 Kaggle test 文件夹中的图片进行预测，生成：

```text
submissions/submission.csv
```

提交文件格式如下：

```csv
id,label
1,cat
2,ship
3,ship
4,airplane
5,deer
```

其中：

```text
id：Kaggle test 图片编号；
label：模型预测的类别名称。
```

需要注意的是，Kaggle 比赛版 CIFAR-10 的 test 文件夹有 300,000 张图片，而不是 torchvision 自带测试集中的 10,000 张图片。D2L 的 Kaggle CIFAR-10 教程也说明，该比赛版 test set 有 300,000 张图片，其中只有 10,000 张用于评分。

因此，不能直接把 torchvision 的 test set 预测结果提交到 Kaggle。正确做法是：

```text
1. 下载 Kaggle 比赛数据；
2. 解压 test 图片；
3. 加载训练好的模型；
4. 对 Kaggle test 图片逐张预测；
5. 生成 submission.csv；
6. 上传到 Kaggle Submit Predictions。
```

---

## 11. 本项目的意义

这个项目虽然模型不复杂，但完整覆盖了深度学习图像分类任务的基本流程。通过这个项目，我理解了：

```text
1. CIFAR-10 图像分类任务的基本形式；
2. PyTorch 中 Dataset 和 DataLoader 的使用方法；
3. CNN 中卷积层、池化层、激活函数和全连接层的作用；
4. 输入图像在网络中的尺寸变化；
5. CrossEntropyLoss 和 SGD optimizer 的基本用法；
6. checkpoint 保存和加载方式；
7. 本地测试与 Kaggle 在线提交之间的区别；
8. 如何生成符合比赛要求的 submission.csv。
```

更重要的是，这个项目完成了从本地训练到 Kaggle 提交的完整闭环。相比只在本地跑出 accuracy，把结果提交到 Kaggle 可以检查预测文件格式、线上评分流程和模型泛化效果，是一个更接近真实机器学习任务的练习方式。

---

## 12. 后续改进方向

当前模型是一个基础 baseline。后续可以从以下几个方向改进：

```text
1. 加入数据增强：
   RandomCrop、RandomHorizontalFlip 等；

2. 提升 CNN 模型容量：
   把通道数从 6/16 提升到 32/64/128；

3. 加入 BatchNorm：
   让训练更稳定，收敛更快；

4. 加入 Dropout：
   缓解过拟合；

5. 使用学习率调度器：
   如 CosineAnnealingLR 或 StepLR；

6. 使用 ResNet18：
   建立更强的 CNN baseline；

7. 使用 MiniViT：
   对比 Transformer 和 CNN 在 CIFAR-10 上的表现；

8. 记录不同模型的 Kaggle 分数：
   形成完整实验对比。
```

可以将后续实验整理成如下版本记录：

```text
v1: LeNet-style CNN baseline
    Kaggle score: 0.63360

v2: BetterCNN + Data Augmentation

v3: CIFAR-ResNet18

v4: Mini Vision Transformer
```

这样，这个项目就不仅是一次简单实验，而是一个可以持续迭代的图像分类学习仓库。

---

## 13. 总结

本次 CIFAR-10 项目使用 PyTorch 实现了一个 LeNet 风格 CNN，并完成了训练、验证、测试和 Kaggle 提交。虽然最终 Kaggle 分数为 0.63360，不属于高分模型，但它成功建立了一个清晰、可复现、可扩展的 baseline。

选择 LeNet 风格 CNN 的主要原因是它结构简单、逻辑清楚、训练速度快，非常适合用于理解 CNN 的基本原理。通过这个项目，可以从实践中理解图像分类任务的完整流程，并为后续学习 ResNet、Vision Transformer、CIFAR-100 分类等更复杂任务打下基础。