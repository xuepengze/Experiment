# **CIFAR10 物体识别报告**

物体识别代码的主要思路是利用PyTorch深度学习框架，构建并训练一个ResNet-18模型，在CIFAR-10数据集上执行图像分类任务。整个流程包括数据准备、模型构建、训练过程和模型评估四个主要部分。

## **1.导入库和设置参数**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
import os

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置，使得我们能够手动输入命令行参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
args = parser.parse_args()
```

​        **导入必要的库**：包括 PyTorch、Torchvision、argparse 以及自定义的 ResNet18 模型。

​	**设置设备**：使用 torch.device 选择运行代码的设备，优先使用 GPU。

​	**命令行参数解析**：使用 argparse 解析命令行参数，允许用户指定模型和日志的保存路径。

## **2.设置超参数**

```python
# 超参数设置
EPOCH = 135          # 遍历数据集次数
pre_epoch = 0        # 定义已经遍历数据集的次数
BATCH_SIZE = 128     # 批处理尺寸
LR = 0.01            # 学习率
```

​	**定义训练过程中的超参数**：

​	•EPOCH：总共的训练轮数。

​	•pre_epoch：已经完成的训练轮数，方便中断后继续训练。

​	•BATCH_SIZE：每个批次处理的样本数量。

​	•LR：学习率，控制模型参数更新的步长。

首先，在数据准备阶段，代码导入了必要的库，包括torch、torchvision和argparse等。其中，torch和torchvision用于深度学习模型的构建和数据处理，argparse用于解析命令行参数，以便灵活地指定模型和结果的保存路径。接着，代码设置了计算设备，根据CUDA是否可用，选择使用GPU还是CPU进行训练，并定义了训练所需的超参数，如训练轮数EPOCH、批处理大小BATCH_SIZE和学习率LR等。

## **3.准备数据集和数据预处理**

```python
# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充4个像素，再随机裁剪成32x32
    transforms.RandomHorizontalFlip(),     # 以50%的概率水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),  # 对R,G,B通道进行标准化
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)  # 加载训练集
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)  # 生成批处理数据

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)  # 加载测试集
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
```

**数据预处理**：

​	•**训练集增强**：

​	•RandomCrop：随机裁剪，增强模型的鲁棒性。

​	•RandomHorizontalFlip：随机水平翻转。

​	•Normalize：标准化数据，提高训练效果。

​	•**测试集预处理**：

​	•仅进行 ToTensor 和 Normalize 操作。

​	•**加载数据集**：

​	•使用 torchvision.datasets.CIFAR10 加载 CIFAR-10 数据集。

​	•使用 DataLoader 创建数据加载器，方便批量处理。

在数据预处理方面，代码对训练集和测试集分别进行了处理。训练集的预处理包括随机裁剪、随机水平翻转和归一化操作，旨在通过数据增强提高模型的泛化能力。测试集则仅进行了归一化处理，以保证评估结果的客观性。随后，使用torchvision.datasets.CIFAR10加载CIFAR-10数据集，并通过DataLoader创建数据加载器trainloader和testloader，以便在训练和测试过程中以批次的方式迭代数据

## 4.**定义模型、损失函数和优化器**

```python
# CIFAR-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义 - ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # 使用带动量的SGD优化器，含L2正则化
```

**模型定义**：

​	•使用自定义的 ResNet18 模型，适合处理图像分类任务。

​	•将模型移动到指定设备上（CPU 或 GPU）。

​	•**定义损失函数**：

​	•使用 CrossEntropyLoss，适用于多分类问题。

​	•**定义优化器**：

​	•使用带有动量的 SGD 优化器，加速收敛。

​	•添加 weight_decay 进行 L2 正则化，防止过拟

模型构建部分，代码采用了自定义的ResNet18模型，这是一个深度残差网络，能够有效地训练更深层的神经网络，缓解梯度消失问题。模型被实例化并移动到指定的设备上（GPU或CPU）。损失函数采用了交叉熵损失nn.CrossEntropyLoss()，这是多分类问题的常用选择。优化器使用了带动量的随机梯度下降optim.SGD()，并添加了权重衰减（L2正则化）来防止过拟合，提高模型的泛化性能。

## 5.**训练模型**

```python
# 训练
if __name__ == "__main__":  # 入口函数：确保只有在直接运行脚本时才会执行训练代码
    if not os.path.exists(args.outf):  # 模型保存路径检查：如果不存在则创建目录
        os.makedirs(args.outf)
    best_acc = 85  # 初始化最佳测试准确率，用于保存最佳模型
    print("Start Training, Resnet-18!")  # 打印训练开始的信息
    with open("acc.txt", "w") as f:  # 打开日志文件 acc.txt，用于记录每个 epoch 的准确率
        with open("log.txt", "w") as f2:  # 打开日志文件 log.txt，用于记录训练过程中的损失和准确率
            for epoch in range(pre_epoch, EPOCH):  # 开始训练循环，遍历每个 epoch
                print('\nEpoch: %d' % (epoch + 1))  # 打印当前的 epoch 数
                net.train()  # 设置模型为训练模式
                sum_loss = 0.0  # 初始化累积损失为 0
                correct = 0.0  # 初始化正确预测的样本数为 0
                total = 0.0  # 初始化总样本数为 0
                length = len(trainloader)  # 获取训练集的批次数量
                for i, data in enumerate(trainloader, 0):  # 遍历训练集的每个批次
                    # 准备数据
                    inputs, labels = data  # 获取输入数据和标签
                    inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到指定设备上
                    optimizer.zero_grad()  # 梯度清零，防止梯度累积

                    # forward + backward
                    outputs = net(inputs)  # 前向传播，计算模型输出
                    loss = criterion(outputs, labels)  # 计算损失
                    loss.backward()  # 反向传播，计算梯度
                    optimizer.step()  # 更新模型参数

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()  # 累加损失
                    _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
                    total += labels.size(0)  # 更新总样本数
                    correct += predicted.eq(labels).sum().item()  # 更新正确预测的样本数
                    print('[Epoch:%d, Iter:%d] Loss: %.03f | Acc: %.3f%%'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))  # 打印当前批次的损失和准确率
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%%'
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))  # 将结果写入日志文件
                    f2.write('\n')
                    f2.flush()  # 刷新文件缓冲区
```

 	在训练过程中，代码首先检查并创建了用于保存模型和日志的输出目录。然后，初始化了最佳准确率best_acc，用于保存训练过程中性能最好的模型。在每个训练轮次（epoch）中，模型被设置为训练模式，并遍历训练数据加载器trainloader。对于每个批次的数据，执行前向传播计算输出，计算损失值，进行反向传播更新模型参数。同时，累积计算损失和准确率，并打印和记录训练日志。

​	在每个epoch结束后，代码进入测试阶段。模型被设置为评估模式，禁用梯度计算以提高效率。遍历测试数据加载器testloader，计算模型在测试集上的预测准确率。如果当前的准确率超过了之前的最佳准确率，则更新best_acc并保存当前模型的参数。所有的测试结果和模型参数都会被保存到指定的文件中，方便后续的分析和部署。

## 输出结果

随着迭代次数的增加，训练集及测试集的准确率逐步增加。

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241128175434794.png" style="zoom:33%;" />

同时数据也会保存在单独的文件中

训练集准确率：

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241128180351719.png" style="zoom:50%;" />

测试集准确率：

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241128180527411.png" style="zoom:50%;" />

最终系统识别出对于测试集最佳的模型数据并给出：

<img src="/Users/pengzexue/Desktop/截屏2024-11-28 18.06.17.png" style="zoom:70%;" />

于是我们在model文件中单独提取出net_077.pth模型进行进一步测试。

## **测试模型**

### 训练集合并检测

```python
                    # 每训练完一个epoch测试一下准确率
                    print("Waiting Test!")  # 打印测试开始的信息
                    with torch.no_grad():  # 关闭梯度计算，节省内存
                        correct = 0  # 初始化正确预测的样本数为 0
                        total = 0  # 初始化总样本数为 0
                        net.eval()  # 设置模型为评估模式
                        for data in testloader:  # 遍历测试集
                            images, labels = data  # 获取测试数据和标签
                            images, labels = images.to(device), labels.to(device)  # 将数据移动到指定设备上
                            outputs = net(images)  # 前向传播，计算模型输出
                            # 取得分最高的那个类
                            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
                            total += labels.size(0)  # 更新总样本数
                            correct += predicted.eq(labels).sum().item()  # 更新正确预测的样本数
                        print('测试分类准确率为：%.3f%%' % (100. * correct / total))  # 打印测试准确率
                        acc = 100. * correct / total  # 计算准确率
                        # 将每次测试结果实时写入acc.txt文件中
                        print('Saving model......')  # 打印模型保存提示
                        torch.save(net.state_dict(), '%s/net_%03d.pth' %
                                   (args.outf, epoch + 1))  # 保存模型参数
                        f.write("EPOCH=%03d,Accuracy= %.3f%%" %
                                (epoch + 1, acc))  # 将测试结果写入日志文件
                        f.write('\n')
                        f.flush()  # 刷新文件缓冲区
                        # 记录最佳测试分类准确率并写入best_acc.txt文件中
                        if acc > best_acc:  # 如果当前准确率高于最佳准确率
                            with open("best_acc.txt", "w") as f3:  # 打开文件记录最佳准确率
                                f3.write("EPOCH=%d,best_acc= %.3f%%" %
                                         (epoch + 1, acc))  # 写入最佳准确率
                            best_acc = acc  # 更新最佳准确率
                print("Training Finished, Total EPOCH=%d" % EPOCH)  # 打印训练完成信息
```

  在测试模型部分，代码在每个 epoch 结束后评估模型在测试集上的性能。它将模型设置为评估模式，关闭梯度计算以节省资源。然后遍历测试数据集，对每个批次的数据进行前向传播，获取模型的预测结果。代码统计正确的预测数量和总样本数，计算测试集上的准确率。最后，打印测试结果，保存当前模型的参数到指定路径，并记录测试结果。如果当前测试准确率超过了之前的最佳准确率，代码会更新最佳准确率并保存对应的模型。这部分代码的核心是评估模型的泛化能力，确保模型在未见过的数据上也有良好的表现。

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241128181036357.png" style="zoom:50%;" />

### 随机单个图像测试

```python
import os
import torch
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from resnet import ResNet18  # 确保 ResNet18 定义在 resnet.py 文件中

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义数据集的预处理（与训练时保持一致）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载测试数据集
testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# CIFAR-10 类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 加载模型
model_path = './model/net_077.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = ResNet18().to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 随机选择一张测试图像
index = random.randint(0, len(testset) - 1)
image, true_label = testset[index]

# 数据需要扩展一个批次维度，并移动到设备上
image = image.unsqueeze(0).to(device)

# 进行预测
with torch.no_grad():
    output = model(image)
    _, predicted_label = torch.max(output, 1)

# 将图像数据转换回原始格式（取消归一化）
unnormalize = transforms.Normalize(
    mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
    std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
)
image = unnormalize(image.squeeze(0).cpu()).numpy().transpose((1, 2, 0))
image = (image - image.min()) / (image.max() - image.min())  # 将像素值归一化到 [0, 1]

# 显示图像及其预测结果
plt.imshow(image)
plt.title(f"True class: {classes[true_label]}, Prediction class: {classes[predicted_label.item()]}")
plt.axis('off')
plt.show()
```

对测试集单个图像随机抽取进行分析都可以正确识别图像。

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241128181140543.png" style="zoom:50%;" />

## 残差网络部分

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),  # 残差块中的卷积层1（计入总层数）
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),      # 残差块中的卷积层2（计入总层数）
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),         # 残差块中的捷径卷积层（不计入总层数）
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),     # 卷积层1（总层数中的第1层）
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 以下每个layer包含2个残差块，每个残差块有2个卷积层（计入总层数）
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)  # 卷积层2-5
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)  # 卷积层6-9
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)  # 卷积层10-13
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)  # 卷积层14-17
        self.fc = nn.Linear(512, num_classes)  # 全连接层（总层数中的第18层）

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides = [stride, 1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))  # 每个残差块添加2个卷积层
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)      # 卷积层1
        out = self.layer1(out)   # 卷积层2-5
        out = self.layer2(out)   # 卷积层6-9
        out = self.layer3(out)   # 卷积层10-13
        out = self.layer4(out)   # 卷积层14-17
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)       # 全连接层（第18层）
        return out

def ResNet18():
    return ResNet(ResidualBlock)
```

​	这篇论文介绍了一种深度残差学习框架，旨在解决训练深度神经网络的困难。相较于之前的方法，该框架通过显式地将神经网络的层重新表示为相对于层输入的学习残差函数，而不是学习无参考的函数，从而简化了网络的训练过程。论文作者通过全面的实证研究证明了残差网络更易于优化，并且可以从显著增加的深度中获得准确性的提升。

​	残差网络（Residual Network）的核心原理是引入了残差学习的概念，通过显式地将神经网络的层重新表示为相对于层输入的学习残差函数，从而简化了网络的训练过程。
​	在传统的深度神经网络中，网络通过堆叠一系列的层来建模复杂的非线性关系。每个层将输入变换为输出，通过学习层的参数来调整这种变换。然而，随着网络层数的增加，由于梯度消失和梯度爆炸等问题，网络的训练变得困难。
​	为了解决这个问题，残差网络提出了跳跃连接（Skip Connections）的概念。在残差网络中，每个层都有一个跳跃连接，将输入直接与输出相加，并传递给下一层。这种连接称为残差连接，因为它传递的是当前层相对于输入的残差。残差网络的基本单位是残差块（Residual Block），它由两个或多个卷积层组成。
​	通过引入残差连接，网络可以学习到对原始输入的增量变化，而不是尝试直接学习复杂的非线性映射。这样做的优势在于，即使网络层很深，也能够通过残差连接轻松地将梯度传递回早期的层。通过增加残差块的层数，网络可以逐渐学习到更复杂的特征表示。
此外，残差网络还引入了批量归一化（Batch Normalization）和全局平均池化（Global Average Pooling）等技术来进一步改善网络的性能和训练效率。
​	总结起来，残差网络的原理是通过引入残差连接，将网络的层表示为相对于输入的学习残差函数，简化了网络的训练过程，并使得网络能够更好地优化和学习深层的特征表示。这一原理的引入在计算机视觉和深度学习领域取得了重要的突破和应用。

残差网络在物体识别任务中的优势主要体现在以下几个方面：

​	训练效果更好：由于残差网络引入了残差学习的机制，使得网络更容易优化和训练。传统的深度神经网络在层数增加时容易出现梯度消失或梯度爆炸的问题，而残差连接能够有效地缓解这些问题，使得深层网络的训练更加稳定和高效。
​	更深的网络结构：残差网络可以构建非常深的网络结构，而不会出现过拟合或性能下降的情况。通过增加残差块的层数，网络可以学习到更复杂的特征表示，从而提高了物体识别的准确率。
​	更好的特征学习：残差网络能够学习到更加有意义和有效的特征表示。通过残差连接传递残差信息，网络可以专注于学习原始输入与目标输出之间的差异，从而提高了网络对目标任务的表征能力。
​	参数效率更高：相较于传统的网络结构，残差网络可以用更少的参数来表达更丰富的特征表示。这种参数效率使得残差网络在物体识别任务中能够更好地应对大规模数据和复杂场景的挑战。
​	在大规模任务上表现优异：残差网络在大规模视觉任务上表现出色，例如在ImageNet数据集上取得了很好的成绩。其在实际的物体识别应用中也表现出了良好的泛化性能和鲁棒性。
 	综上所述，残差网络在物体识别任务中具有更好的训练效果、更深的网络结构、更好的特征学习能力、更高的参数效率以及在大规模任务上的优异表现等优势，使其成为当前物体识别领域的重要技术之一。

**层数统计说明：**

​	1.**初始卷积层**：

​	•self.conv1 中的 nn.Conv2d：**第1层**

​	2.**残差块**：

​	•每个残差块包含2个卷积层（self.left 中的两个 nn.Conv2d），计入总层数。

​	•共 **4 个 layer**（self.layer1 到 self.layer4），每个 layer 有 2 个残差块，故共有 8 个残差块。

​	•8 个残差块 × 2 个卷积层 = **16 层**（第2层至第17层）

​	3.**全连接层**：

​	•self.fc：**第18层**

**总计层数：**

​	•卷积层：1（初始卷积层） + 16（残差块中的卷积层） = 17 层

​	•全连接层：1 层

​	•**总层数：17 + 1 = 18 层**

**注意：**快捷连接（self.shortcut）中的卷积层不计入总层数，因为在 ResNet 的计数方式中，只统计主要路径上的卷积层。

# 总结

​	总结来说，这段代码完整地实现了一个深度学习模型的训练过程，从数据预处理、模型构建、训练循环到模型评估和保存，都遵循了深度学习的最佳实践。通过使用ResNet-18模型和各种优化技巧，如数据增强、动量SGD优化器和L2正则化，模型能够在CIFAR-10数据集上取得较好的分类性能，为物体识别任务提供了有效的解决方案。