# VGGNet学习笔记及仿真

## 引言

VGGNet（Visual Geometry Group）[<sup>[1]</sup>](#ref-1)是2014年又一个经典的卷积神经网络，VGGNet的主要的想法是如何设计网络架构的问题，VGGNet在2014年获得了分类第二，定位第一的成绩，而分类第一的成绩由GoogLeNet[<sup>[2]</sup>](#ref-2)获得。

## VGGNet论文笔记

### VGGNet架构

* 卷积层的输入是一个固定大小的 224 x 224 尺寸的RGB图像。唯一做的预处理就是减去平均的RBG值，在卷积层使用了非常小的感视野大小的过滤器3x3（这是捕捉左/右、上/下，中心概念的最小尺寸）。在其中一种配置中，还用到了 1 x 1 的卷积滤波器，可以看做是输入通道的线性变换。卷积的步长固定为1，卷积层的空间填充使得空间分辨率在卷积之后得到保留。
* 空间池化是由五个最大池化层实现的（不是所有的卷积层之后就是最大池化层），最大池化层是在步长为2，尺寸为2x2的滤波器下实现的。

* 卷积池化处理之后就是全连接层，前两个层各有4096个神经元，第三层是1000个神经单元，表示ILSVRC的1000中分类，最后一层是softmax层，网络中的全连接层的配置一致。

所有的隐藏层之后都紧跟着ReLU激活函数，在VGGNet中没有用到AlexNet[<sup>[3]</sup>](#ref-3)中的LPN，VGGNet的作者认为这种规范化方法并不能提升模型的性能，反而会导致增加内存消耗和计算时间。

### 参数详解

![](https://www.madao33.com/media/VGGNet学习笔记及仿真/convnet_configuration.png)

VGGNet的不同的类型如上表所示，没一列表示不同配置的VGGNet，每个网络的深度都有所不同，从A的11个权重层（8个卷积层和3个全连接层）到网络E中的19个权重层（16个卷积层和3个全连接层）。

虽然VGGNet的深度很大，但是权重参数的数量并不比AlexNet[<sup>[3]</sup>](#ref-3)的权重数量多，VGGNet每个配置的参数如下表2所示。

![](https://www.madao33.com/media/VGGNet学习笔记及仿真/parameter_num.png)

### 创新点

比起AlexNet等网络，VGGNet使用的是 3 x 3 的卷积核，两个 3 x 3 卷积层的串联相当于一个 5 x 5 的卷积层，3个 3 x 3 的卷积层相当于 7 x 7 的卷积层其他的优点是：

* 3个 3 x 3 卷积层的串联合并了三个非线性操作，而不是一个，这可以是的决策函数更有区分性
* 这样可以大大地减少参数量，3个 3 x 3卷积层的参数量只有 7 x 7 卷积层的一半左右
  * 3个 3 x 3 卷积层有 $C$ 个通道，那么3个卷积层总共的参数有：$3(3^2C^2)=27C^2$
  * 单个7 x 7卷积层的参数：$7^2C^2=49C^2$

通过 3 x 3的滤波器分解，可以看做是一种对 7 x 7 卷积层的一种正则化

1 x 1 卷积层的引入是为了在不影响卷积层感受野的情况下增加决策函数非线性的方法。 1 x 1 卷积本质上是**同一纬度空间上的线性投影（输入和输出通道的数量相同）**，整流函数也会引入额外的非线性。

### 训练细节

VGGNet的训练步骤遵循AlexNet的训练步骤[<sup>[3]</sup>](#ref-3)（除了多尺度地对训练图像进行采样）

* 优化方法是（小批量动量梯度下降方法）mini-batch gradient descent with momentum

* batch = 256
* momentum = 0.9
* L2 penalty = 0.5
* 前两个全连接层droupout = 0.5
* learning rate = $10^{-2}$
* 随机初始化的参数是从均值0，方差$10^{-2}$的正态分布中采样，偏差为0
* 为了获得224 x 224 大小的输入图像，对重新缩放的图像进行随机裁剪（在每次SGD迭代的时候裁剪），为了扩充数据集，裁剪的图像还进行了水平翻转，具体的细节可以参考AlexNet网络预处理数据增强（data augmentation）部分[<sup>[3]</sup>](#ref-3)

### 结果

结果对比如下图所示

![](https://www.madao33.com/media/VGGNet学习笔记及仿真/comparison.png)

VGGNet的结果是**top-1 val error 23.7**，**top-5 val error 6.8**, **top-5 test error 6.8%**

## 代码实现

以下代码参考[<sup>[4]</sup>](#ref-4)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets 
import torchvision
import time
import os
```


```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = dsets.CIFAR10(root='./data/CIFAR10/', 
                            train=True, 
                            transform=transform_train,
                            download=True)
test_dataset = dsets.CIFAR10(root='./data/CIFAR10/', 
                           train=False, 
                           transform=transform_test)
```

    Files already downloaded and verified



```python
# reducing the dataset
reduced_train_dataset = []
for images, labels in train_dataset:
    if labels < 3:
        reduced_train_dataset.append((images, labels))       
reduced_test_dataset = []
for images, labels in test_dataset:
    if labels < 3:
        reduced_test_dataset.append((images, labels))
```


```python
print("The number of training images : ", len(reduced_train_dataset))
print("The number of test images : ", len(reduced_test_dataset))
```

    The number of training images :  15000
    The number of test images :  3000



```python
print('STEP 2: MAKING DATASET ITERABLE')
train_loader = torch.utils.data.DataLoader(dataset=reduced_train_dataset, 
                                           batch_size=128, 
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=reduced_test_dataset, 
                                          batch_size=100, 
                                          shuffle=False)
class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

    STEP 2: MAKING DATASET ITERABLE


### Visualize a few images


```python
import matplotlib.pyplot as plt
%matplotlib inline  
import numpy as np
```


```python
def imshow(inp, title=None):

    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
```


```python
train_loader_sample = torch.utils.data.DataLoader(dataset=reduced_train_dataset, 
                                           batch_size=4, 
                                           shuffle=True)

# Get a batch of training data
inputs, classes = next(iter(train_loader_sample))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
```


​    
![png](https://www.madao33.com/media/VGGNet学习笔记及仿真/VGGNet_9_0.png)
​    



```python
class VGG(nn.Module):

    def __init__(self):

        super(VGG, self).__init__()
        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )

        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )

        self.conv4 = nn.Sequential(

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv5 = nn.Sequential(

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(512, 3)


    def forward(self, x):

        # Convolutions
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        # Resize
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
```


```python
## Instantiate Model Class
model = VGG()
num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```

    The number of parameters :  14724675





    VGG(
      (conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv3): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv4): Sequential(
        (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv5): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (fc1): Linear(in_features=512, out_features=3, bias=True)
    )




```python
## Loss/Optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-2
momentum = 0.9
weight_decay = 5e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum, weight_decay = weight_decay)
```


```python
## Training
num_epochs = 50
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images) 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * correct.item() / total
    # Print Loss
    print('Epoch {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))
#############
```

    /usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


    Epoch 0. Loss: 0.5742393732070923. Accuracy: 83.66666666666667
    Epoch 1. Loss: 0.40269795060157776. Accuracy: 86.63333333333334
    Epoch 2. Loss: 0.2371339052915573. Accuracy: 88.5
    Epoch 3. Loss: 0.44756925106048584. Accuracy: 88.23333333333333
    Epoch 4. Loss: 0.39500197768211365. Accuracy: 88.8
    Epoch 5. Loss: 0.22342568635940552. Accuracy: 90.76666666666667
    Epoch 6. Loss: 0.05593661591410637. Accuracy: 91.23333333333333
    Epoch 7. Loss: 0.011301316320896149. Accuracy: 90.63333333333334
    Epoch 8. Loss: 0.04968326911330223. Accuracy: 92.16666666666667
    Epoch 9. Loss: 0.12825727462768555. Accuracy: 90.73333333333333
    Epoch 10. Loss: 0.008245733566582203. Accuracy: 92.16666666666667
    Epoch 11. Loss: 0.19846688210964203. Accuracy: 91.8
    Epoch 12. Loss: 0.06683117896318436. Accuracy: 91.46666666666667
    Epoch 13. Loss: 0.06584116816520691. Accuracy: 92.3
    Epoch 14. Loss: 0.07137342542409897. Accuracy: 91.96666666666667
    Epoch 15. Loss: 0.0017633828101679683. Accuracy: 92.86666666666666
    Epoch 16. Loss: 0.06365344673395157. Accuracy: 93.1
    Epoch 17. Loss: 0.002621516352519393. Accuracy: 91.63333333333334
    Epoch 18. Loss: 0.2748899459838867. Accuracy: 92.93333333333334
    Epoch 19. Loss: 0.011702283285558224. Accuracy: 93.4
    Epoch 20. Loss: 0.0003703629190567881. Accuracy: 92.66666666666667
    Epoch 21. Loss: 0.0024486021138727665. Accuracy: 93.7
    Epoch 22. Loss: 0.0356595404446125. Accuracy: 93.33333333333333
    Epoch 23. Loss: 0.0003634128952398896. Accuracy: 92.76666666666667
    Epoch 24. Loss: 0.04696182534098625. Accuracy: 92.56666666666666
    Epoch 25. Loss: 0.09958905726671219. Accuracy: 92.7
    Epoch 26. Loss: 0.17162220180034637. Accuracy: 92.36666666666666
    Epoch 27. Loss: 0.009680134244263172. Accuracy: 92.93333333333334
    Epoch 28. Loss: 0.08440369367599487. Accuracy: 93.5
    Epoch 29. Loss: 0.06431729346513748. Accuracy: 93.26666666666667
    Epoch 30. Loss: 0.009013362228870392. Accuracy: 93.46666666666667
    Epoch 31. Loss: 0.18230386078357697. Accuracy: 93.4
    Epoch 32. Loss: 0.0009615866583772004. Accuracy: 92.63333333333334
    Epoch 33. Loss: 0.13079260289669037. Accuracy: 92.13333333333334
    Epoch 34. Loss: 0.00011742366041289642. Accuracy: 93.26666666666667
    Epoch 35. Loss: 0.13156653940677643. Accuracy: 93.13333333333334
    Epoch 36. Loss: 0.07050688564777374. Accuracy: 92.36666666666666
    Epoch 37. Loss: 0.0008148620836436749. Accuracy: 93.23333333333333
    Epoch 38. Loss: 0.03399736061692238. Accuracy: 93.66666666666667
    Epoch 39. Loss: 0.0019302064320072532. Accuracy: 93.36666666666666
    Epoch 40. Loss: 0.0008526549208909273. Accuracy: 93.26666666666667
    Epoch 41. Loss: 0.0002856892824638635. Accuracy: 93.23333333333333
    Epoch 42. Loss: 0.001912760897539556. Accuracy: 92.9
    Epoch 43. Loss: 0.007499492261558771. Accuracy: 93.26666666666667
    Epoch 44. Loss: 0.00024754248443059623. Accuracy: 93.06666666666666
    Epoch 45. Loss: 0.025322042405605316. Accuracy: 93.06666666666666
    Epoch 46. Loss: 0.00015217023610603064. Accuracy: 93.03333333333333
    Epoch 47. Loss: 0.0003425484464969486. Accuracy: 93.26666666666667
    Epoch 48. Loss: 0.0007479854975827038. Accuracy: 93.9
    Epoch 49. Loss: 0.009067901410162449. Accuracy: 92.93333333333334

## 参考文献

<div id="ref-1"></div>

- [1] [Simonyan K ,  Zisserman A . Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. Computer Science, 2014.](https://arxiv.org/pdf/1409.1556.pdf)

<div id="ref-2"></div>

- [2] [Szegedy C , Liu W , Jia Y , et al. Going Deeper with Convolutions[J]. IEEE Computer Society, 2014.](https://arxiv.org/pdf/1409.4842.pdf)

<div id="ref-3"></div>

- [3] [Krizhevsky A , Sutskever I , Hinton G . ImageNet Classification with Deep Convolutional Neural Networks[J]. Advances in neural information processing systems, 2012, 25(2).](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

<div id="ref-4"></div>

- [4] [jays0606/VGGNet](https://github.com/jays0606/VGGNet)
