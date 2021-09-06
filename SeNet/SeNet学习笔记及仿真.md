# SeNet学习笔记及仿真

## 前言

SENet[<sup>[1]</sup>](#ref-1)是ImageNet 2017年的冠军模型，自SeNet提出后，ImageNet挑战赛就停止举办了。SENet同之前的ResNet一样，引入了一些技巧，可以在很大程度上降低模型的参数，并且提升模型的运算速度。

SENet全称Squeeze-and-Excitation Networks，中文名可以翻译为挤压和激励网络。SENet在ImageNet 2017取得了第一名的成绩，Top-5 error rate降低到了2.251%，官方的模型和代码在github仓库中可以找到[<sup>[2]</sup>](#ref-2)。

## SE block

SENet提出的动机是将通道之间的关系结合起来，于是引出了一个*Squeeze-and-excitation*（SE）块[<sup>[1]</sup>](#ref-1)，它的目的就是**通过显式建模网络卷积特征的信道之间的相互依赖性来提高网络表征的质量**。SE块的机制也可以说是**通过学习全局信息来选择性地强调有用的特征和抑制不太有用的特征**，SENet块如[fig1](#fig-1)所示。

<div id="fig-1"></div>

![](https://www.madao33.com/media/SeNet学习笔记及仿真/fig1.png)

SE模块可以看作是一个计算单元，用 $F_{tr}$ 表示，可以将输入 $X \in \R^{H' \times W' \times C'}$ 映射为特征图 $U \in \R^{H \times W \times C}$。以下的符号中，$F_{tr}$ 表示卷积操作，$\bold{V}=[V_1, V_2, \dots, V_C]$ 来表示学习到的一组滤波器核，其中 $V_c$ 表示的是第 $c$ 个滤波器的参数，所以输出可以表示为 $\bold{U}=[U_1, U_2, \dots, U_C]$，其中：

<div id="eqn-1"></div>

$$
U_c=V_c * \bold{X}=\sum_{s=1}^{C'}V_c^s * X^s
$$

[公式1](#eqn-1)中 $*$ 表示的是卷积操作，$V_c=[V_c^1, V_c^2, \dots, V_c^{C'}], \quad \bold{X}=[X^1, X^2, \dots, X^{C'}]$ 以及 $u_c \in \R^{H \times W}$， $V_c^s$ 表示的是 $\bold{X}$ 对应单个 $V_c$ 通道的 2D 空间核。

对于以上公式有以下的说明：

* 为了简化符号表达，省略了偏差项
* 从以上的卷积公式可以看出，各个通道的卷积进行了求和操作，所以通道的特征信息和卷积核学习到的空间关系混合到一起，所以需要**分离两个特征信息，让模型学习到通道的特征关系**

### Squeeze: Global Information Embedding

为了解决通道依赖的问题，需要考虑将输出特征中每个通道对应的信号。每一个训练的滤波器都有一个局部感受野，因此每个神经元的转换输出都不能很好地利用这个区域之外的上下文信息。

为了解决这个问题，SeNet 将全局空间信息压缩到通道描述符中，这是通过使用全局平均池化（global average pooling）来生成通道统计数据来实现的。形式上，统计量 $Z \in \R^C$ 是通过收缩 $U$ 的空间维度 $H \times W$ 来生成的，从而 $Z$ 的第 $c$ 个元素通过以下方式计算：

<div id = "eqn-1"></div>

$$
z_c = F_{sq}(u_c)=\frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c (i, j)
$$

### Excitation: Adaptive Recalibration

为了利用在 *Squeeze* 操作中聚集到的信息，接下来进行第二个操作，目的是为了完全捕获通道依赖信息。为了实现这一目标，该功能必须满足两个标准：

1. 它必须要是灵活的，特别地，它必须能够学习通道之间的非线性相互作用
2. 它必须学习一种非互斥的关系，因为我们希望确保允许强调多个通道

为了满足这些标准，这里选择了带有 **sigmoid** 激活函数的简单门控机制：

<div id="ref-3"></div>

$$
s = F_{ex}(z, W) = \sigma (g(z, \bold{W}))=\sigma(\bold{W}_2 \delta(\bold{W}_1 z))
$$

其中 $\delta$ 表示的是 *ReLU* 函数，$\bold{W}_1 \in \R^{\frac{C}{r} \times C} ,\quad \bold{W}_2 \in \R^{C \times \frac{C}{r}}$ 。为了降低模型复杂度以及提升泛化能力，这里用到了两个全连接层的bottleneck结构，其中第一个全连接层起到降维的作用，降维系数为r是个超参数，然后采用ReLU激活，最后的全连接层恢复原始的维度，最后将学习到的各个通道的激活值（sigmoid激活，值为0~1）乘上$U$ 上的原始特征：

<div id="eqn-4"></div>

$$
\tilde{x}_c = F_{scale}(u_c, s_c) = s_c \cdot u_c
$$

其中 $\widetilde{\bold{X}}=[\widetilde{X}_1, \widetilde{X}_2, \dots, \widetilde{X}_C]$, $F_{scale}(u_c, s_c)$ 表示的是标量 $S_c$ 和特征图 $u_c \in \R^{H \times W}$ 的乘法

> 其实整个操作可以看做学习到了各个通道的权重参数，从而使得模型对各个通道的特征更加有辨别能力，这应该也算一种attention机制[<sup>[3]</sup>](#ref-3)

## SE block的应用

SE模块十分灵活，可以直接应用到现用的网络架构中。例如GoogLeNet和ResNet等，如[图2](#fig-2)和[图3](#fig-3)所示

<div id="fig2"></div>

![](https://www.madao33.com/media/SeNet学习笔记及仿真/fig2.png)



<div id="fig3"></div>

![](https://www.madao33.com/media/SeNet学习笔记及仿真/fig3.png)

同样地，SE模块还可以应用在其他的网络结构，这里给出论文中的原表格，SE-ResNet-50和SE-ResNetXt-50的具体结构，见[表格1](#tab-1)

<div id="tab-1"></div>

![](https://www.madao33.com/media/SeNet学习笔记及仿真/tab1.png)

增加了SE模块后，模型的参数以及计算量都会相应的增加，这些增加的参数仅仅由门控门控机制的两个全连接层产生，因此只占网络容量的一小部分。具体的计算公式如[公式5](#eqn-5)：

<div id="eqn-5"></div>

$$
\frac{2}{r}\sum_{s=1}^s N_s \cdot C_s^2
$$

其中 $r$ 表示的是降维系数，$S$ 表示的是级数（the number of stages），一个级数指的是对公共空间维度的特征图进行操作的块的集合，$C_s$ 表示的输出通道的维度，$N_s$ 表示的级数 $S$ 重复块的数量。

当 $r=16$ 时， SE-ResNet-50 只增加了约10%的参数量，但是计算量却增加不到1%

## SE模型性能

SE模块可以很容易地引入到其他网络中，为了验证SE模块的效果，在主流的流行网络中引入了SE模块，对比其在ImageNet上的效果，如[表2](#tab-2)所示：

<div id="tab-2"></div>

![](https://www.madao33.com/media/SeNet学习笔记及仿真/tab2.png)

可以看到所有的网络在加入SE模块后分类准确度均有一定的提升，为了实际地体会SE模块，之后就是尝试仿真实现，更加深入的了解其网络架构和效果

## SE模块仿真

以下代码参考的是github代码[<sup>[4]</sup>](#ref-4)


```python
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import time

device = ('cuda' if torch.cuda.is_available() else 'cpu')
device
```




```
'cpu'
```




```python
# 超参数
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 1e-1
WEIGHT_DECAY = 1e-4
```

### 获取数据

使用`torchvision.dataset`获取数据


```python
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]))
```

```
Files already downloaded and verified
Files already downloaded and verified
```



```python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
```

### 定义SeNet模型


```python
# Squeeze and Excitation Block Module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1) # Squeeze
        w = self.fc(x)
        w, b = w.split(w.data.size(1) // 2, dim=1) # Excitation
        w = torch.sigmoid(w)
        return x * w + b # Scale and add bias
```


```python
# Residual Block with SEBlock
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv_lower = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.conv_upper = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.se_block = SEBlock(channels)

    def forward(self, x):
        path = self.conv_lower(x)
        path = self.conv_upper(path)
        path = self.se_block(path)
        path = x + path
        return F.relu(path)
```


```python
# Network Module
class Network(nn.Module):
    def __init__(self, in_channel, filters, blocks, num_classes):
        super(Network, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks - 1)])

        self.out_conv = nn.Sequential(
            nn.Conv2d(filters, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )


        self.fc = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)    
        x = self.out_conv(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.data.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
```

### 训练模型 


```python
net = Network(3, 128, 10, 10).to(device)
ACE = nn.CrossEntropyLoss().to(device)
opt = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=.9, nesterov=True)
```


```python
for epoch in range(1, EPOCHS + 1):
	print('[Epoch %d]' % epoch)
	train_loss = 0
	train_correct, train_total = 0, 0
	start_point = time.time()
	for inputs, labels in train_loader:
		inputs, labels = Variable(inputs).to(device),Variable(labels).to(device)
		opt.zero_grad()
		preds = net(inputs)
		loss = ACE(preds, labels)
		loss.backward()
        opt.step()	
        train_loss += loss.item()
        train_correct += (preds.argmax(dim=1) == labels).sum().item()
        train_total += len(preds)
    
	print('train-acc : %.4f%% train-loss : %.5f' % (100 * train_correct / train_total, train_loss / len(train_loader)))
	print('elapsed time: %ds' % (time.time() - start_point))

	test_loss = 0
	test_correct, test_total = 0, 0
    
    for inputs, labels in test_loader:
        with torch.no_grad():
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            preds = net(inputs)
            test_loss += ACE(preds, labels).item()
            test_correct += (preds.argmax(dim=1) == labels).sum().item()
            test_total += len(preds)


	print('test-acc : %.4f%% test-loss : %.5f' % (100 * test_correct / test_total, test_loss / len(test_loader)))
    
	torch.save(net.state_dict(), './data/checkpoint/checkpoint-%04d.bin' % epoch)
```

```
[Epoch 1]
train-acc : 62.9240% train-loss : 1.02725
elapsed time: 167s
test-acc : 59.9800% test-loss : 1.13711
[Epoch 2]
train-acc : 69.3160% train-loss : 0.85710
elapsed time: 170s
test-acc : 67.6300% test-loss : 0.92139
[Epoch 3]
train-acc : 73.9000% train-loss : 0.74356
elapsed time: 171s
test-acc : 70.7700% test-loss : 0.84002
[Epoch 4]
train-acc : 77.2340% train-loss : 0.65098
elapsed time: 171s
test-acc : 74.3400% test-loss : 0.75001
[Epoch 5]
train-acc : 79.7560% train-loss : 0.58424
elapsed time: 171s
test-acc : 74.8000% test-loss : 0.71813
[Epoch 6]
train-acc : 81.8820% train-loss : 0.52713
elapsed time: 171s
test-acc : 77.7400% test-loss : 0.66449
[Epoch 7]
train-acc : 83.0260% train-loss : 0.49098
elapsed time: 171s
test-acc : 79.3000% test-loss : 0.60599
[Epoch 8]
train-acc : 84.2880% train-loss : 0.45633
elapsed time: 171s
test-acc : 78.0500% test-loss : 0.64819
[Epoch 9]
train-acc : 85.2660% train-loss : 0.43147
elapsed time: 171s
test-acc : 80.7400% test-loss : 0.57734
[Epoch 10]
train-acc : 86.2080% train-loss : 0.39924
elapsed time: 171s
test-acc : 81.9000% test-loss : 0.53836
[Epoch 11]
train-acc : 86.9320% train-loss : 0.38040
elapsed time: 171s
test-acc : 82.7100% test-loss : 0.51160
[Epoch 12]
train-acc : 87.4740% train-loss : 0.36286
elapsed time: 170s
test-acc : 81.8500% test-loss : 0.54868
[Epoch 13]
train-acc : 88.1580% train-loss : 0.34673
elapsed time: 171s
test-acc : 83.0700% test-loss : 0.49779
[Epoch 14]
train-acc : 88.9260% train-loss : 0.31996
elapsed time: 171s
test-acc : 83.8900% test-loss : 0.48193
[Epoch 15]
train-acc : 89.1380% train-loss : 0.31583
elapsed time: 171s
test-acc : 83.9900% test-loss : 0.49245
[Epoch 16]
train-acc : 89.5460% train-loss : 0.30087
elapsed time: 170s
test-acc : 84.0100% test-loss : 0.49648
[Epoch 17]
train-acc : 90.0420% train-loss : 0.29067
elapsed time: 171s
test-acc : 85.2700% test-loss : 0.44473
[Epoch 18]
train-acc : 90.3720% train-loss : 0.28137
elapsed time: 171s
test-acc : 83.8900% test-loss : 0.49883
[Epoch 19]
train-acc : 90.6020% train-loss : 0.26961
elapsed time: 171s
test-acc : 84.4700% test-loss : 0.47203
[Epoch 20]
train-acc : 91.1460% train-loss : 0.25927
elapsed time: 170s
test-acc : 84.4200% test-loss : 0.49412
[Epoch 21]
train-acc : 91.1540% train-loss : 0.25661
elapsed time: 170s
test-acc : 85.3500% test-loss : 0.43626
[Epoch 22]
train-acc : 91.3620% train-loss : 0.24741
elapsed time: 171s
test-acc : 86.2200% test-loss : 0.41310
[Epoch 23]
train-acc : 91.9760% train-loss : 0.23271
elapsed time: 171s
test-acc : 86.5600% test-loss : 0.40795
[Epoch 24]
train-acc : 92.0000% train-loss : 0.23080
elapsed time: 171s
test-acc : 84.8000% test-loss : 0.46834
[Epoch 25]
train-acc : 92.1460% train-loss : 0.22744
elapsed time: 171s
test-acc : 85.4300% test-loss : 0.44402
[Epoch 26]
train-acc : 92.2120% train-loss : 0.22320
elapsed time: 170s
test-acc : 86.3300% test-loss : 0.41405
[Epoch 27]
train-acc : 92.3740% train-loss : 0.21625
elapsed time: 170s
test-acc : 87.3800% test-loss : 0.38440
[Epoch 28]
train-acc : 92.6960% train-loss : 0.21098
elapsed time: 171s
test-acc : 84.9300% test-loss : 0.46326
[Epoch 29]
train-acc : 92.8700% train-loss : 0.20541
elapsed time: 171s
test-acc : 86.5900% test-loss : 0.41840
[Epoch 30]
train-acc : 93.0700% train-loss : 0.20067
elapsed time: 170s
test-acc : 86.8400% test-loss : 0.42302
[Epoch 31]
train-acc : 93.2300% train-loss : 0.19319
elapsed time: 171s
test-acc : 87.1700% test-loss : 0.39542
[Epoch 32]
train-acc : 93.2280% train-loss : 0.19576
elapsed time: 171s
test-acc : 86.6500% test-loss : 0.43697
[Epoch 33]
train-acc : 93.5900% train-loss : 0.18686
elapsed time: 170s
test-acc : 86.8300% test-loss : 0.40863
[Epoch 34]
train-acc : 93.5820% train-loss : 0.18315
elapsed time: 170s
test-acc : 86.8200% test-loss : 0.42321
[Epoch 35]
train-acc : 93.6140% train-loss : 0.18232
elapsed time: 170s
test-acc : 86.1700% test-loss : 0.43491
[Epoch 36]
train-acc : 93.9620% train-loss : 0.17560
elapsed time: 170s
test-acc : 86.9100% test-loss : 0.41068
[Epoch 37]
train-acc : 93.9920% train-loss : 0.17193
elapsed time: 170s
test-acc : 87.0600% test-loss : 0.41822
[Epoch 38]
train-acc : 93.8620% train-loss : 0.17253
elapsed time: 170s
test-acc : 88.0500% test-loss : 0.38560
[Epoch 39]
train-acc : 94.2040% train-loss : 0.16850
elapsed time: 170s
test-acc : 86.7000% test-loss : 0.42949
[Epoch 40]
train-acc : 94.2940% train-loss : 0.16422
elapsed time: 170s
test-acc : 87.2100% test-loss : 0.39914
```


## 导入保存的模型


```python
net.load_state_dict(torch.load('data\\checkpoint\\checkpoint-0040.bin', map_location=torch.device('cpu')))
net.eval()
```




```
Network(
  (conv_block): Sequential(
    (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (res_blocks): Sequential(
    (0): ResBlock(
      (conv_lower): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_upper): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (se_block): SEBlock(
        (fc): Sequential(
          (0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(8, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
    )
    (1): ResBlock(
      (conv_lower): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_upper): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (se_block): SEBlock(
        (fc): Sequential(
          (0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(8, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
    )
    (2): ResBlock(
      (conv_lower): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_upper): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (se_block): SEBlock(
        (fc): Sequential(
          (0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(8, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
    )
    (3): ResBlock(
      (conv_lower): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_upper): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (se_block): SEBlock(
        (fc): Sequential(
          (0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(8, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
    )
    (4): ResBlock(
      (conv_lower): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_upper): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (se_block): SEBlock(
        (fc): Sequential(
          (0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(8, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
    )
    (5): ResBlock(
      (conv_lower): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_upper): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (se_block): SEBlock(
        (fc): Sequential(
          (0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(8, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
    )
    (6): ResBlock(
      (conv_lower): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_upper): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (se_block): SEBlock(
        (fc): Sequential(
          (0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(8, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
    )
    (7): ResBlock(
      (conv_lower): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_upper): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (se_block): SEBlock(
        (fc): Sequential(
          (0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(8, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
    )
    (8): ResBlock(
      (conv_lower): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_upper): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (se_block): SEBlock(
        (fc): Sequential(
          (0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(8, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
    )
  )
  (out_conv): Sequential(
    (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (fc): Linear(in_features=128, out_features=10, bias=True)
)
```




```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
sns.set()
```


```python
for images, labels in test_loader:
    pred = torch.argmax(net(images), axis=1)
    print('confusion_matrix: \n', confusion_matrix(pred, labels))
    print('accuracy_score:', accuracy_score(pred, labels))
    print('precision_score:', precision_score(pred, labels, average='micro'))
    print('f1-score:', f1_score(pred, labels, average='micro'))
    break
```

```
confusion_matrix: 
 [[11  0  0  0  0  0  0  0  1  0]
 [ 0  9  0  0  0  0  0  0  0  0]
 [ 0  0 10  0  1  2  0  0  0  0]
 [ 0  0  1 11  0  0  0  1  0  0]
 [ 0  0  0  1  9  0  0  0  0  0]
 [ 0  0  0  1  0  7  1  0  0  0]
 [ 0  0  0  0  0  0 18  0  0  0]
 [ 1  0  0  2  0  0  0 12  0  0]
 [ 1  0  0  0  0  0  0  0 16  0]
 [ 0  1  0  0  0  0  0  0  0 11]]
accuracy_score: 0.890625
precision_score: 0.890625
f1-score: 0.890625
```



```python
pred
```




```
tensor([3, 8, 8, 8, 6, 6, 1, 6, 3, 1, 0, 9, 5, 7, 9, 8, 5, 7, 0, 6, 7, 0, 4, 9,
        2, 2, 4, 0, 9, 6, 6, 5, 4, 5, 9, 3, 4, 9, 9, 5, 4, 6, 5, 6, 0, 9, 4, 9,
        7, 6, 9, 8, 7, 3, 8, 8, 7, 3, 2, 5, 7, 5, 6, 3, 6, 2, 1, 2, 7, 7, 2, 6,
        8, 8, 0, 2, 9, 3, 7, 8, 8, 1, 1, 7, 2, 2, 2, 7, 8, 9, 0, 3, 8, 6, 4, 6,
        6, 0, 0, 7, 4, 5, 6, 3, 1, 1, 3, 6, 8, 7, 4, 0, 6, 2, 1, 3, 0, 4, 2, 7,
        8, 3, 1, 2, 8, 0, 8, 3])
```




```python
conf_mat = confusion_matrix(labels, pred)
conf_mat
```




```
array([[11,  0,  0,  0,  0,  0,  0,  1,  1,  0],
       [ 0,  9,  0,  0,  0,  0,  0,  0,  0,  1],
       [ 0,  0, 10,  1,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0, 11,  1,  1,  0,  2,  0,  0],
       [ 0,  0,  1,  0,  9,  0,  0,  0,  0,  0],
       [ 0,  0,  2,  0,  0,  7,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  1, 18,  0,  0,  0],
       [ 0,  0,  0,  1,  0,  0,  0, 12,  0,  0],
       [ 1,  0,  0,  0,  0,  0,  0,  0, 16,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 11]], dtype=int64)
```




```python
df = pd.DataFrame(conf_mat, index=test_dataset.classes, columns=test_dataset.classes)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airplane</th>
      <th>automobile</th>
      <th>bird</th>
      <th>cat</th>
      <th>deer</th>
      <th>dog</th>
      <th>frog</th>
      <th>horse</th>
      <th>ship</th>
      <th>truck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>airplane</th>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>automobile</th>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>bird</th>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cat</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>deer</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>dog</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>frog</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>horse</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ship</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>truck</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 绘制混淆矩阵图
plt.figure(figsize=(12, 12))
plt.rcParams['font.sans-serif']=['SimHei']
sns.heatmap(df, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()
```


![png](https://www.madao33.com/media/SeNet学习笔记及仿真/SeNet_20_0.png)
    


## 参考文献

<div id="ref-1"></div>

- [1] [Hu J, Shen L, Sun G. Squeeze-and-excitation networks[J]. arXiv preprint arXiv:1709.01507, 2017, 7.](https://arxiv.org/pdf/1709.01507.pdf)

<div id="ref-2"></div>

- [2] [hujip-frank/SENet](https://github.com/hujie-frank/SENet)

<div id="ref-3"></div>

- [3] [知乎文章：最后一届ImageNet冠军模型：SENet](https://zhuanlan.zhihu.com/p/65459972/)

<div id="ref-4"></div>

- [4] [JYPark09/SENet-Pytorch](https://github.com/JYPark09/SENet-PyTorch)