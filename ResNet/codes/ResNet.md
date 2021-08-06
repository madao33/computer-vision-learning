# 基于CIFAR10仿真ResNet


```python
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
```


```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```


```python
train_set = torchvision.datasets.CIFAR10('../data', train=True, 
                                         download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10('../data', train=False, 
                                        download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, 
                                           shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, 
                                          shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
```

    Using downloaded and verified file: ../data\cifar-10-python.tar.gz
    Extracting ../data\cifar-10-python.tar.gz to ../data
    Files already downloaded and verified
    


```python
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()
    
image_iter = iter(train_loader)
images, _ = image_iter.next()
imshow(torchvision.utils.make_grid(images[:4]))
```


    
![png](ResNet_files/ResNet_4_0.png)
    



```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
```




    'cuda'



### BasicBlock

![image.png](../imgs/fig2.png)


```python
class BasicBlock(nn.Module):
    """
    对于浅层网络，我们使用基本的Block
    基础块没有维度压缩，所以expansion=1
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # 如果输入输出维度不等，则使用1x1卷积层来改变维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )
            
    def forward(self, x):
        out = self.features(x)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
```


```python
# 测试
basic_block = BasicBlock(64, 128)
print(basic_block)
x = torch.randn(2, 64, 32, 32)
y = basic_block(x)
print(y.shape)
```

    BasicBlock(
      (features): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shortcut): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    torch.Size([2, 128, 32, 32])
    

### Bottleneck Block
![image.png](../imgs/fig5.png)


```python
class Bottleneck(nn.Module):
    """
    对于深层网络，我们使用BottleNeck，论文中提出其拥有近似的计算复杂度，但能节省很多资源
    zip_channels: 压缩后的维数，最后输出的维数是 expansion * zip_channels
    """
    expansion = 4
    
    def __init__(self, in_channels, zip_channels, stride=1):
        super(Bottleneck, self).__init__()
        out_channels = self.expansion * zip_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, zip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(zip_channels, zip_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(zip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.features(x)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
```


```python
# 测试
bottleneck = Bottleneck(256, 128)
print(bottleneck)
x = torch.randn(2, 256, 32, 32)
y = bottleneck(x)
print(y.shape)
```

    Bottleneck(
      (features): Sequential(
        (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shortcut): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    torch.Size([2, 512, 32, 32])
    

### ResNet
![image.png](../imgs/fig3.png)


```python
class ResNet(nn.Module):
    """
    不同的ResNet架构都是统一的一层特征提取、四层残差，不同点在于每层残差的深度。
    对于cifar10，feature map size的变化如下：
    (32, 32, 3) -> [Conv2d] -> (32, 32, 64) -> [Res1] -> (32, 32, 64) -> [Res2] 
 -> (16, 16, 128) -> [Res3] -> (8, 8, 256) ->[Res4] -> (4, 4, 512) -> [AvgPool] 
 -> (1, 1, 512) -> [Reshape] -> (512) -> [Linear] -> (10)
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # cifar10经过上述结构后，到这里的feature map size是 4 x 4 x 512 x expansion
        # 所以这里用了 4 x 4 的平均池化
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifer = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 第一个block要进行降采样
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            # 如果是Bottleneck Block的话需要对每层输入的维度进行压缩，压缩后再增加维数
            # 所以每层的输入维数也要跟着变
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.features(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        return out
```


```python
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
```


```python
net = ResNet34().to(device)
print(net)
if device == 'cuda':
    net = nn.DataParallel(net)
    # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
    torch.backends.cudnn.benchmark = True
```

    ResNet(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (layer1): Sequential(
        (0): BasicBlock(
          (features): Sequential(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
        (1): BasicBlock(
          (features): Sequential(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
        (2): BasicBlock(
          (features): Sequential(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (features): Sequential(
            (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (features): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
        (2): BasicBlock(
          (features): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
        (3): BasicBlock(
          (features): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (features): Sequential(
            (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (features): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
        (2): BasicBlock(
          (features): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
        (3): BasicBlock(
          (features): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
        (4): BasicBlock(
          (features): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
        (5): BasicBlock(
          (features): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (features): Sequential(
            (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (features): Sequential(
            (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
        (2): BasicBlock(
          (features): Sequential(
            (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
        )
      )
      (avg_pool): AvgPool2d(kernel_size=4, stride=4, padding=0)
      (classifer): Linear(in_features=512, out_features=10, bias=True)
    )
    


```python
# 测试
x = torch.randn(2, 3, 32, 32).to(device)
y = net(x)
print(y.shape)
```

    torch.Size([2, 10])
    


```python
lr = 1e-1
momentum = 0.9
weight_decay = 5e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.1, patience=3, verbose=True)
```


```python
# Training
def train(epoch):
    print('\nEpoch: %d' % (epoch))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.6f |  Acc: %.3f%% (%d/%d)' %
                  (epoch + 1, batch_idx + 1, train_loss, 100.*correct/total, correct, total))
    return train_loss
```


```python
load_model = False
if load_model:
    checkpoint = torch.load('./checkpoint/res18.ckpt')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
else:
    start_epoch = 0
print('start_epoch: %s' % start_epoch)
```

    start_epoch: 0
    


```python
for epoch in range(start_epoch, 50):
    loss = train(epoch)
    print('Total loss: %.6f' % loss)
    start_epoch = epoch
    scheduler.step(loss)
```

    
    Epoch: 0
    [1,   100] loss: 118.672118 |  Acc: 56.953% (7290/12800)
    [1,   200] loss: 235.403772 |  Acc: 57.355% (14683/25600)
    [1,   300] loss: 342.972803 |  Acc: 58.721% (22549/38400)
    Total loss: 436.702453
    
    Epoch: 1
    [2,   100] loss: 95.490529 |  Acc: 65.930% (8439/12800)
    [2,   200] loss: 189.245391 |  Acc: 66.246% (16959/25600)
    [2,   300] loss: 279.333860 |  Acc: 66.758% (25635/38400)
    Total loss: 357.071455
    
    Epoch: 2
    [3,   100] loss: 76.589579 |  Acc: 73.203% (9370/12800)
    [3,   200] loss: 151.513288 |  Acc: 73.492% (18814/25600)
    [3,   300] loss: 224.068864 |  Acc: 73.836% (28353/38400)
    Total loss: 286.139592
    
    Epoch: 3
    [4,   100] loss: 62.455524 |  Acc: 78.664% (10069/12800)
    [4,   200] loss: 125.946750 |  Acc: 78.246% (20031/25600)
    [4,   300] loss: 186.423765 |  Acc: 78.414% (30111/38400)
    Total loss: 240.281207
    
    Epoch: 4
    [5,   100] loss: 54.547970 |  Acc: 81.414% (10421/12800)
    [5,   200] loss: 110.654127 |  Acc: 80.898% (20710/25600)
    [5,   300] loss: 166.249208 |  Acc: 80.919% (31073/38400)
    Total loss: 215.950420
    
    Epoch: 5
    [6,   100] loss: 48.104260 |  Acc: 83.109% (10638/12800)
    [6,   200] loss: 99.401246 |  Acc: 82.734% (21180/25600)
    [6,   300] loss: 148.791911 |  Acc: 82.807% (31798/38400)
    Total loss: 194.453984
    
    Epoch: 6
    [7,   100] loss: 44.256123 |  Acc: 85.352% (10925/12800)
    [7,   200] loss: 90.796863 |  Acc: 84.582% (21653/25600)
    [7,   300] loss: 138.052944 |  Acc: 84.396% (32408/38400)
    Total loss: 181.198413
    
    Epoch: 7
    [8,   100] loss: 40.750996 |  Acc: 85.906% (10996/12800)
    [8,   200] loss: 84.440442 |  Acc: 85.539% (21898/25600)
    [8,   300] loss: 127.751372 |  Acc: 85.430% (32805/38400)
    Total loss: 168.756287
    
    Epoch: 8
    [9,   100] loss: 40.119882 |  Acc: 86.266% (11042/12800)
    [9,   200] loss: 79.863018 |  Acc: 86.211% (22070/25600)
    [9,   300] loss: 120.620995 |  Acc: 86.188% (33096/38400)
    Total loss: 158.876436
    
    Epoch: 9
    [10,   100] loss: 35.623312 |  Acc: 87.477% (11197/12800)
    [10,   200] loss: 75.740778 |  Acc: 86.777% (22215/25600)
    [10,   300] loss: 115.162053 |  Acc: 86.703% (33294/38400)
    Total loss: 151.007361
    
    Epoch: 10
    [11,   100] loss: 34.881428 |  Acc: 88.039% (11269/12800)
    [11,   200] loss: 71.926582 |  Acc: 87.699% (22451/25600)
    [11,   300] loss: 109.965547 |  Acc: 87.375% (33552/38400)
    Total loss: 145.488317
    
    Epoch: 11
    [12,   100] loss: 32.708189 |  Acc: 88.977% (11389/12800)
    [12,   200] loss: 66.790455 |  Acc: 88.594% (22680/25600)
    [12,   300] loss: 103.832237 |  Acc: 88.125% (33840/38400)
    Total loss: 137.913376
    
    Epoch: 12
    [13,   100] loss: 31.950675 |  Acc: 89.242% (11423/12800)
    [13,   200] loss: 65.730325 |  Acc: 88.820% (22738/25600)
    [13,   300] loss: 101.885522 |  Acc: 88.430% (33957/38400)
    Total loss: 135.173613
    
    Epoch: 13
    [14,   100] loss: 29.611200 |  Acc: 89.805% (11495/12800)
    [14,   200] loss: 62.823584 |  Acc: 89.129% (22817/25600)
    [14,   300] loss: 97.116191 |  Acc: 88.849% (34118/38400)
    Total loss: 129.670478
    
    Epoch: 14
    [15,   100] loss: 29.925015 |  Acc: 89.875% (11504/12800)
    [15,   200] loss: 62.226747 |  Acc: 89.516% (22916/25600)
    [15,   300] loss: 95.177161 |  Acc: 89.206% (34255/38400)
    Total loss: 126.196716
    
    Epoch: 15
    [16,   100] loss: 28.724815 |  Acc: 90.039% (11525/12800)
    [16,   200] loss: 60.983424 |  Acc: 89.551% (22925/25600)
    [16,   300] loss: 93.304751 |  Acc: 89.354% (34312/38400)
    Total loss: 123.906554
    
    Epoch: 16
    [17,   100] loss: 27.764434 |  Acc: 90.305% (11559/12800)
    [17,   200] loss: 57.115116 |  Acc: 90.148% (23078/25600)
    [17,   300] loss: 89.535789 |  Acc: 89.685% (34439/38400)
    Total loss: 118.733271
    
    Epoch: 17
    [18,   100] loss: 26.901688 |  Acc: 90.781% (11620/12800)
    [18,   200] loss: 56.244663 |  Acc: 90.316% (23121/25600)
    [18,   300] loss: 87.834935 |  Acc: 89.872% (34511/38400)
    Total loss: 116.597480
    
    Epoch: 18
    [19,   100] loss: 27.289408 |  Acc: 90.633% (11601/12800)
    [19,   200] loss: 57.436502 |  Acc: 90.137% (23075/25600)
    [19,   300] loss: 88.500381 |  Acc: 89.846% (34501/38400)
    Total loss: 115.314192
    
    Epoch: 19
    [20,   100] loss: 24.068543 |  Acc: 91.852% (11757/12800)
    [20,   200] loss: 53.208921 |  Acc: 90.828% (23252/25600)
    [20,   300] loss: 84.727040 |  Acc: 90.203% (34638/38400)
    Total loss: 112.072869
    Epoch    21: reducing learning rate of group 0 to 1.0000e-02.
    
    Epoch: 20
    [21,   100] loss: 17.140250 |  Acc: 94.570% (12105/12800)
    [21,   200] loss: 30.638147 |  Acc: 95.156% (24360/25600)
    [21,   300] loss: 41.904663 |  Acc: 95.573% (36700/38400)
    Total loss: 51.213734
    
    Epoch: 21
    [22,   100] loss: 7.424102 |  Acc: 97.984% (12542/12800)
    [22,   200] loss: 14.609958 |  Acc: 97.918% (25067/25600)
    [22,   300] loss: 21.400117 |  Acc: 97.964% (37618/38400)
    Total loss: 27.305064
    
    Epoch: 22
    [23,   100] loss: 5.123270 |  Acc: 98.586% (12619/12800)
    [23,   200] loss: 9.734514 |  Acc: 98.684% (25263/25600)
    [23,   300] loss: 14.615595 |  Acc: 98.648% (37881/38400)
    Total loss: 19.189702
    
    Epoch: 23
    [24,   100] loss: 3.460799 |  Acc: 99.172% (12694/12800)
    [24,   200] loss: 6.807557 |  Acc: 99.203% (25396/25600)
    [24,   300] loss: 10.049018 |  Acc: 99.211% (38097/38400)
    Total loss: 13.315432
    
    Epoch: 24
    [25,   100] loss: 2.478284 |  Acc: 99.469% (12732/12800)
    [25,   200] loss: 4.640014 |  Acc: 99.492% (25470/25600)
    [25,   300] loss: 6.763096 |  Acc: 99.505% (38210/38400)
    Total loss: 9.023635
    
    Epoch: 25
    [26,   100] loss: 1.528404 |  Acc: 99.680% (12759/12800)
    [26,   200] loss: 2.968595 |  Acc: 99.711% (25526/25600)
    [26,   300] loss: 4.535004 |  Acc: 99.706% (38287/38400)
    Total loss: 5.932488
    
    Epoch: 26
    [27,   100] loss: 1.165903 |  Acc: 99.852% (12781/12800)
    [27,   200] loss: 2.107836 |  Acc: 99.867% (25566/25600)
    [27,   300] loss: 3.091869 |  Acc: 99.875% (38352/38400)
    Total loss: 4.153865
    
    Epoch: 27
    [28,   100] loss: 0.732892 |  Acc: 99.945% (12793/12800)
    [28,   200] loss: 1.651403 |  Acc: 99.883% (25570/25600)
    [28,   300] loss: 2.452116 |  Acc: 99.888% (38357/38400)
    Total loss: 3.153302
    
    Epoch: 28
    [29,   100] loss: 0.618112 |  Acc: 99.945% (12793/12800)
    [29,   200] loss: 1.297444 |  Acc: 99.941% (25585/25600)
    [29,   300] loss: 1.891233 |  Acc: 99.943% (38378/38400)
    Total loss: 2.471666
    
    Epoch: 29
    [30,   100] loss: 0.548403 |  Acc: 99.945% (12793/12800)
    [30,   200] loss: 1.019150 |  Acc: 99.953% (25588/25600)
    [30,   300] loss: 1.501457 |  Acc: 99.958% (38384/38400)
    Total loss: 1.972135
    
    Epoch: 30
    [31,   100] loss: 0.435647 |  Acc: 99.969% (12796/12800)
    [31,   200] loss: 0.805838 |  Acc: 99.977% (25594/25600)
    [31,   300] loss: 1.260185 |  Acc: 99.971% (38389/38400)
    Total loss: 1.586358
    
    Epoch: 31
    [32,   100] loss: 0.347193 |  Acc: 99.992% (12799/12800)
    [32,   200] loss: 0.633916 |  Acc: 99.992% (25598/25600)
    [32,   300] loss: 0.942637 |  Acc: 99.995% (38398/38400)
    Total loss: 1.342172
    
    Epoch: 32
    [33,   100] loss: 0.301421 |  Acc: 99.984% (12798/12800)
    [33,   200] loss: 0.628001 |  Acc: 99.980% (25595/25600)
    [33,   300] loss: 0.972346 |  Acc: 99.982% (38393/38400)
    Total loss: 1.273002
    
    Epoch: 33
    [34,   100] loss: 0.271426 |  Acc: 100.000% (12800/12800)
    [34,   200] loss: 0.522611 |  Acc: 100.000% (25600/25600)
    [34,   300] loss: 0.791412 |  Acc: 99.997% (38399/38400)
    Total loss: 1.069014
    
    Epoch: 34
    [35,   100] loss: 0.272007 |  Acc: 100.000% (12800/12800)
    [35,   200] loss: 0.500229 |  Acc: 100.000% (25600/25600)
    [35,   300] loss: 0.741914 |  Acc: 99.997% (38399/38400)
    Total loss: 0.962418
    
    Epoch: 35
    [36,   100] loss: 0.200861 |  Acc: 100.000% (12800/12800)
    [36,   200] loss: 0.421179 |  Acc: 100.000% (25600/25600)
    [36,   300] loss: 0.637626 |  Acc: 100.000% (38400/38400)
    Total loss: 0.835531
    
    Epoch: 36
    [37,   100] loss: 0.200766 |  Acc: 100.000% (12800/12800)
    [37,   200] loss: 0.397603 |  Acc: 100.000% (25600/25600)
    [37,   300] loss: 0.606028 |  Acc: 99.995% (38398/38400)
    Total loss: 0.800073
    
    Epoch: 37
    [38,   100] loss: 0.178643 |  Acc: 100.000% (12800/12800)
    [38,   200] loss: 0.374064 |  Acc: 100.000% (25600/25600)
    [38,   300] loss: 0.577130 |  Acc: 100.000% (38400/38400)
    Total loss: 0.768444
    
    Epoch: 38
    [39,   100] loss: 0.192881 |  Acc: 100.000% (12800/12800)
    [39,   200] loss: 0.412415 |  Acc: 99.996% (25599/25600)
    [39,   300] loss: 0.607835 |  Acc: 99.997% (38399/38400)
    Total loss: 0.769075
    
    Epoch: 39
    [40,   100] loss: 0.174156 |  Acc: 100.000% (12800/12800)
    [40,   200] loss: 0.356172 |  Acc: 100.000% (25600/25600)
    [40,   300] loss: 0.544260 |  Acc: 100.000% (38400/38400)
    Total loss: 0.711841
    
    Epoch: 40
    [41,   100] loss: 0.197980 |  Acc: 99.992% (12799/12800)
    [41,   200] loss: 0.405721 |  Acc: 99.996% (25599/25600)
    [41,   300] loss: 0.596260 |  Acc: 99.997% (38399/38400)
    Total loss: 0.783890
    
    Epoch: 41
    [42,   100] loss: 0.195553 |  Acc: 99.992% (12799/12800)
    [42,   200] loss: 0.377932 |  Acc: 99.996% (25599/25600)
    [42,   300] loss: 0.565132 |  Acc: 99.997% (38399/38400)
    Total loss: 0.740863
    
    Epoch: 42
    [43,   100] loss: 0.184922 |  Acc: 100.000% (12800/12800)
    [43,   200] loss: 0.370228 |  Acc: 99.996% (25599/25600)
    [43,   300] loss: 0.563876 |  Acc: 99.997% (38399/38400)
    Total loss: 0.738950
    
    Epoch: 43
    [44,   100] loss: 0.188086 |  Acc: 100.000% (12800/12800)
    [44,   200] loss: 0.363090 |  Acc: 100.000% (25600/25600)
    [44,   300] loss: 0.529876 |  Acc: 100.000% (38400/38400)
    Total loss: 0.684271
    Epoch    45: reducing learning rate of group 0 to 1.0000e-03.
    
    Epoch: 44
    [45,   100] loss: 0.180676 |  Acc: 99.992% (12799/12800)
    [45,   200] loss: 0.349191 |  Acc: 99.996% (25599/25600)
    [45,   300] loss: 0.512983 |  Acc: 99.997% (38399/38400)
    Total loss: 0.664923
    
    Epoch: 45
    [46,   100] loss: 0.166781 |  Acc: 100.000% (12800/12800)
    [46,   200] loss: 0.320780 |  Acc: 100.000% (25600/25600)
    [46,   300] loss: 0.477024 |  Acc: 100.000% (38400/38400)
    Total loss: 0.632136
    
    Epoch: 46
    [47,   100] loss: 0.153178 |  Acc: 100.000% (12800/12800)
    [47,   200] loss: 0.315739 |  Acc: 100.000% (25600/25600)
    [47,   300] loss: 0.473674 |  Acc: 100.000% (38400/38400)
    Total loss: 0.619973
    
    Epoch: 47
    [48,   100] loss: 0.163760 |  Acc: 100.000% (12800/12800)
    [48,   200] loss: 0.322436 |  Acc: 100.000% (25600/25600)
    [48,   300] loss: 0.487261 |  Acc: 100.000% (38400/38400)
    Total loss: 0.619886
    
    Epoch: 48
    [49,   100] loss: 0.160341 |  Acc: 100.000% (12800/12800)
    [49,   200] loss: 0.328375 |  Acc: 100.000% (25600/25600)
    [49,   300] loss: 0.497179 |  Acc: 100.000% (38400/38400)
    Total loss: 0.643087
    
    Epoch: 49
    [50,   100] loss: 0.156791 |  Acc: 100.000% (12800/12800)
    [50,   200] loss: 0.309782 |  Acc: 100.000% (25600/25600)
    [50,   300] loss: 0.466129 |  Acc: 100.000% (38400/38400)
    Total loss: 0.606150
    Epoch    51: reducing learning rate of group 0 to 1.0000e-04.
    


```python
save_model = True
if save_model:
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    os.makedirs('checkpoint', exist_ok=True)
    torch.save(state, './checkpoint/res18.ckpt')
```


```python
dataiter = iter(test_loader)
images, labels = dataiter.next()
images = images[:4]
labels = labels[:4]
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images.to(device))
_, predicted = torch.max(outputs.cpu(), 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```


    
![png](ResNet_files/ResNet_22_0.png)
    


    GroundTruth:    cat  ship  ship plane
    Predicted:    cat  ship   car plane
    Accuracy of the network on the 10000 test images: 90 %
    Accuracy of plane : 90 %
    Accuracy of   car : 100 %
    Accuracy of  bird : 81 %
    Accuracy of   cat : 72 %
    Accuracy of  deer : 89 %
    Accuracy of   dog : 87 %
    Accuracy of  frog : 94 %
    Accuracy of horse : 93 %
    Accuracy of  ship : 95 %
    Accuracy of truck : 96 %
    


```python

```
