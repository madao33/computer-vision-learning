# VGGNet网络学习笔记

## 引言

VGGNet（Visual Geometry Group）[<sup>[1]</sup>](#ref-1)是2014年又一个经典的卷积神经网络，VGGNet的主要的想法是如何设计网络架构的问题，VGGNet在2014年获得了分类第二，定位第一的成绩，而分类第一的成绩由GoogLeNet[<sup>[2]</sup>](#ref-2)获得。

## VGGNet论文笔记

### VGGNet架构

* 卷积层的输入是一个固定大小的 224 x 224 尺寸的RGB图像。唯一做的预处理就是减去平均的RBG值，在卷积层使用了非常小的感视野大小的过滤器3x3（这是捕捉左/右、上/下，中心概念的最小尺寸）。在其中一种配置中，还用到了 1 x 1 的卷积滤波器，可以看做是输入通道的线性变换。卷积的步长固定为1，卷积层的空间填充使得空间分辨率在卷积之后得到保留。
* 空间池化是由五个最大池化层实现的（不是所有的卷积层之后就是最大池化层），最大池化层是在步长为2，尺寸为2x2的滤波器下实现的。

* 卷积池化处理之后就是全连接层，前两个层各有4096个神经元，第三层是1000个神经单元，表示ILSVRC的1000中分类，最后一层是softmax层，网络中的全连接层的配置一致。

所有的隐藏层之后都紧跟着ReLU激活函数，在VGGNet中没有用到AlexNet[<sup>[3]</sup>](#ref-3)中的LPN，VGGNet的作者认为这种规范化方法并不能提升模型的性能，反而会导致增加内存消耗和计算时间。

### 参数详解

![](https://www.madao33.com/media/VGGNet网络学习笔记/convnet_configuration.png)

VGGNet的不同的类型如上表所示，没一列表示不同配置的VGGNet，每个网络的深度都有所不同，从A的11个权重层（8个卷积层和3个全连接层）到网络E中的19个权重层（16个卷积层和3个全连接层）。

虽然VGGNet的深度很大，但是权重参数的数量并不比AlexNet[<sup>[3]</sup>](#ref-3)的权重数量多，VGGNet每个配置的参数如下表2所示。

![](https://www.madao33.com/media/VGGNet网络学习笔记/parameter_num.png)

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

![](https://www.madao33.com/media/VGGNet网络学习笔记/comparison.png)

VGGNet的结果是**top-1 val error 23.7**，**top-5 val error 6.8**, **top-5 test error 6.8%**

## 参考文献

<div id="ref-1"></div>

- [1] [Simonyan K ,  Zisserman A . Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. Computer Science, 2014.](https://arxiv.org/pdf/1409.1556.pdf)

<div id="ref-2"></div>

- [2] [Szegedy C , Liu W , Jia Y , et al. Going Deeper with Convolutions[J]. IEEE Computer Society, 2014.](https://arxiv.org/pdf/1409.4842.pdf)

<div id="ref-3"></div>

- [3] [Krizhevsky A , Sutskever I , Hinton G . ImageNet Classification with Deep Convolutional Neural Networks[J]. Advances in neural information processing systems, 2012, 25(2).](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

