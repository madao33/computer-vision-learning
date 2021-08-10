# SeNet学习笔记及仿真

## 前言

SENet[<sup>[1]</sup>](#ref-1)是ImageNet 2017年的冠军模型，自SeNet提出后，ImageNet挑战赛就停止举办了。SENet同之前的ResNet一样，引入了一些技巧，可以在很大程度上降低模型的参数，并且提升模型的运算速度。

SENet全称Squeeze-and-Excitation Networks，中文名可以翻译为挤压和激励网络。SENet在ImageNet 2017取得了第一名的成绩，Top-5 error rate降低到了2.251%，官方的模型和代码在github仓库中可以找到[<sup>[2]</sup>](#ref-2)。

## SE block

SENet提出的动机是将通道之间的关系结合起来，于是引出了一个*Squeeze-and-excitation*（SE）块[<sup>[1]</sup>](#ref-1)，它的目的就是**通过显式建模网络卷积特征的信道之间的相互依赖性来提高网络表征的质量**。SE块的机制也可以说是**通过学习全局信息来选择性地强调有用的特征和抑制不太有用的特征**，SENet块如[fig1](#fig-1)所示。

<div id="fig-1"></div>

![](imgs/fig1.png)

SE模块可以看作是一个计算单元，用 $F_{tr}$ 表示，可以将输入 $X \in \R^{H' \times W' \times C'}$​ 映射为特征图 $U \in \R^{H \times W \times C}$​。以下的符号中，$F_{tr}$​ 表示卷积操作，$\bold{V}=[V_1, V_2, \dots, V_C]$ 来表示学习到的一组滤波器核，其中 $V_c$ 表示的是第 $c$​ 个滤波器的参数，所以输出可以表示为 $\bold{U}=[U_1, U_2, \dots, U_C]$​​，其中：

<div id="eqn-1"></div>

$$
U_c=V_c * \bold{X}=\sum_{s=1}^{C'}V_C^s * X^s
$$

[公式1](#eqn-1)中 $*$ 表示的是卷积操作，$V_c=[V_c^1, V_c^2, \dots, V_c^{C'}], \quad \bold{X}=[X^1, X^2, \dots, X^{C'}]$ 以及 $u_c \in \R^{H \times W}$， 



## 参考文献

<div id="ref-1"></div>

- [1] [Hu J, Shen L, Sun G. Squeeze-and-excitation networks[J]. arXiv preprint arXiv:1709.01507, 2017, 7.](https://arxiv.org/pdf/1709.01507.pdf)

<div id="ref-2"></div>

- [2] [hujip-frank/SENet](https://github.com/hujie-frank/SENet)

