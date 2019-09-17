>本文包含大量行内公式，将公式转为图片会导致各种排版问题，建议您使用浏览器插件[MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)渲染该页公式。后续我们会将该文档迁移至[PaddlePaddle官网](https://www.paddlepaddle.org)。

<div align="center">
  <h3>
    <a href="usage.md">
      使用文档
    </a>
    <span> | </span>
    <a href="demo.md">
      示例文档
    </a>
    <span> | </span>
    <a href="model_zoo.md">
      Model Zoo
    </a>
  </h3>
</div>

---
# Paddle模型压缩工具库算法原理介绍

## 目录

- [剪裁原理介绍](#1-卷积核剪裁原理)




## 1. 卷积核剪裁原理

该策略参考paper: [Pruning Filters for Efficient ConvNets](https://arxiv.org/pdf/1608.08710.pdf)

该策略通过减少卷积层中卷积核的数量，来减小模型大小和降低模型计算复杂度。

### 1.1  剪裁卷积核

**剪裁注意事项1**
剪裁一个conv layer的filter，需要修改后续conv layer的filter. 如**图4**所示，剪掉Xi的一个filter，会导致$X_{i+1}$少一个channel, $X_{i+1}$对应的filter在input_channel纬度上也要减1.


<p align="center">
<img src="images/tutorial/pruning_0.png" height=200 width=600 hspace='10'/> <br />
<strong>图4</strong>
</p>


**剪裁注意事项2**

如**图5**所示，剪裁完$X_i$之后，根据注意事项1我们从$X_{i+1}$的filter中删除了一行（图中蓝色行），在计算$X_{i+1}$的filters的l1_norm(图中绿色一列)的时候，有两种选择：
算上被删除的一行：independent pruning
减去被删除的一行：greedy pruning

<p align="center">
<img src="images/tutorial/pruning_1.png" height=200 width=450 hspace='10'/> <br />
<strong>图5</strong>
</p>

**剪裁注意事项3**
在对ResNet等复杂网络剪裁的时候，还要考虑到后当前卷积层的修改对上一层卷积层的影响。
如**图6**所示，在对residual block剪裁时，$X_{i+1}$层如何剪裁取决于project shortcut的剪裁结果，因为我们要保证project shortcut的output和$X_{i+1}$的output能被正确的concat.


<p align="center">
<img src="images/tutorial/pruning_2.png" height=240 width=600 hspace='10'/> <br />
<strong>图6</strong>
</p>

### 1.2 Uniform剪裁卷积网络

每层剪裁一样比例的卷积核。
在剪裁一个卷积核之前，按l1_norm对filter从高到低排序，越靠后的filter越不重要，优先剪掉靠后的filter.


### 1.3 基于敏感度剪裁卷积网络

根据每个卷积层敏感度的不同，剪掉不同比例的卷积核。

#### 两个假设

- 在一个conv layer的parameter内部，按l1_norm对filter从高到低排序，越靠后的filter越不重要。
- 两个layer剪裁相同的比例的filters，我们称对模型精度影响更大的layer的敏感度相对高。

#### 剪裁filter的指导原则

- layer的剪裁比例与其敏感度成反比
- 优先剪裁layer内l1_norm相对低的filter

#### 敏感度的理解

<p align="center">
<img src="images/tutorial/pruning_3.png" height=200 width=400 hspace='10'/> <br />
<strong>图7</strong>
</p>

如**图7**所示，横坐标是将filter剪裁掉的比例，竖坐标是精度的损失，每条彩色虚线表示的是网络中的一个卷积层。
以不同的剪裁比例**单独**剪裁一个卷积层，并观察其在验证数据集上的精度损失，并绘出**图7**中的虚线。虚线上升较慢的，对应的卷积层相对不敏感，我们优先剪不敏感的卷积层的filter.

#### 选择最优的剪裁率组合

我们将**图7**中的折线拟合为**图8**中的曲线，每在竖坐标轴上选取一个精度损失值，就在横坐标轴上对应着一组剪裁率，如**图8**中黑色实线所示。
用户给定一个模型整体的剪裁率，我们通过移动**图5**中的黑色实线来找到一组满足条件的且合法的剪裁率。

<p align="center">
<img src="images/tutorial/pruning_4.png" height=200 width=400 hspace='10'/> <br />
<strong>图8</strong>
</p>

#### 迭代剪裁
考虑到多个卷积层间的相关性，一个卷积层的修改可能会影响其它卷积层的敏感度，我们采取了多次剪裁的策略，步骤如下：

- step1: 统计各卷积层的敏感度信息
- step2: 根据当前统计的敏感度信息，对每个卷积层剪掉少量filter, 并统计FLOPS，如果FLOPS已满足要求，进入step4，否则进行step3。
- step3: 对网络进行简单的fine-tune，进入step1
- step4: fine-tune训练至收敛
