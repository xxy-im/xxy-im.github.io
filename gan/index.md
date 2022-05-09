# [论文复现] Generative Adversarial Nets (原生GAN)


生死看淡，不服就GAN

<!--more-->

# Generative Adversarial Nets


[论文下载](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)

## 基本概览

文中提到在此之前，深度生成模型没什么进展是因为在最大似然估计和相关策略出现的许多棘手的概率问题很难找到合适的计算方式，（这里说的是G和D的Divergence吗？不懂）以及在生成模型中那些已经在分类模型里很有用的分层线性单元的优势显现不出来。所以提出了一个全新的生成模型，绕过这些难点。

> 简而言之就是以前的方法都是想着先构造出样本的概率分布函数，然后给函数提供些参数用于学习（最大化对数似然函数？）。但这东西很难算，尤其是维度比较高的时候。
> 

通过对抗(adversarial)的方式，同时训练两个模型，即生成器(Generator)，一个判别器(Discriminator)，分别用G和D表示。

生成器通过捕捉真实样本(训练数据)的数据分布用于生成数据，判别器用于对一个样本进行评估，给出其来自真实样本和由生成器生成的概率，即判别数据是real or synthesis，所以判别器其实就是个二分类模型。

> 固定G训练D，再固定D训练G，这样不断对抗的训练。论文中把G比喻成造假币的，D比喻成警察，双方互相促使着对方的技术手段进步，直到假币无法辨别。（零和博弈）
> 

## 模型及训练

论文中的生成器和判别器都是用的多层感知机(MLP)，这样便可以使用backpropagation，SGD和dropout这些手段来训练这两个模型。

生成器的输入是随机噪声(就均匀分布，高斯分布这样的东西，1维到高维都可以)。

判别器的输出是0到1的标量，越接近1表示越真。

生成器 $G$ 的目标是要使得输入噪声 $z$ 后生成的图像 $G(z)$ 在判别器 $D$ 中得到的分数 $D(G(z))$ 很高。即  $1-D(G(z))$ 要很小，论文中对其取对数，于是生成器的目标是 $\min(log(1-D(G(z)))$。

判别器 $D$ 的目标则是对于输入的真实数据 $x$ ，$D(x)$ 的值越大越好。同样对其取对数，即$max(log(D(x)))$。

综上两点，GAN的目标函数是这样的：

$$
\min_{G}\max_{D}V(D, G)=\mathbb{E}_{\boldsymbol{x}\sim p_{\text {data }}(\boldsymbol{x})}[\log{D}(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z}\sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$


> **但是在训练刚开始的时候生成的数据几乎都是一眼假，判别器给出的分数很接近0，梯度太小导致一开始很难训练，所以实际训练时候还是把目标改为$\max(logD(G(z)))$。这样目标函数就统一为了最大化交叉熵函数的负数，即最小化交叉熵函数。所以GAN的loss函数为BCELoss。**


分别对那两个对数函数求期望后相加。(其实就是交叉熵。)

固定判别器 $D$ 的情况下，目标函数最小化；（计算$D(G(z)))$与全1的交叉熵，从而优化生成器）

固定生成器 $G$ 的情况下，目标函数最大化。（计算$D(x)$与1的交叉熵 和 $D(G(z))$与0的交叉熵）

不断交替训练，然后得到一个数值解。由于训练判别器的内层循环的计算代价非常高，而且在有限的数据集上会导致过拟合。论文中采用的方法是训练 $k$ 次 $D$ 再训练 $1$ 次 $G$ 。(这里 $1$ 次是指一个mini-batch)。让 $G$ 走慢点，这样能维持 $D$ 在最优解附近。

理想状态是 $G$ 和 $D$ 有足够的容量，在不断地交替训练后，生成的数据分布和真实的数据分布重合，即 $p_g = p_{data}$， 判别器无法分别真假分布，使得 $D(x)=\frac{1}{2}$ 。

> GAN是出了名的难训练，因为容易出现****海奥维提卡现象(the Helvetica Scenario),**** 也叫做****模型坍塌(Mode collapse)****。生成器可能发现了某些输出能很好的骗过判别器，使得判别器分辨不出真假，于是生成器和判别器就都开摆了，不会再进步了。
> 

## 核心代码

前面提到了生成器和判别器其实都是多层感知机。

生成器的输出大小等于图像拉成一维向量的长度（注意像素是整型），判别器输出为图片为real的概率。比如我想在`CIFAR-10` 上跑GAN，所以生成器最后的输出为 $3\times32\times32$

**生成器**

```python
import numpy as np
import torch
from torch import nn

# 生成器
class Generator(nn.Module):
    def __init__(self, in_features, img_shape, init_weights = False):
        super().__init__()
        self.img_shape = img_shape
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, np.prod(img_shape)),
            nn.Tanh()
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, z):
        gz = self.net(z)
        return gz.view(-1, *self.img_shape)
```

---

**判别器**

```python
class Discriminator(nn.Module):
    def __init__(self, img_shape, init_weights = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.prod(img_shape), 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        x = imgs.view(imgs.shape[0], -1)
        return self.net(x)
```

---

**Loss计算**

```python
z = torch.randn((bs, 128), device=device)        # 随机噪声
real = torch.ones((bs, 1), device=device)        # 全真标签
fake = torch.zeros((bs, 1), device=device)       # 全假标签
loss = nn.BCELoss()

# 计算判别器loss
r_loss = loss(D(imgs), real)         # 识别真实图片的loss
f_loss = loss(D(gz), fake)           # 识别假图片的loss
D_loss = (r_loss + f_loss)  / 2      # 取平均

# 计算生成器loss
G_loss = loss(D(G(z)), real)
```

## 效果

论文中对于`CIFAR-10`有对比两种不同方案

1. 普通MLP的G和D
2. G使用转置卷积，D使用卷积

反正对比的图片我觉得两种效果都差不多，都不怎么好，毕竟是第一个GAN，重要的是思想。

下面是我自己在`CIFAR-10`上跑出来的效果

---

**100 epoch**：

![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/gan_epoch100.png)

**200 epoch**：

![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/gan_epoch200.png)

**600 epoch**：

![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/gan_epoch600.png)

## 总结

效果不太好，但还是能看出来它好像真的有在努力画出真实图像。训练的时候我也遇到了模型坍塌的问题，后面不管怎么train都没变化，不知道是不是权重初始化的问题。之前用的xavier权重初始化，效果更差，索性不初始化了还比之前好点，为什么会这样不知道有没有大佬解答下。

完整代码：[https://github.com/xxy-im/Just4GAN](https://github.com/xxy-im/Just4GAN)

直接 `python train.py --config ./config/vanilla.yaml` 就可以默认训练CIFAR-10了。

*不太会python，代码写的菜，轻喷。*
