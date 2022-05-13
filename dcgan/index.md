# [论文复现] Deep Convolutional Generative Adversarial Nets (DCGAN)


GAN开始卷起来了

<!--more-->

# Deep Convolutional Generative Adversarial Nets


[论文下载](https://arxiv.org/pdf/1511.06434v2.pdf)

## 基本概览

这篇论文给我的第一印象是很长，有16页那么多，之前看的论文基本都10页左右。论文以现在的眼光来看会觉得用CNN替换掉原始GAN中的MLP是很理所当然的事情，但论文提到在当时CNN在无监督学习中的应用是不怎么被关注的

论文的贡献有：

- 提出并验证了卷积GANs网络结构上的一些限制，使其在大多数情况下能稳定训练（即DCGAN）
- 使用与训练好的图像分类器作为判别器，与其他无监督算法相比有更好的性能
- 可视化了GAN学到的滤波器，实验表明不同的滤波器能绘制出不同的图像
- 展现了生成器的一些有趣的向量运算属性，这使得我们可以对生成样本的语义质量做一些简单的修改

然后提到了之前的图像生成模型在生成想MNIST这种数据集虽然还可以，但是在生成自然图片上效果还是不行（让我想到我在CIFAR-10上跑的GAN，效果惨不忍睹）

## 模型及训练

![DCGAN 生成器结构图 （LSUN数据集）](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/dcgenerator.png)



在此之前已经有人尝试过用CNN扩展GAN，但是都失败了。作者最开始使用监督学习领域常用的CNN结构试图扩展GAN时也失败了。但是在做了一番模型探索后，确定了一类结构族，这些结构能够提供稳定的训练，并能够训练更高分辨率和更深层次的生成模型（*卡多就可以为所欲为吗*）

核心方法采用了三个CNN架构的改进方法：

1. 全卷积网络：使用逐步卷积代替确定性空间池化函数（如maxpooling），这样网络可以自己学习空间下采样。用于生成器和判别器中便可以自行学习图像的上下采样（上下采样就是放大缩小）
2. 消除最顶层卷积层的全连接层，就如图像分类里常用的global average pooling那样。（一整个通道做一个average pooling，输出一个值），可以增强模型稳定性，但减缓了收敛速度
3. 使用Batch Normalization。但是直接对所有的层采用批处理规范化会导致样本震荡和模型不稳定，可以通过对生成器的输出层和辨别器的输入层不采用批处理规范化来避免这种情况。

生成器输出层使用Tanh激活函数，其他层使用ReLU激活函数。而在判别器上则使用LeakyReLU激活函数效果更好，特别是在高分辨率图像上。

论文给出了详细的训练细节（太良心了），除了将像素缩放到Tanh的范围[-1, 1]之外，图像没做任何预处理。使用mini-batch SGD训练，batch size为128。权重初始化用的 $(0, 0.02^2)$的正态分布初始化。LeakyReLU的p设为0.2。使用Adam优化器，学习率为0.0002，beta1设为0.5。

> 终于知道论文为什么这么长了，真的太详细了。
> 论文剩下部分是一大堆关于验证和可视化的东西。

## 核心代码

**卷积层输出大小计算公式：**

$$
N=(W-K+2P)/S+1
$$

```
N: 输出大小
W: 图像宽高
K: 卷积核大小
P: 填充值大小
S: 步长大小
```

**转置卷积层输出大小计算公式：**

$$
N=(W-1)\times S-2P+K
$$

**生成器：**

```python
# 生成器
class DCGenerator(nn.Module):
    def __init__(self, in_features, img_shape, init_weights=True):
        super().__init__()
        self.img_shape = img_shape

        # 默认每次放大2倍宽高，用于上采样
        def upsampling_block(in_channel, out_channel, normalize=True, activation=None, kernel_size=4, stride=2, padding=1):
            layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True) if activation is None else activation)
            return layers

        self.linear = nn.Sequential(
            # BN层前面的层bias可以为False
            nn.Linear(in_features, 1024 * np.prod(self.img_shape[1:]), bias=False),
            nn.BatchNorm1d(1024 * np.prod(self.img_shape[1:])),
            nn.ReLU()
        )

        self.net = nn.Sequential(
            *upsampling_block(1024, 512),       # 8 * 8
            *upsampling_block(512, 256),        # 16 * 16
            *upsampling_block(256, 128),        # 32 * 32
            *upsampling_block(128, 3, False, nn.Tanh())     # 64 * 64
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], *self.img_shape)     # 变换成二维用于卷积
        return self.net(x)
```

**判别器：**

```python
# 判别器
class DCDiscriminator(nn.Module):
    def __init__(self, img_shape, init_weights=True):
        super().__init__()

        # 默认每次缩小2倍宽高，用于下采样
        def downsampling_block(in_channel, out_channel, normalize=True, activation=None, padding=1):
            layers = [nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.LeakyReLU(0.2, inplace=True) if activation is None else activation)
            return layers

        self.net = nn.Sequential(
            *downsampling_block(3, 128, False),     # 32 * 32
            *downsampling_block(128, 256),          # 16 * 16
            *downsampling_block(256, 512),          # 8 * 8
            *downsampling_block(512, 1024),         # 4 * 4
            *downsampling_block(1024, 1, False, activation=nn.Sigmoid(), padding=0),
            #nn.AdaptiveAvgPool2d((1, 1)), nn.Sigmoid()
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, images):
        y = self.net(images)
        return y.view(y.shape[0], -1)
```

判别器的输出层那里我看网上的代码基本都是用的padding为0的卷积层，然后我有在动漫头像数据集上试过论文提到的全局average pooling层（注释的代码）。同样训练了一个epoch后，用全局池化的效果的确是差一些，但训练速度提升了点，没试过一直训练下去会怎么样

> 其余部分与原始的GAN没什么太大区别

## 效果

我依然是在CIFAR-10上训练的，虽然论文中写到他们从未在CIFAR-10上训练过，但为了和之前做的GAN有个直观的对比，所以还是在CIFAR-10上训练。

> 虽然作者没在CIFAR-10训练，但是他们在ImageNet-1k上做的预训练模型在CIFAR-10上提取特征后在分类的准确度仍然很高，说明这个模型有很高的鲁棒性
> 

一开始在CIFAR-10上训练的是时候一直没什么效果，经过在动漫头像上的效果对比后排除了模型了问题，所以那就是数据分布的问题了，于是便把batch size 调大一点（由原论文的128调到512），让模型一次”看到“的数据多一点，效果立竿见影，终于跑出了像样的图片了。

**30 epoch：**

![30epoch on CIFAR-10](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/dcgan_30e.png)

**50 epoch：**

![50epoch on CIFAR-10](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/dcgan_50e.png)

**80 epoch：**

![80epoch on CIFAR-10](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/dcgan_80e.png)

*比原生GAN好点，但还是看不出生成的到底是啥，再也不到 CIFAR-10上跑GAN了*

> 并不是epoch越多效果就更好，有可能20epoch的时候效果已经还可以，30的时候又很差，40epoch又好起来了。单看loss很难确定哪个效果好，不知道后面的论文有没有更好的验证方法。
> 

## 总结

没有像论文里那样先做预训练。

一开始我在CIFAR10上跑的时候loss没有像正常的GAN那样起伏，调了很久，最后发现原因是判别器的输出层接了BN层导致的。因为输出的是概率，被BN层一处理就会有问题了。应该是个常识问题，我傻逼了。

还有就是不同数据集效果也差很大，像动漫头像（CrypkoFaces）那样的数据集训练一个epoch就能有明显效果。可能因为动漫头像就一类数据，数据分布比较简单更容易拟合，而像CIFAR10那样的分类数据集的分布要复杂点，

**One epoch on CrypkoFaces：**

![DCGAN on Crypko](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/crypko_1epoch.png)

完整代码：[https://github.com/xxy-im/Just4GAN](https://github.com/xxy-im/Just4GAN)

直接 `python train.py --config ./config/dcgan.yaml` 就可以默认训练CIFAR-10了。

默认训练CIFAR10，如果需要训练自定义数据可能需要改几行代码

*coding十分钟，debug俩小时*
