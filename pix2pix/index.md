# [论文复现] pix2pix


GAN，越来越有意思了

<!--more-->

# Image-to-Image Translation with Condition Adversarial Networks

[论文下载（CVPR）](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)

[论文下载（arxiv，更详细）](https://arxiv.org/pdf/1611.07004v3.pdf)

## 概述

![pix2pix案例](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/pix2pix.png)

论文开篇直接放了张图片告诉你这个网络可以做哪些图片到图片的翻译任务。这些任务包括但不限于语义标签图到生成图，物体边缘轮廓图到构建出的实体图，图片上色等。论文将这些任务同一称为像素到像素的映射（map pix to pix）。这篇论文的团队又是一个良心团队，不仅给了代码，还有示例网站，还给了colab页面以及网友们自己做的艺术创作。都在这里 [https://phillipi.github.io/pix2pix/](https://phillipi.github.io/pix2pix/)  

所有这些图片翻译任务都只需要用同一个网络结构，喂不同的数据就可以实现。这就是它牛逼的地方，直接给了一个通用解决方案。

因为需要输入一张图片，可以把这个输入的图片作为条件，所以这个GAN模型是有条件的（conditional GAN）。

论文提到了一个叫 **“structure loss”** 的东西，说以前的图片翻译问题通常会将输出空间认为是无结构化（”unstructured“），像素和像素之间是条件独立的（与周围的像素无关，只跟输入图片中对应的像素有关）。而cGAN就能学到一个 **“structure loss”** ，对输出图片中相邻的像素进行惩罚。

> cGAN就是在GAN的基础上加了一个条件向量。生成图片的时候在噪声后面接个条件向量，判别的时候图片也是和这个条件向量一起判别，这个条件向量在MNIST数据集上可以代表数字，CIFAR数据集上可以代表类别，总之按你给定的条件生成相应的图像。理解了GAN的话很容易就能写出cGAN的代码，所以就没写cGAN的复现。
> 

## 模型及训练

模型大体的框架是用的和DCGAN类似的结构，生成器和判别器都是 **convolution-BatchNorm-ReLU** 这样的 **CBR** 结构。但不同的是，推理过程是用测试集的统计数据进行batch normalization，当batch size为1时又叫做 **“instance normalization”**，这是图像生成任务常用的方法。参考这篇论文《Instance normalization: The missing ingredient for fast stylization》

**生成器：**

![U-Net Generator](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/unet.png)


用的是**U-Net**那样的一个U型卷积结构，是图像分割领域的经典论文，至今仍活跃于医学图像领域。

过去大部分做 **Image-to-Image** 任务的GAN的生成器都是通过对输入先下采样再上采样的方式生成图像（encoder-decoder结构）。但是这样会导致在下采样通过瓶颈层时丢失掉很多特征，而我们的任务需要输出图像与输入图像的一些底层特征的相同的，如轮廓和边缘。而 **U-Net** 结构就很好的解决了这个问题，用类似 **ResNet** 那样的方法把通过瓶颈层前的特征直接送到**对称**的上采样层上，这样就保留了图像的底层特征 。

**判别器：**

论文给取了个名字叫**马尔可夫判别器**，又叫 **PatchGAN** 分类器，这个判别器将图片分成很多小块（Patch）分别判别真假概率（Patch之间相互独立）。这样判别器的输出就不再是一个数值了，图片为真的概率为判别器输出结果平均的平均值。这么做的一个目的是为了方便捕捉图片的高频信息（纹理，边缘，风格等）。论文在 **Cityscapes** 数据集上做的 **label→photo** 实验，Patch为 70x70 得出的效果最好。

这样的判别器将一张图片视为一个马尔可夫随机场，如果像素之间的距离超过了一个Patch的直径就认为它们是独立无关的。

> **低频**就是颜色缓慢变化，也就是灰度缓慢地变化，代表着那是连续渐变的一块区域；  
> **高频**就是频率变化快，相邻区域之间灰度相差很大。

*具体代码实现的时候并不是真的把图片分成 NxN 块后再判别，而是通过改变卷积操作的感受野来实现*

**目标函数：**

除了GAN原本的目标函数，还需要一个函数评估生成图与真实图的“距离”（像素之间的差异），论文用的 **L1** 距离，选用L1是因为这些距离函数作用在像素层面上会激励图像模糊化，而L1距离相较L2来说图像的模糊程度会更少。（不会捕捉高频信息，但能捕捉到低频信息，高频信息已经丢给判别器去捕捉了）

$$
\mathcal{L}_ {L 1}(G)=\mathbb{E}_{x, y, z}\left[\|y-G(x, z)\|_{1}\right]
$$

加在原目标函数后，最终目标函数为

$$
G^{*}=\arg \min _{G} \max _{D} \mathcal{L} _{c G A N}(G, D)+\lambda \mathcal{L} _{L 1}(G)
$$

和传统cGAN还有个不同的就是，pix2pix把噪声采样 $z$ 给拿掉了，因为生成器很容易会忽略噪声输入。论文最终通过使用**dropout**来引入随机性，不单是训练过程用的dropout，推理过程也用dropout。但也提到了，这种方法带来的随机性也不是很大。

> 所以论文说如何使cGAN产生高随机性也是个重要的工作。
> 

## 评价指标

在Improved GAN 中提到过一个叫做 **Inception Score** 的评价指标。这篇论文里又提出了一个 **FCN-score** 用于语义标签转图片这个任务上评估图像生成质量。

用一个现成的FCN模型给生成图做语义分割得到的label和真实的label做比较，这时就可以用语义分割领域现有的评价指标，如 **per-pixel accuracy**，**per-class accuracy** 和 **Class IOU**。

## 核心代码

**生成器：**

```python
class UNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, normalize=True, down=True, activation=None, dropout=False):
        super().__init__()

        # 参数 4, 2, 1，在下采样是宽高缩小两倍，上采样时扩大两倍
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, 2, 1, bias=False if normalize else True) if down
            else nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1, bias=False if normalize else True),
        )
        if normalize:
            self.net.append(nn.BatchNorm2d(out_channel))

        self.net.append(nn.LeakyReLU(0.2, True) if activation is None else activation)

        if dropout:
            self.net.append(nn.Dropout(0.5))

    def forward(self, x):
        return self.net(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, init_weights=True):
        super().__init__()

        conv_channels = [64, 128, 256, 512, 512, 512, 512, 512, 512]

        self.down1 = UNetBlock(in_channels, conv_channels[0], down=True)
        self.down2 = UNetBlock(conv_channels[0], conv_channels[1], down=True)
        self.down3 = UNetBlock(conv_channels[1], conv_channels[2], down=True)
        self.down4 = UNetBlock(conv_channels[2], conv_channels[3], down=True)
        self.down5 = UNetBlock(conv_channels[3], conv_channels[4], down=True)
        self.down6 = UNetBlock(conv_channels[4], conv_channels[5], down=True)
        self.down7 = UNetBlock(conv_channels[5], conv_channels[6], down=True)

        self.bottleneck = UNetBlock(conv_channels[6], conv_channels[7], down=True)

        self.up1 = UNetBlock(conv_channels[7], conv_channels[6], down=False, activation=nn.ReLU(True))
        self.up2 = UNetBlock(conv_channels[6] * 2, conv_channels[5], down=False, activation=nn.ReLU(True), dropout=True)
        self.up3 = UNetBlock(conv_channels[5] * 2, conv_channels[4], down=False, activation=nn.ReLU(True))
        self.up4 = UNetBlock(conv_channels[4] * 2, conv_channels[3], down=False, activation=nn.ReLU(True), dropout=True)
        self.up5 = UNetBlock(conv_channels[3] * 2, conv_channels[2], down=False, activation=nn.ReLU(True))
        self.up6 = UNetBlock(conv_channels[2] * 2, conv_channels[1], down=False, activation=nn.ReLU(True), dropout=True)
        self.up7 = UNetBlock(conv_channels[1] * 2, conv_channels[0], down=False, activation=nn.ReLU(True))

        self.out = UNetBlock(conv_channels[0] * 2, in_channels, normalize=False, down=False, activation=nn.Tanh())

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        d1 = self.down1(x)      # 假设x.shape = (N, 3, 512, 512), d1.shape = （N, 64, 256, 256)
        d2 = self.down2(d1)     # (N, 128, 128, 128)
        d3 = self.down3(d2)     # (N, 256, 64, 64)
        d4 = self.down4(d3)     # (N, 512, 32, 32)
        d5 = self.down5(d4)     # (N, 512, 16, 16)
        d6 = self.down6(d5)     # (N, 512, 8, 8)
        d7 = self.down7(d6)     # (N, 512, 4, 4)

        bottleneck = self.bottleneck(d7)            # (N, 512, 2, 2)

        u1 = self.up1(bottleneck)                   # (N, 512, 4, 4)
        u2 = self.up2(torch.cat((u1, d7), 1))       # (N, 512, 8, 8)
        u3 = self.up3(torch.cat((u2, d6), 1))       # (N, 512, 16, 16)
        u4 = self.up4(torch.cat((u3, d5), 1))       # (N, 512, 32, 32)
        u5 = self.up5(torch.cat((u4, d4), 1))       # (N, 256, 64, 64)
        u6 = self.up6(torch.cat((u5, d3), 1))       # (N, 128, 128, 128)
        u7 = self.up7(torch.cat((u6, d2), 1))       # (N, 64, 256, 256)
        return self.out(torch.cat((u7, d1), 1))     # (N, 3, 512, 512)
```

**判别器：（和DCGAN的判别器挺像的）**

```python
# 默认 70x70 的感受野（patch）
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, init_weights=True):
        super().__init__()

        conv_channels = [64, 128, 256, 512]

        def cbr_block(in_channel, out_channel, normalize=True, kernel_size=4, stride=2, padding=1, activation=None):
            layers = [
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False if normalize else True),
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.LeakyReLU(0.2, inplace=True) if activation is None else activation)
            return layers

        # 感受野计算公式为 (output_size - 1) * stride + ksize
        # 倒着往上推就能算出感受野为70，最后一个output_size按1算
        self.net = nn.Sequential(
            *cbr_block(in_channels, conv_channels[0], normalize=False),
            *cbr_block(conv_channels[0], conv_channels[1]),
            *cbr_block(conv_channels[1], conv_channels[2]),
            *cbr_block(conv_channels[2], conv_channels[3], stride=1),
            *cbr_block(conv_channels[3], 1, normalize=False, stride=1, activation=nn.Sigmoid())
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        return self.net(torch.cat((x, y), 1))
```

*总体上还是能看出DCGAN的影子*

## 效果

我用的漫画人物草图上色数据集。图片有点大，最近因为网的问题没显卡跑，所以拖了这么久才更新（其实是因为懒）。

数据集我放网盘了

链接：https://pan.baidu.com/s/1vtAp96HaPBLEE6NVUljfHA?pwd=0bjz  
提取码：0bjz

*随便跑了几十个epoch，感觉效果不是很好呀，是我哪里写错了吗，可能加上关于色彩亮度的数据增强会好点吧*

![Anime Colorize](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/anime_colorize.png)

## 总结

这东西就很牛，你能想到的Image to Image任务几乎都能用这个来做，虚拟主播都能用这东西做。arxiv上的论文比正式投稿的论文上多很多示例（因为投稿限制了页数）。

完整代码

https://github.com/xxy-im/Just4GAN/tree/main/models/pix2pix

如果会web的同学也可以做一个很好玩的网站出来。（反正我不会）
  

*不能再懒下去了*
