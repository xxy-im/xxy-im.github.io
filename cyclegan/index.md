# [论文复现] CycleGAN


GAN来GAN去的CycleGAN

<!--more-->

# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

论文下载地址：  
[https://arxiv.org/pdf/1703.10593.pdf](https://arxiv.org/pdf/1703.10593.pdf)

文章其实早就写好了，但是一直没有卡测试代码所以就没发，~~这次真不是懒~~

最后还是只有一块很差的卡给我用，一个epoch要跑半小时☹️

## 概述

![与pix2pix的区别](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/unpaired.png)

这篇论文和pix2pix是同一个团队，pix2pix的输入输出图片属于是像素级对应的，训练时需要成对的样本数据集（paired）。而这篇论文和pix2pix相比多了一个**Unpaired**，相较pix2pix往前再走了一步，可以不需要成对的样本便能训练出较好的图像迁移模型。从论文开篇的风格迁移示例可以看出，A图转成B图的同时B也可以转成A，所以称为**CycleGAN**。换句话说，CycleGAN解决的是两个不同的图像域之间相互转换的问题。

*有前面几篇GAN的基础的话，很容易就能看懂这篇论文*

![Cycle-consistency loss](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/cyc-loss.png)

CycleGAN和之前学过的GAN不同的地方是它有两个GAN（两个生成器，两个判别器），以及两个**cycle-consistency loss**。X和Y是两个不同风格的数据集，生成器$G$为$X$风格照片生成对应的Y风格的照片，即$G(X)\rightarrow Y$。判别器$D_X$则判别图片是否为$X$的风格。生成器$F$则将输入的$Y$风格的照片转化为$X$风格的照片，即$F(Y)\rightarrow X$。判别器$D_Y$则判别图片是否为$Y$风格的照片。而**cycle-consistency loss**可以计算原风格图与对应生成图片转回为原风格图片，即$X$与$F(G(X))$之间的误差，防止相互转换时丢失信息。这个$F$就有点类似$G$的反函数的感觉，而$G$就像是$X$和$Y$两个图像域之间的双射函数（两个域的图片数量不需要一样）。论文里用了一个语言翻译的例子描述**cycle consistent（循环一致性）**。一句英文翻译成对应的法文后，这句法文翻译回英文时能得到和原英文一样的句子，这就叫**循环一致性**。而cycle-consistency loss 的目的就是使得$F(G(x))\approx x$以及$G(F(y))\approx y$。将这两个cycle-consistency loss与两个GAN的loss结合起来便是整个CycleGAN的loss。

## 模型及训练

### 生成器

生成网络参考的是Justin Johnson（李飞飞实验室）论文《Perceptual Losses for Real-Time Style Transfer and Super-Resolution》里面的用到了图像风格迁移网络。

![CycleGAN 生成器](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/residual-gen.png)  
  

对于不同的图片输入大小残差块数量不同。若输入图片大小为128x128则残差块有6个，大于等于256x256的有9个。在论文的附录7中给出了详细的网络结构。  

**6残差块结构：**  

c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3

**9残差块结构：**  

c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3

> - c7s1-64表示卷积核大小为7，步长为1，输出通道为64的卷积层
> - d128表示输出通道为128的下采样卷积层
> - R256表示输出通道为256的残差块
> - u128表示输出通道为128的上采样卷积层

上采样通过stride=1/2实现，即将输入特征先扩大一倍再通过stride为1的卷积核计算，这样就等同于stride为1/2。  

对于前后两个**c7s1**的卷积层是不改变输入特征宽高的，所以padding为3。风格迁移中常用的padding操作是**镜像填充（ReflectionPad）**，镜像填充的方式相比于使用固定数值进行填充有可能获得更好的卷积结果。  

### 判别器

判别器和pix2pix的判别器一样用的是 70x70的PatchGAN，略过。  

### 目标函数

**Cycle-consistency loss：**  

$$
\mathcal{L}_ {\text {cyc }}(G, F) =\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\|F(G(x))-x\|_{1}\right] + \mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\|G(F(y))-y\|_{1}\right]
$$

有点类似**pix2pix**里在原loss后面加的那个L1距离，因为cycle-consistency loss得保证$x$同$F(G(x))$之间的**paired**。*cyc loss 的引入使得训练时两个生成器需要同时更新参数（可以使用itertools.chain）*  

**CycleGAN完整的目标函数：**  

$$
\begin{aligned}\mathcal{L}\left(G, F, D_{X}, D_{Y}\right) &=\mathcal{L}_{\mathrm{GAN}}\left(G, D_{Y}, X, Y\right) \\&+\mathcal{L}_{\mathrm{GAN}}\left(F, D_{X}, Y, X\right) \\&+\lambda \mathcal{L}_{\text {cyc }}(G, F)\end{aligned}
$$

$\lambda$用于控制两个目标之间的循环一致性强度。论文中训练时$\lambda$设为10

### 训练细节

为了模型训练的稳定性，GAN的loss函数不再使用原生GAN那样的对数似然函数，而是使用最小二乘损失函数。即对于训练生成器$G$，目标是最小化$\mathbb{E}_ {x \sim p_ {\text {data }}(x)}\left[(D(G(x))-1)^{2}\right]$，对于训练判别器$D$，目标是最小化$\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[(D(y)-1)^{2}\right] + \mathbb{E}_{x \sim p_{\text {data }}(x)}\left[D(G(x))^{2}\right]$。这样可以使得GAN模型训练更稳定，且生成更高质量的结果。（参考 Least squares generative adversarial networks. In CVPR. IEEE, 2017.）  

还有一个防止模型震荡的方法。维护一个容量为50的图片缓存用于存储之前生成的50张图片，而不是使用生成器最新生成的图片。（参考 Learning from simulated and unsupervised images through adversarial training. In CVPR, 2017.）  

使用Adam优化器，batch size设为1，学习率为0.0002，前100个epoch学习率相同，后100个epoch学习率线性衰退为0。所以需要用到**lr_scheduler**。  

模型权重初始化使用高斯分布$N(0, 0.02)$  

对于**painting→photo**任务，论文中提到可以在损失函数中多加一项$\mathcal{L}_{\text {identity }}(G, F)$

$$
\mathcal{L}_ {\text {identity }}(G, F)=\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\|G(y)-y\|_{1}\right]+   \mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\|F(x)-x\|_{1}\right] 
$$

这样可以使生成图与原图保持色彩分布的一致性  

## 核心代码

**生成器：**

```python
def build_cbr_block(in_channels, out_channels, kernel_size, stride=1, padding=1,
                    activation=None, normalize=True, reflect_pad=True):
    layers = []

    if reflect_pad:
        layers.append(nn.ReflectionPad2d(padding))
        padding = 0

    layers.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False if normalize else True
        ),
    )

    if normalize:
        layers.append(nn.InstanceNorm2d(out_channels))

    if activation is not None:
        layers.append(activation)

    return layers

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, reflect_pad=True):
        super().__init__()

        self.block = nn.Sequential(
            *build_cbr_block(in_channels, in_channels, 3, 1, 1, nn.ReLU(True)),
            *build_cbr_block(in_channels, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualGenerator(nn.Module):
    def __init__(self, in_channels=3, res_nums=6, reflect_pad=True, init_weights=True):
        super().__init__()

        conv_channels = [64, 128, 256, 128, 64, 3]

        blocks = [
            *build_cbr_block(in_channels, conv_channels[0],
                             kernel_size=7, stride=1, padding=3, activation=nn.ReLU(True)),

            # downsampling
            *build_cbr_block(conv_channels[0], conv_channels[1],
                             kernel_size=3, stride=2, padding=1, activation=nn.ReLU(True), reflect_pad=False),
            *build_cbr_block(conv_channels[1], conv_channels[2],
                             kernel_size=3, stride=2, padding=1, activation=nn.ReLU(True), reflect_pad=False),
        ]

        for _ in range(res_nums):
            blocks.append(ResidualBlock(conv_channels[2], reflect_pad))

        # upsampling
        blocks += [
            nn.Upsample(scale_factor=2),
            *build_cbr_block(conv_channels[2], conv_channels[3],
                             kernel_size=3, stride=1, padding=1, activation=nn.ReLU(True), reflect_pad=False),
            nn.Upsample(scale_factor=2),
            *build_cbr_block(conv_channels[3], conv_channels[4],
                             kernel_size=3, stride=1, padding=1, activation=nn.ReLU(True), reflect_pad=False)
        ]

        # map to RGB
        blocks += [
            *build_cbr_block(conv_channels[4], conv_channels[5],
                             kernel_size=7, stride=1, padding=3, activation=nn.Tanh(), normalize=False)
        ]

        self.net = nn.Sequential(*blocks)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
```

**Cycle-consistency loss：**

```python
cyc_loss = torch.nn.L1Loss()
loss_cyc_a = cyc_loss(G_BA(fake_B), real_A)
loss_cyc_b = cyc_loss(G_AB(fake_A), real_B)
loss_cycle = (loss_cyc_a + loss_cyc_b) / 2
```

**L_identity：**

```python
l_identity = torch.nn.L1Loss()
loss_id_a = l_identity(G_BA(real_A), real_A)
loss_id_b = l_identity(G_AB(real_B), real_B)
loss_identity = (loss_id_a + loss_id_b) / 2
```

## 效果

训练速度实在感人，所以最后只跑了几十个epoch。感觉照片转梵高风格的效果要比梵高转照片效果好。梵高风格画转照片的画我发现有水有云的转换效果最好，碰到人物画像多半不行。截了些相对较好的效果图出来。  

数据集下载：  

链接：https://pan.baidu.com/s/1TTu-fe4J2FaJW42RmTMJvg?pwd=d6az  
提取码：d6az

![梵高风格转照片](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/vangogh2photo.png)
  
![照片转梵高风格画](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/photo2vangogh.png)


完整代码：  
https://github.com/xxy-im/Just4GAN/tree/main/models/cyclegan

## 总结

~~没啥好总结的，没卡玩个毛深度学习啊~~

继续GAN

*最近有几位新朋友关注了我，突然又有了继续写下去的动力。*
