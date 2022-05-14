# [论文阅读] Improved GAN


还以为终于能GAN倒CIFAR10了

<!--more-->

# Improved Techniques for Training GANs


论文下载地址：

[https://arxiv.org/pdf/1606.03498v1.pdf](https://arxiv.org/pdf/1606.03498v1.pdf)

## 基本概览

写完DCGAN的时候我说再也不用CIFAR10训练GAN了，但是这篇论文开篇就告诉我他们用这些新方法训练GAN在CIFAR10上取得了很好的效果。看到这里的时候我还以为我要跟CIFAR10要死磕到底了。

GAN它爹 **Ian Goodfellow** 也在这篇论文的团队里。在第一篇GAN论文中提到过理想状态下是GAN的对抗过程达到一个纳什均衡，但通常用梯度下降方法训练出来的模型是使得损失函数的loss更小，而不是达到纳什均衡。Ian的另外一篇论文On distinguishability criteria for estimating generative models([https://arxiv.org/pdf/1412.6515.pdf](https://arxiv.org/pdf/1412.6515.pdf))也提到过，当试图达到纳什均衡时，算法无法收敛。

于是这篇论文提出了一些用于GAN的新的结构特征和训练过程，提升了半监督学习的性能以及样本生成质量。这些新技术产生的动机是由非收敛问题的一个启发式理解（heuristic understanding）。

GAN训练的过程中，生成器和判别器都希望能最小化自己的损失函数，这样就会存在一个问题。在第一篇GAN复现中提到过，GAN的目标函数是

$$
\min_{G}\max_{D}V(D, G)=\mathbb{E}_{\boldsymbol{x}\sim p_{\text {data }}(\boldsymbol{x})}[\log{D}(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z}\sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

可以看到，对于传统的GAN训练，生成器和判别器都希望最小化它们各自的损失函数。用$J^{(D)}=(\theta^{D},\theta^{G})$ 表示判别器损失函数，$J^{(G)}=(\theta^{D},\theta^{G})$ 表示生成器损失函数，其中 $\theta^{(D)}$ 和 $\theta^{(G)}$ 分别为判别器和生成器模型的权重。当修改 $\theta^{(D)}$以减小 $J^{(D)}$时会导致 $J^{(G)}$增长，同样修改 $\theta^{(G)}$减少 $J^{(G)}$会导致 $J^{(D)}$增长。因此梯度下降法很难使得GAN收敛到纳什均衡的状态。

论文举了个栗子，有两个模型，一个需要最小化 $xy$，另一个要最小化 $-xy$。使用梯度下降法虽然可以收敛到一个平稳的点，但是无法收敛到 (0, 0)，这个是应该是 Ian 的花书里提到的栗子 *大佬就是这样，引用全是自己的论文和书* 。

下面就是论文提出的有助于收敛的一些技术

## Improved Techniques for GAN

### 特征匹配（Feature matching）

换掉了生成器的目标函数，不再是最大化判别器的输出了 （$\max(logD(G(z)))$）。新的目标是当经过判别器的中间层时，真实数据 $x$ 与生成数据 $G(z)$ 的中间层特征尽可能相似。即 $|f(x)-f(G(z))|$ 要尽可能小，$f$ 为判别器中间层输出的 feature map

完整的目标函数定义如下：

$$
\left\|\mathbb{E}_ {\boldsymbol{x} \sim p_{\text {data }}} \mathbf{f}(\boldsymbol{x})-\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})} \mathbf{f}(G(\boldsymbol{z}))\right\|_{2}^{2}
$$

这种方法相比用概率去拟合分布能更好的收敛吧，以前认为判别器输出为真的概率越大生成的数据与训练数据的分布更加拟合，而现在通过比较生成数据与真实数据通过判别器时中间输出的特征来达到拟合效果。

> 给我的直观感觉就是向着那个“黑盒”里面走了一步。
> 

## 小批量判别器 （Minibatch discriminator）

复现GAN的时候提到过GAN不容易训练的一个原因是会出现模型坍塌（Mode collapse），但是生成器总是生成同样的东西。避免这个问题的一个方法就是让判别器每次“看”一批图片 （在复现DCGAN的时候我把batch size调大了原来就是这个道理吗）。不过现在训练模型应该已经没有一张张图片训练的吧。不过他这里的minibatch操作不只是简单的读批量图片，还有一系列的计算，有点复杂。

$f(x_i) \in \mathbb{R}^{A}$ 为判别器某一层的输入向量 $x_i$ 对应的输出，然后乘上一个张量 $\dot{T} \in \mathbb{R}^{A \times B \times C}$ 得到矩阵 $M_i$ 。 批量大小为 $B$ 的输入得到 $M_i \dots M_B$，然后对不同矩阵的同一行计算 $L_1$距离再计算$exp$。令 $c_{b}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\exp(-||M_{i,b}-M_{j,b}||_{L_1})$ 

则，

$$
o\left(\boldsymbol{x}_ {i}\right)_{b}=\sum_{j=1}^{n} c_{b}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right) \in \mathbb{R}
$$

$$
o\left(\boldsymbol{x}_ {i}\right)=\left[o\left(\boldsymbol{x}_{i}\right)_{1}, o\left(\boldsymbol{x}_{i}\right)_{2}, \ldots, o\left(\boldsymbol{x}_{i}\right)_{B}\right] \in \mathbb{R}^{B}
$$

$$
o(\mathbf{X}) \in \mathbb{R}^{n \times B}
$$

最后还要讲 minibatch层的输出 $o\left(\boldsymbol{x}_{i}\right)$ 与 $f(x_i)$ 串联后再输入到下一层，感觉好复杂啊。

论文中还提到，Minibatch discrimination可以让GAN很快的生成较好的图片，而Feature matching更适合用于半监督学习的分类任务。

### 历史参数平均（Historical averaging）

在生成器和判别器的损失函数中加一项

$$
\left\|\boldsymbol{\theta}-\frac{1}{t} \sum_{i=1}^{t} \boldsymbol{\theta}[i]\right\|^{2}
$$

其中 $\boldsymbol{\theta}[i]$ 表示第 $i$ 时刻的两个模型的参数。这个方式使得模型更能想平衡点靠拢。

### 单边标签平滑（One-sided label smoothing）

如果用 $\alpha$ 代替真实样本的标签 1，用 $\beta$ 代替标签 0，就能得到一个更好的判别器

$$
D(\boldsymbol{x})=\frac{\alpha p_{\text {data }}(\boldsymbol{x})+\beta p_{\text {model }}(\boldsymbol{x})}{p_{\text {data }}(\boldsymbol{x})+p_{\text {model }}(\boldsymbol{x})}
$$

> 这公式的原公式在GAN论文中也有说明

为防止 $p_{data}(x)$ 接近 0 而 $p_{model}(x)$ 很大导致无法向真实数据拟合，只用 $\alpha$ 替换 1，负样本的 0 标签保持不变。因此叫做单边smooth。

### 虚拟批量归一化（Virtual Batch Normalization）

BN层虽然很有用，但是也会使得神经网络对于一个输入 $x$ 的输出高度依赖于同一批次中的其他一些输入 $x^\prime$ 。因此论文提出了一个虚拟批量归一化方法（VBN），在这个过程中，每个输入 $x$ 的归一化都基于一些输入的作为参考批量（reference batch）收集来的统计数据以及 $x$ 本身， 这些输入在训练的一开始就选择好并且固定了。参考批量仅使用其自身的统计数据进行规范化处理

VBN的计算代价很大，因为它需要在两个小批量数据上运行前向传播，所以我们只在生成器网络中使用它。

## 图片质量评估

之前复现DCGAN的时候也提到了，不是epoch越多生成图片质量越好，希望后面论文能有更好的评估方法。这不就来了吗。

**Inception Score :**

论文提出的方式是将所有生成的图片 $\boldsymbol{x}$ 输入进一个 Inception 分类模型得到它的条件标签分布 $p(y|\boldsymbol{x})$。包含有意义物体的图像应该有一个较低熵的条件标签分布 $p(y|\boldsymbol{x})$。而对于积分$\int p(y \mid \boldsymbol{x}=G(z)) d z$ 若有一个较高的熵则表明生成的图像具有多样性。综合考虑这两点得到了 **IS分数** 的计算公式：

$$
\exp \left(\mathbb{E}_{\boldsymbol{x}} \operatorname{KL}(p(y \mid \boldsymbol{x})|| p(y))\right)
$$

> 在网上查了下，IS分数虽然已经有了广泛的应用程度，但还是有很多缺陷的。
> 

## 半监督学习

标准的分类模型都是有监督的，通常使用交叉熵函数训练。

论文提出了一个将GAN运用于任何标准分类分类模型实现半监督学习的方法。

比如在 K 分类模型中，将GAN的生成的图片作为第 K+1 类，相应的这个分类模型的输出也改为 K+1类。这时 $p_{model}(y=K+1|\boldsymbol{x})$ 表示 $\boldsymbol{x}$ 为假的概率，对应GAN里面的 $1-D(\boldsymbol{x})$。

这样我们通过最大化 $p_{model}(y=1\dots K|\boldsymbol{x})$ 就可以使用无标注的数据上分类模型上训练了，只要我们知道它对应于真实类别 $(1\dots K)$ 的哪一类。

这时，损失函数变成这样：

$$
\begin{aligned}
L &=-\mathbb{E}_ {\boldsymbol{x}, y \sim p_{\text {data }}(\boldsymbol{x}, y)}\left[\log p_{\text {model }}(y \mid \boldsymbol{x})\right]-\mathbb{E}_ {\boldsymbol{x} \sim G}\left[\log p_ {\text {model }}(y=K+1 \mid \boldsymbol{x})\right] 
\end{aligned}
$$
$$
=L_ {\text {supervised }}+L_ {\text {unsupervised }}, \text { where } 
$$
$$
L_ {\text {supervised }} =-\mathbb{E}_ {\boldsymbol{x}, y \sim p_ {\text {data }}(\boldsymbol{x}, y)} \log p_ {\text {model }}(y \mid \boldsymbol{x}, y<K+1) 
$$
$$
L_ {\text {unsupervised }} =- \mathbb{E}_ {\boldsymbol{x} \sim p_ {\text {data }}(\boldsymbol{x})} \log \left[1-p_ {\text {model }}(y=K+1 \mid \boldsymbol{x})\right]+\mathbb{E}_ {\boldsymbol{x} \sim G} \log \left[p_ {\text {model }}(y=K+1 \mid \boldsymbol{x})\right]
$$

> 可以看出无监督的损失函数就是来自标准的GAN

文中还提到了一个将这个分类模型作为GAN中的判别器同 $G$ 一同训练的方法，这种方法使得G和分类器之间产生了互动。

## 实验部分

论文对 **MINST**，**CIFAR-10**，**SVHN**和**ImageNet** 四个数据集进行了实验，我就只看CIFAR-10的吧，毕竟要跟它死磕到底。

GAN中的判别器是中使用了9层带dropout和weight normalization的深度卷积网络。而生成器是一个带BN层的4层深度卷积网络。

分别使用了**Feature matching**和**minibatch discrimination**的半监督学习。说实话论文图片的效果还是不咋地，跟我用DCGAN跑出来的效果差不多。

*就是这效果图打消了我复现代码的念头，于是标题由论文复现改为了论文阅读*

## 总结

虽然感觉没有啥太明显的效果，但是论文的提出的这些技术都是很好的，不知道后面的提出的 **GAN** 有没有用到这篇论文里的东西。
