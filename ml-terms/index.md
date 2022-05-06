# 机器学习中的一些评价指标名词解释


持续更新中...

<!--more-->

## TP、TN、FP、FN
- **TP**(True Positive, 真正): **实际为正，预测为正**
- **TN**(True Negative, 真负): **实际为负，预测为负**
- **FP**(False Positive, 假正): **实际为负，预测为正**
- **FN**(False Negative, 假负): **实际为正，预测为负**  
  
$$
\begin{array}{|c|c|}
\hline 
TP & FN\\\ 
\hline 
FP & TN\\\ 
\hline 
\end{array}
$$
这东西就叫**混淆矩阵**(Confusion matrix)  

- $TP+TN+FP+FN$ 为总样本数
- $TP+FN$ 为实际正样本数
- $TP+FP$ 为预测正样本数
- $TN+FP$ 为实际负样本数
- $TN+FN$ 为预测负样本数

## TPR、FPR
- **TPR**(True Positive Rate):  
正例样本被正确预测出来的比例，和[Recall](#recall召回率查全率)相等
$$
TPR = \frac{TP}{TP+FN}
$$
- **FPR**(False Positive Rate):   
误分类为正实际为负的样本占所有负样本的比例
$$
FPR = \frac{FP}{TN+FP}
$$

## Precision(精确率、查准率)
所有预测为正的样本中预测正确的比例
$$
Precision = \frac{TP}{TP+FP}
$$

## Recall(召回率、查全率)
正例样本被正确预测出来的比例
$$
Recall = \frac{TP}{TP+FN}
$$

## Accuracy(准确率)
预测正确的比例
$$
Acc = \frac{TP+TN}{TP+TN+FP+FN}
$$

## F1score
综合评价Precision和Recall的一个评价指标  
[这篇文章](https://zhuanlan.zhihu.com/p/161703182)的分析很好
$$
F1-score = \frac{2Precision\times Recall}{Precision+Recall}
$$

## PR曲线
横坐标为[Recall](#recall召回率查全率)，纵坐标为[Precision](#precision精确率查准率)  
将每个样本按置信度排序后，分别计算每个样本作为阈值情况下的Recall和Precision，然后绘制曲线图

## ROC曲线
全称为Receiver Operating Characteristic(“受试者工作特征”)  
横坐标为[FPR](#tprfpr)，纵坐标为[TPR](#tprfpr)  

## AUC(Area under Curve)
ROC曲线下的面积，介于0.1和1之间，作为数值可以直观的评价分类器的好坏，值越大越好。

## IOU(Intersection over Union)
用来预测的锚框和真实边界框(ground-truth bounding box)的交并比
$$
IOU = \frac{A\cap B}{A\cup B}
$$

## AP和mAP  
全称为Average Precision和mean Average Precision，是目标检测任务的评价指标  
在目标检测任务中  
**TP**为 $IOU > IOU_{threshold}$ 的锚框数量(同一ground-truth bounding box只计算一次)  

**FP** 为 $IOU \leq IOU_{threshold}$ 的锚框数量或者是检测到同一个 GT 的多余检测框的数量

**FN**为没有检测到的 GT 的数量  
**TN**在 mAP 评价指标中不会使用到  

**AP**是计算某一类 [PR曲线](#pr曲线)下的面积，**mAP**则是计算所有类别 [PR曲线](#pr曲线)下面积的平均值。




> VOC2010之前和VOC2010之后的mAP计算方法不同，可参考[GluonCV库](https://github.com/dmlc/gluon-cv)中的[voc_detection.py](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/utils/metrics/voc_detection.py)里的两种计算方式


## 信息熵
反映的是要表示一个概率分布需要的平均信息量
$$
H=-\sum_{i=1}^{N} p\left(x_{i}\right) \log p\left(x_{i}\right)
$$

## 交叉熵
$$
L=\frac{1}{N} \sum_{i} L_{i}=\frac{1}{N} \sum_{i}-\left[y_{i} \cdot \log \left(p_{i}\right)+\left(1-y_{i}\right) \cdot \log \left(1-p_{i}\right)\right]
$$

**多分类情况：**  
$$
L=\frac{1}{N} \sum_{i} L_{i}=-\frac{1}{N} \sum_{i} \sum_{c=1}^{M} y_{i c} \log \left(p_{i c}\right)
$$
- $M$: 类别数
- $y_{ic}$: 样本 $i$ 的真实类别为 $c$ 则该值为1，否则为0
- $p_{ic}$: 对样本 $i$ 预测为 $c$ 类的概率

## KL散度 (Kullback-Leibler Divergence)
KL散度又叫相对熵，是用于衡量两个概率分布相似性的一个度量指标。  
$$
D_{K L}(p \| q)=\sum_{i=1}^{N} p\left(x_{i}\right) \cdot\left(\log p\left(x_{i}\right)-\log \left(q\left(x_{i}\right)\right)\right.
$$
或者  
$$
D_{K L}(p \| q)=\sum_{i=1}^{N} p\left(x_{i}\right) \cdot \log \frac{p\left(x_{i}\right)}{q\left(x_{i}\right)}
$$
散度越小，说明概率 $p$ 与概率 $q$ 之间越接近，那么估计的概率分布于真实的概率分布也就越接近。

## JS散度 (Jenson’s Shannon)
由于KL散度的不对称性问题使得在训练过程中可能存在一些问题，为了解决这个问题，在KL散度基础上引入了JS散度。  
不太懂，直接看这个吧 https://blog.csdn.net/weixin_44441131/article/details/105878383

**continue...**
