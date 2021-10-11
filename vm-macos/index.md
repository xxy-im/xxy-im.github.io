# 虚拟机下的iOS开发(MacOS + XCode)


手上有个很老的项目对应的苹果包需要在xcode上做些修改，苦于买不起Mac，只好出此下策

<!--more-->

现在后悔去年买的那台2070显卡的笔记本了，就很后悔，为什么不买mac  
好在家里的PC配置还可以，所以就想到用虚拟机来玩玩

## 安装VM Player
虚拟机我选的是VMware Workstation **Player**, 注意后面这个player  
个人用户用player就行了, 和Workstation相比, player免费, 体积小, 够用  

直接到[官网下载](https://www.vmware.com/go/getplayer-win)就行  

### 给VM打上MacOS补丁
距离我上一次用虚拟机装MacOS可能有八九年那么久了, 装完VM后才发现虚拟机的系统选项里已经没有Mac这个选项了, 网上查到需要通过补丁来解锁这个选项  
这里使用Auto Unlocker解锁  
  
软件直接放这了  
链接: https://pan.baidu.com/s/1SS0VCgJo9Ey1LjjTh2aqkw  
提取码: ajwo  
软件转载于[ypojie](https://www.ypojie.com/10493.html)  
  
下完解压直接unlock然后等待完成就行了  
完成后创建虚拟机的时候选骚后安装操作系统, 下一步后就可以选择Apple Mac OS了, 版本默认就行, 没多大关系
![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/CreateVM.png "创建虚拟机")

![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/60gb.png "磁盘选项")
磁盘大小这里有个坑, 建议80gb, 我这里选了60为后面安装XCode埋下了个大坑  

最后虚拟机设置里把下载来的MacOS的IOS镜像放到虚拟机驱动器里就可以了  
这是我用的镜像  
链接: https://pan.baidu.com/s/18zXlfSU6OkaifQ-aHeQVtQ  
提取码: 8ilm  
   
配置完这些后启动虚拟就可以开始安装MacOS了

![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/maconvm.png "系统安装成功")

## 安装XCode
前面有说到磁盘大小会给安装XCode埋坑  
因为我安装完系统后没有第一时间安装XCode, 我先装了些常用的软件, 反正装完一些软件后是还有40GB的磁盘空间, 然后我直接在app store上安装XCode, 下载完成后安装的过程中提示安装失败了, 我再点下载提示**可用磁盘空间不足, 无法安装此产品**  
XCode下载的大小才11个多G, 用了网上的删除Time Machine的方法也没用, 因为我系统确实是只有40GB空闲  
搞不懂为什么需要那么多空间安装, 于是想通过下载xip文件的方式来安装  
下载地址是 https://developer.apple.com/download/all/?q=xcode  我下的是12.5版本  
![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/downloadxcode.png "网页下载xcode")
下载完后点击安装还是报磁盘空间不足  
无语...  
果断关掉虚拟机, 打开虚拟机设置, 将磁盘扩展至80GB, 扩展需要花点时间  

### 虚拟机磁盘空间扩展
扩展完成后并不是Mac里也会同步分区好, 需要手动给系统分区扩容, 因为虚拟机的硬盘变大后系统的分区表信息并不会变  
Mac中打开终端输入   
```bash
    diskutil list
```
可以看到现在是磁盘信息, 比如图片中可以看到我的磁盘总空间是85.9GB但是Apple_APFS(disk0s2)只用了64.2GB, 还有21.5GB的free  
![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/beforeresize.png "系统扩容前")

现在需要将disk0s2(每个人的数字可能不一样)扩容, 因为是APFS格式, 所以用```resizeContainer```命令  
```bash
    diskutil apfs resizeContainer disk0s2 85.6GB
```
命令执行成功后可以看到扩展的21.5GB也全都加到disk0s2中了  
![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/afterresize.png)
![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/disksize.png "系统扩容后")

这时候再安装XCode就ok了, xip解压出来是29.6GB左右, 加上xip本身是11.几GB, 所以应该至少要有42GB左右的空闲空间才够
![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/xcode.png "安装成功")

## 小结
这是一段因为穷而导致的莫名奇妙的经历[doge]
