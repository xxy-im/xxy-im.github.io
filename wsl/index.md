# Windows下运行Linux的正确姿势


最近需要在Linux下跑写些小程序，但是平时更多时候都离不开Windows~~打游戏~~。所以打算使用Win下的Linux子系统

<!--more-->

## 启动WSL功能
首先在控制面板的打开或启动Windows程序中将Linux子系统功能勾选上，点确认后会提示重启计算机
![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/control-panel.png "控制面板->程序->启动或关闭Windows功能")  


## 安装Linux子系统
这里我选择了一个最方便直接的方法，在Windows商店下载安装，直接在商店搜索WSL，Ubuntu，或者Linux就能找到，比如我安装的是Ubuntu 20.04
![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/ubuntu-wsl.png "安装Ubuntu子系统")

安装完成打开后输入用户名密码就可以使用了。  
现在最新的WSL2是可以支持GPU的，所有一些跑Deep的小伙伴可以试试，可以在Windows命令行中输入如下命令查看当前的WSL版本，因为我不需要用到子系统的GPU，所有我没有升级到WSL2，有需要的可以自行找下教程
```bash
    wsl --list -v
``` 
如果没有Windows商店没有满足你要求的Linux子系统，网上貌似也有教程教你运行各种不同的Linux子系统。
## 文件共享
### 子系统访问Windows
在子系统的bash中``` cd /mnt ```可以看到Windows下的磁盘已经被挂载到子系统下，可以直接copy需要的文件到子系统中
### Windows访问子系统文件
子系统的磁盘空间对应Windows下的存储目录默认是在``` C:\Users\用户名\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu20.04onWindows_79rhkp1fndgsc(这里改为自己的目录)\LocalState\rootfs ``` 

## 关于子系统桌面安装  
这部分没内容，因为我并不推荐为Linux子系统安装桌面环境。  

## 小结
后续使用过程中当然还会遇到许多坑，毕竟还有很多地方不成熟，比如使用ssh的时候可能会有端口占用问题，Windows访问子系统的权限问题等等。但是相对虚拟机来说，确实方便和实用许多，从系统功能完整性来说，个人认为是在虚拟机之下，Cygwin之上，毕竟Cygwin只是假装自己是个Linux，而WSL是实实在在的用Windows API实现Linux，对于用户层来说就是是实在在的Linux。
