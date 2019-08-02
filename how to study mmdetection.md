# mmdetection 学习相关
## 一、深度学习部分
### 1、深度学习500问
https://github.com/scutan90/DeepLearning-500-questions.git
#### CNN
https://www.jianshu.com/p/c0215d26d20a
<br> same  = 向上取整（  输入size                  / 步长）
<br> vaild = 向上取整（（输入size-filter size + 1）/ 步长)
<br> le-Net5 图片小
<br> AlexNet 图片大了，dropout，relu
<br> vggNet 规则卷积
<br> inception（googlenet） 多个卷积和pooling并行当成一个inception
<br> resNet 层叠的卷积跳跃连接
### 2、论文画图
http://alexlenail.me/NN-SVG/index.html
https://github.com/HarisIqbal88/PlotNeuralNet
https://cbovar.github.io/ConvNetDraw/
https://github.com/gwding/draw_convnet
## 二、目标检测部分
### 1、mAP学习
https://blog.csdn.net/hsqyc/article/details/81702437
### 2、多尺度训练/测试
https://www.zhihu.com/question/271781123
### 3、POI pooling和Align
https://zhuanlan.zhihu.com/p/73138740
## 三、pytorch部分
### 1、pytorch 中文文档
https://github.com/zergtant/pytorch-handbook
https://pytorch-cn.readthedocs.io/zh/latest/
### 2、pytorch backward
out.backward()
<br>https://www.cnblogs.com/JeasonIsCoding/p/10164948.html
<br>https://www.jianshu.com/p/a105858567df
### 3、例子
图像分类：https://blog.csdn.net/m0_37673307/article/details/81268222
<br> 20190724_torch_ImageClassification.py
## 四、mmdetection部分
   回到问题本身：最好的开源目标检测算法？是指框架还是精度最高的算法实现呢？如果都指的话大概是mmdet中的cascade mask rcnn x101 dcn版本吧？
坑点也会有一些，如果自己深度使用的话，建议把mmcv和mmdet不要装到python package，而是作为母目录下面的两个文件夹，方便自己修改。
### 1、mmdetection安装参考，但是不要装torch0.4,编译失败
https://blog.csdn.net/qq_36302589/article/details/85798060
### 2、mmdetection github
https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md
### 3、Hybrid Task Cascade
https://www.leiphone.com/news/201903/CctvkMTejB1Fvgxp.html
### 4、mmdetection介绍
https://www.jiqizhixin.com/articles/2018-10-17-10
### 5、下载预训练模型
https://github.com/open-mmlab/mmdetection/blob/master/MODEL_ZOO.md
### 6、训练自己的模型
https://github.com/hekesai/MMDetection-Introduce
<br> https://blog.csdn.net/weicao1990/article/details/93484603
<br> e.g. python tools/train.py configs/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x.py --gpus 2
### 7、源码解析
https://heary.cn/posts/mmdetection-%E5%9F%BA%E4%BA%8EPyTorch%E7%9A%84%E5%BC%80%E6%BA%90%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E7%B3%BB%E7%BB%9F/
## 五、docker部分
https://www.jianshu.com/p/f19102e0e1e6
