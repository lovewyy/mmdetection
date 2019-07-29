# mmdetection install总结
## 一、说明
安装的mmdetection版本为0.6.0(2019.7.20时最新版)，安装在open-mmlab环境上(mmcv/mmdetection)。
<br>mmdetection官网依赖：
<br>https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md
<br>如下：
```
  Linux
  Python 3.5+ 
  PyTorch 1.0+ or PyTorch-nightly
  CUDA 9.0+
  NCCL 2+
  GCC 4.9+
  mmcv
```
<br>我的GPU：
```
  GTX1050 对应的cuda9.0
```
<br>我的配置：
```
  ubuntu 18.04
  python 3.7
  pytorch 1.1.0 torchvision 0.3.0 cudatoolkit 9.0
  cuda 9.0 cudnn7.                                #（cuda 9.0对应的版本）
  NCCl                                            # 未安装（应该和多线程加速有关）
  gcc 5.5.0                                       # (将原来的7降版本为5，cuda要求6.0以下，torch要求4.9以上）
  mmcv                                            # （如下安装）
  ```
## 二、安装mmdetection
（安装cuda、cudnn 参考https://blog.csdn.net/xierhacker/article/details/53035989）
<br>（安装torch 参考https://blog.csdn.net/jianjuly/article/details/93916871）
### 1、下载conda 官网下载.sh文件
### 2、降级gcc：
```
  sudo apt install gcc-5
  sudo apt install g++-5
  sudo mv gcc gcc.bak      # 备份
  sudo ln -s gcc-5 gcc     # 重新链接gcc
  sudo mv g++ g++.bak      # 备份
  sudo ln -s g++-5 g++　   # 重新链接g++
```
### 3、下载cuda9.0以及4个补丁，官网下载（18.04的可以使用17.04的）
  配置环境：
```
    sudo vim ~/.bashrc
    export PATH=/usr/local/cuda-9.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64$LD_LIBRARY_PATH
```
<br>  重启测试：
```
    cd /usr/local/cuda-8.0/samples/1_Utilities/deviceQuery
    sudo make
    ./deviceQuery
```
### 4、下载cudnn cuda9.0对应的版本
<br>参考：https://developer.nvidia.com/rdp/cudnn-download
```
  sudo cp cudnn.h /usr/local/cuda/include/
  sudo cp lib* /usr/local/cuda/lib64/          # 复制动态链接库
  cd /usr/local/cuda/lib64/
  sudo rm -rf libcudnn.so libcudnn.so.5        # 删除原有动态文件
  sudo ln -s libcudnn.so.5.1.5 libcudnn.so.5   # 生成软衔接
  sudo ln -s libcudnn.so.5 libcudnn.so         # 生成软链接
```
### 5、配置conda镜像
vi .condarc
<br> 可能出现清华等镜像不可用的问题
```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123
  - defaults
show_channel_urls: true
```
<br> or
```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud//pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
show_channel_urls: true
```
### 6、创建环境
```
  conda create -n open-mmlab python=3.7 
  conda activate open-mmlab
```
### 7、下载cython
```
  conda install cython
```
### 8、下载对应的torch、mmcv
```
  conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0
  git clone https://github.com/open-mmlab/mmcv.git
  cd mmcv
  pip install .
  # （或 pip install mmcv）
  # （中间有些opencv、matplotlib下载不下来可以下载whl）
  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
  python setup.py develop
```
## 三、测试或参照github get_start的tools/test
test.py
```python
#coding=utf-8
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result
# 模型配置文件
config_file = '../../configs/faster_rcnn_r50_fpn_1x.py'
# 预训练模型文件
checkpoint_file = '../../checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:1')
# 测试单张图片并进行展示
img = 'test1.jpg'
result = inference_detector(model, img)
show_result(img, result, model.CLASSES)
# 测试一个图像列表并保存结果图像
imgs = ['test1.jpg', 'test2.jpg', 'test3.jpg']
for i, result in enumerate(inference_detector(model, imgs)):
    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))
```
### 1、在mmcv/mmdetection目录下新建my_code/test和checkpoints
### 2、下载checkpoints，放到checkpoints文件夹
官网：https://github.com/open-mmlab/mmdetection/blob/master/MODEL_ZOO.md
<br>我的下载：https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
### 3、图片三张test123.jpg放到test文件夹
## 四、数据准备
```
It is recommended to symlink the dataset root to $MMDETECTION/data.
mkdir data
ln -s $COCO_ROOT data
```
