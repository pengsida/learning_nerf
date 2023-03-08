# LearningNeRF
1. 确保你已经熟悉使用python, 尤其是debug工具： ipdb.

2. 计算机科学是一门实验科学，如果有一些内容没有介绍的十分详细，请先自己尝试通过实验的方式理解。

3. 如果还是有问题，可以来这个页面查找有没有相似的已归档的问题，
[常见问题](https://haotong.notion.site/Learning-NeRF-a6f53ffa496e4a04a4e1965169ffba17)

4. 如果没有相似的问题，请在待回答界面创建问题，并且钉钉提醒学长/学姐来回答。
[待回答](https://haotong.notion.site/6bf1d84551734419aa1a1038abc41b18)

 

## Data preparation

Download NeRF synthetic dataset and add a link to the data directory. After preparation, you should have the following directory structure: 
```
data/nerf_synthetic
|-- chair
|   |-- test
|   |-- train
|   |-- val
|-- drums
|   |-- test
......
```


# 从Image fitting demo来学习这个框架


## 任务定义

利用一个MLP，以某一张图像的像素坐标作为输入, 输出这一张图像在该像素坐标的位置。


## Training

```
python train_net.py --cfg_file configs/img_fit/lego_view0.yaml
```

## Evaluation

```
python run.py --type evaluate --cfg_file configs/img_fit/lego_view0.yaml
```

## 查看loss曲线

```
tensorboard --logdir=data/record --bind_all
```


# 开始复现NeRF

## 配置文件

我们已经在configs/nerf/ 创建好了一个配置文件，nerf.yaml。其中包含了复现NeRF必要的参数。
你可以根据自己的喜好调整对应的参数的名称和风格。


## 创建dataset： lib.datasets.nerf.synthetic.py

核心函数包括：init, getitem, len.

init函数负责从磁盘中load指定格式的文件，计算并存储为特定形式。

getitem函数负责在运行时提供给网络一次训练需要的输入，以及groundtruth的输出。
例如对NeRF，分别是1024条rays以及1024个RGB值。

len函数是训练或者测试的数量。getitem函数获得的index值通常是[0, len-1]。


### debug：

```
python run.py --type dataset --cfg_file configs/img_fit/lego_view0.yaml
```

## 创建network:

核心函数包括：init, forward.

init函数负责定义网络所必需的模块，forward函数负责接收dataset的输出，利用定义好的模块，计算输出。例如，对于NeRF来说，我们需要在init中定义两个mlp以及encoding方式，在forward函数中，使用rays完成计算。


### debug：

```
python run.py --type network --cfg_file configs/img_fit/lego_view0.yaml
```

## loss模块和evaluator模块

这两个模块较为简单，不作仔细描述。


debug方式分别为：

```
python train_net.py --cfg_file configs/img_fit/lego_view0.yaml
```


```
python run.py --type evaluate --cfg_file configs/img_fit/lego_view0.yaml
```
