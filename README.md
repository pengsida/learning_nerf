# Learning NeRF

This repository is initially created by [Haotong Lin](https://haotongl.github.io/).

## Motivation of this repository

1. 面向实验室本科生的科研训练。通过复现NeRF来学习NeRF的算法、PyTorch编程。
2. 这个框架是实验室常用的代码框架，实验室的一些算法在这个框架下实现，比如：[ENeRF](https://github.com/zju3dv/enerf), [NeuSC](https://github.com/zju3dv/NeuSC), [MLP Maps](https://github.com/zju3dv/mlp_maps), [Animatable NeRF](https://github.com/zju3dv/animatable_nerf), [Neural Body](https://github.com/zju3dv/neuralbody)。实验室通过大量实践证明了这个代码框架的灵活性。学会使用这个框架，方便后续参与实验室的Project，也方便自己实现新的算法。

## Prerequisites

1. 确保你已经熟悉使用python, 尤其是debug工具：ipdb。

2. 计算机科学非常讲究自学能力和自我解决问题的能力，如果有一些内容没有介绍的十分详细，请先自己尝试探索代码框架。如果遇到代码问题，请先搜索网上的资料，或者查看仓库的Issues里有没有相似的已归档的问题。

3. 如果还是有问题，可以在这个仓库的issue里提问。

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


## 从Image fitting demo来学习这个框架


### 任务定义

训练一个MLP，将某一张图像的像素坐标作为输入, 输出这一张图像在该像素坐标的RGB value。

### Training

```
python train_net.py --cfg_file configs/img_fit/lego_view0.yaml
```

### Evaluation

```
python run.py --type evaluate --cfg_file configs/img_fit/lego_view0.yaml
```

### 查看loss曲线

```
tensorboard --logdir=data/record --bind_all
```


## 开始复现NeRF

### 配置文件

我们已经在configs/nerf/ 创建好了一个配置文件，nerf.yaml。其中包含了复现NeRF必要的参数。
你可以根据自己的喜好调整对应的参数的名称和风格。


### 创建dataset： lib.datasets.nerf.synthetic.py

核心函数包括：init, getitem, len.

init函数负责从磁盘中load指定格式的文件，计算并存储为特定形式。

getitem函数负责在运行时提供给网络一次训练需要的输入，以及groundtruth的输出。
例如对NeRF，分别是1024条rays以及1024个RGB值。

len函数是训练或者测试的数量。getitem函数获得的index值通常是[0, len-1]。


#### debug：

```
python run.py --type dataset --cfg_file configs/img_fit/lego_view0.yaml
```

### 创建network:

核心函数包括：init, forward.

init函数负责定义网络所必需的模块，forward函数负责接收dataset的输出，利用定义好的模块，计算输出。例如，对于NeRF来说，我们需要在init中定义两个mlp以及encoding方式，在forward函数中，使用rays完成计算。


#### debug：

```
python run.py --type network --cfg_file configs/img_fit/lego_view0.yaml
```

### loss模块和evaluator模块

这两个模块较为简单，不作仔细描述。

debug方式分别为：

```
python train_net.py --cfg_file configs/img_fit/lego_view0.yaml
```

```
python run.py --type evaluate --cfg_file configs/img_fit/lego_view0.yaml
```
