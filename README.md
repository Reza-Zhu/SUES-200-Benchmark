# SUES-200: A Multi-height Multi-scene Cross-view Image Matching Benchmark Across UAV and Satellite

 **Early Access** 
 
 Paper Link : https://arxiv.org/abs/2204.10704

## Datasets

Download SUES-200 dataset. You may send me an email and claim not use this dataset for profit. SUES-200 is **ONLY  available to academic research**.

My email : m025120503@sues.edu.cn



## Quickly Start

### Installation

- Install Pytorch Torchvision https://pytorch.org/get-started/locally/
- install other libs

```
pip install timm pyyaml pytorch-metric-learning scipy pandas opencv-python grad-cam
```

### Config File

default: settings.yaml

```yaml
# dateset path
dataset_path: /media/data1/Datasets
weight_save_path: /media/data1/save_model_weight

# intial parameters
fp16 : 0  # apex
classes : 120 # 200*0.6=120
image_size: 384

# choose model
model : resnet

# super parameters
batch_size : 32
num_epochs : 80
drop_rate : 0.2
weight_decay : 0.0005
lr : 0.005

# test and evaluate


# if LPN
block : 4

# if SUES-200
height : 150

```

### Split Dataset

```bash
python script/split_dataset.py --path your_path --coff 0.6
mkdir your_path/Dataset
mv your_path/Training your_path/Dataset
mv your_path/Testing your_path/Dataset
```



### Train

```bash
python train --cfg settings.yaml
```



### Test & evaluate

```bash
python test_and_evaluate --cfg settings.yaml --name resnet_150_2022-04-25-10:26:34 --seq -3
```



## TO-DO List

- [ ] Improve README.md (ing...)
  - [ ] Evaluation methods
  - [ ] Visualization
  - [ ] Multiqueries
  - [ ] Draw heat map
  - [ ] ...

- [ ] Support University-1652 (ing....)
- [ ] Support CVUSA and CVACT
- [ ] ...

##  

## Citation

```
@misc{zhu2022sues200,
      title={SUES-200: A Multi-height Multi-scene Cross-view Image Benchmark Across Drone and Satellite}, 
      author={Runzhe Zhu},
      year={2022},
      eprint={2204.10704},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```





## Chinese Version

### 双分支卷积网络训练和测试：

1. 配置文件：settings.yaml

   该配置文件配置了 

   - 数据集地址 dataset_path
   - 权重文件保存地址 weight_save_path
   - 选取不同高度的数据 height
   - 训练时选用的特征提取模型 model
   - 训练时的学习率 lr
   - 训练轮数 num_epoch
   - 模型中的drop out  drop_rate
   - 训练时的批次大小 batch_size

2. 开始训练：执行 train.py 会根据上面配置好的参数进行训练，比较好的模型权重会保存在权重文件保存地址下的save_model_weight文件夹中（训练时会自动创建该文件夹）

3. 开始测试：执行 test_and_evaluate.py 会开始测试并输出测试结果，最后的结果会保存在save_model_weight中

4. 开始评估：执行 evaluation_methods.py 评估算法的适应性、稳定性、实时性。

基于网格搜素的自动调参数文件：AutoTuning.py

定义特征提取算法的文件：model_.py

CBAM_ResNet 算法模型定义：senet/cbam_resnet.py

### 数据集预处理，和其它一些算法在本数据集上的复现

数据集预处理文件夹：script

VLAD 复现代码：VLAD文件夹

NetVLAD 复现代码：NetVLAD文件夹，train_NetVLAD.py test_NetVlAD.py

### 其它文件

数据增强方式：autoaugment.py random_erasing.py
