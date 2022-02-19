# SUES-200: A Multi-height Multi-scene Cross-view Image Matching Datasets Between UAV and Satellite
 **Early Access**
 
### 用双分支卷积网络训练和测试：

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
