# Pytorch Implementation of PointNet and PointNet++ 

This repo is from [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

data from [modelnet40](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset)


## What we need to do
* DataLoader (Done)
* Model change (ing)
* Set Pipeline each experiment (plan)


```
├── data_utils
│   ├──ModelNetDataLoader.py
├── data
|   | modelnet40_normal_resampled
|      ├── airplane
|          ├── test
|              ├── airplane_0627.off
|              ├── airplane_0628.off
|              ├── ...
|          ├── train
|              ├── airplane_0001.off
|              ├── airplane_0002.off
|              ├── ...
|      ├── bathtub
|      ├── ...
├── models
|   ├── pointnet2_cls_msg.py
|   ├── pointnet2_cls_ssg.py
|   ├── pointnet2_utils.py
|   ├── ...
├── test_classification.py
├── train_classification.py
```
