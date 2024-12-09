# Pytorch Implementation of PointNet and PointNet++ 

This repo is mainly from [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

data from [modelnet40](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset)


## What we need to do
* DataLoader (Done)
* Model change (Done)
* Set Pipeline each experiment (Done)


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

```shell
# ModelNet40
## Select different models in ./models 

## e.g., pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg

## e.g., pointnet2_ssg with normal features
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal

## e.g., pointnet2_ssg with uniform sampling
python train_classification.py --model pointnet2_cls_ssg --use_uniform_sample --log_dir pointnet2_cls_ssg_fps
python test_classification.py --use_uniform_sample --log_dir pointnet2_cls_ssg_fps
```
