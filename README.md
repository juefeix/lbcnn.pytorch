# [work in progress] Pytorch Implementation of LBCNN

Pytorch Implementation of LBCNN [https://arxiv.org/abs/1608.06049] on various vision tasks.


## Cifar10

The cifar10 codes highly reuse https://github.com/kuangliu/pytorch-cifar, I manually modified conv layer in certain model to use LBCNN.

```sh
git checkout cifar10

sh run.sh 0
```


## ImageNet

The ImageNet codes highly reuse pytorch example. Run the following to train a resnet101 with LBCNN.

```sh
git checkout imagenet

sh run.sh 0,1,2,3 resnet101 512
```


## Detection

Checkout to seg branch

```sh
git checkout seg
```

Install mmcv-full and mmdet.

```sh
cd mmcv
MMCV_WITH_OPS=1,FORCE_CUDA=1 python setup.py develop

# The mmdetection is copied from https://github.com/open-mmlab/mmdetection
# And I modified the configuration file `faster_rcnn_r50_fpn_1x_coco.py`
#   1. no pretrain model
#   2. batch per gpu = 4 instead of 2
#   3. Use schedule_20e since no pretrain
cd ../mmdetection
python setup.py develop
```
Then follow mmdet's documentation to run normal detection training, 



## Segmentation

Checkout to seg branch

```sh
git checkout seg
```

Install mmcv-full and mmsegmentation.

```sh
# Skip if already installed
cd mmcv
MMCV_WITH_OPS=1,FORCE_CUDA=1 python setup.py develop

# The mmsegmentation is copied from https://github.com/open-mmlab/mmsegmentation
# And I modified the configuration file `fcn_r50-d8_512x1024_80k_cityscapes.py`
#   1. no pretrain model
#   2. batch per gpu = 8 instead of 2
#   3. syncbn -> bn
cd ../mmsegmentation
python setup.py develop
```
Then follow mmseg's documentation to run normal segmentation training or run following script to run lbcnn version

```sh
cd mmsegmentation
PYTHONPATH=${LBCNN_GIT_PATH} CUDA_VISIBLE_DEVICES=6,7 USE_LBCNN=1 ./tools/dist_train.sh configs/fcn/fcn_r50-d8_512x1024_80k_cityscapes.lbcnn.py 2
```




### References

* Felix Juefei-Xu, Vishnu Naresh Boddeti, and Marios Savvides, [**Local Binary Convolutional Neural Networks**](https://arxiv.org/abs/1608.06049),
* *IEEE Computer Vision and Pattern Recognition (CVPR), 2017*. (Spotlight Oral Presentation)

* @inproceedings{juefei-xu2017lbcnn,<br>
&nbsp;&nbsp;&nbsp;title={{Local Binary Convolutional Neural Networks}},<br>
&nbsp;&nbsp;&nbsp;author={Felix Juefei-Xu and Vishnu Naresh Boddeti and Marios Savvides},<br>
&nbsp;&nbsp;&nbsp;booktitle={IEEE Computer Vision and Pattern Recognition (CVPR)},<br>
&nbsp;&nbsp;&nbsp;month={July},<br>
&nbsp;&nbsp;&nbsp;year={2017}<br>
}

and  

* Felix Juefei-Xu, Changqing Zhou, Vishnu Naresh Boddeti, and Marios Savvides, **Local Binary Convolutional Neural Networks and Beyond**,
* Work in Progress, 2021.


