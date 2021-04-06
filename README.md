# lbcnn.pytorch

Test (LBCNN)[https://arxiv.org/abs/1608.06049] on various tasks.


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



## Segmentation

