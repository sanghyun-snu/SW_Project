#!/bin/bash

gpu=1
arch=vgg16_gap
name=test1
dataset="PascalVOC"
data_root="/srv/PascalVOC/VOCdevkit/VOC2012/"
epoch=100
decay=40
batch=32
wd=1e-4
lr=0.001

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --arch ${arch} \
    --name ${name} \
    --pretrained True \
    --data ${data_root} \
    --dataset ${dataset} \
    --epochs ${epoch} \
    --LR-decay ${decay} \
    --batch-size ${batch} \
    --lr ${lr} \
    --wd ${wd} \
    --nest True