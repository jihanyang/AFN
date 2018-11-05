#!/bin/bash

post='2'
repeat='3'

data_root='/data/da/data/Visda2017'
snapshot='/data/da/partial/Visda2017/HAFN/snapshot'
result='/home/xuruijia/yjh/domain_adaptation/partial/Visda2017/HAFN/result'
epoch=30
model='resnet50'
gpu_id='4'

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
   --data_root ${data_root} \
   --snapshot ${snapshot} \
   --post ${post} \
   --repeat ${repeat} \
   --model ${model} \
   --epoch ${epoch}

CUDA_VISIBLE_DEVICES=${gpu_id} python eval.py \
    --post ${post} \
    --data_root ${data_root} \
    --snapshot ${snapshot} \
    --result ${result} \
    --epoch ${epoch} \
    --model ${model} \
    --repeat ${repeat}

