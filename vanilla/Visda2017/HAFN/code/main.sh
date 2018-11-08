#!/bin/bash

post='1'
repeat='1'

data_root='/data/da/data/Visda2017'
snapshot='/data/da/vanilla/Visda2017/HAFN/snapshot'
result='/home/xuruijia/yjh/domain_adaptation/vanilla/Visda2017/HAFN/result'
epoch=10
model='resnet101'
gpu_id='1'

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

