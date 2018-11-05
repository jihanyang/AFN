#!/bin/bash

post='14'
repeat='3'

data_root='/data/da/data/Visda2017'
snapshot='/data/da/vanilla/Visda2017/IAFN_ES/snapshot'
result='/home/xuruijia/yjh/domain_adaptation/vanilla/Visda2017/IAFN_ES/result'
epoch=50
model='resnet101'
gpu_id='5'

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
