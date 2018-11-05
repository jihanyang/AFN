#!/bin/bash

# usage:
#   ./main.sh [post(any content to record the conducted experiment)]
post='1'
repeat='3'

data_root='/data/da/data/Visda2017'
snapshot='/data/da/partial/Visda2017/SIAFN_ES/snapshot'
result='/home/xuruijia/yjh/domain_adaptation/partial/Visda2017/SIAFN_ES/result'
epoch=120
gpu_id='5'

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
   --data_root ${data_root} \
   --snapshot ${snapshot} \
   --post ${post} \
   --epoch ${epoch} \
   --repeat ${repeat}


CUDA_VISIBLE_DEVICES=${gpu_id} python eval.py \
    --post ${post} \
    --data_root ${data_root} \
    --snapshot ${snapshot} \
    --result ${result} \
    --epoch ${epoch} \
    --repeat ${repeat}


