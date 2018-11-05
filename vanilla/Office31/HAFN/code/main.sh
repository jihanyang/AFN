#!/bin/bash

task=('A->W' 'D->W' 'W->D' 'A->D' 'D->A' 'W->A')
source=('amazon' 'dslr' 'webcam' 'amazon' 'dslr' 'webcam')
target=('webcam' 'webcam' 'dslr' 'dslr' 'amazon' 'amazon')

post='1'
repeat='3'
data_root='/data/da/data/Office31'
snapshot='/data/da/vanilla/Office31/HAFN/snapshot'
result='/home/xuruijia/yjh/domain_adaptation/vanilla/Office31/HAFN/result'
epoch=60
gpu_id='5'

for((index=0; index < 6; index++))
do
    echo ">> traning task ${index} : ${task[index]}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
       --data_root ${data_root} \
       --snapshot ${snapshot} \
       --task ${task[index]} \
       --source ${source[index]} \
       --target ${target[index]} \
       --epoch ${epoch} \
       --post ${post} \
       --repeat ${repeat}
    
    echo ">> testing task ${index} : ${task[index]}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python eval.py \
        --data_root ${data_root} \
        --snapshot ${snapshot} \
        --result ${result} \
        --task ${task[index]} \
        --target ${target[index]} \
        --epoch ${epoch} \
       --post ${post} \
       --repeat ${repeat}
done
