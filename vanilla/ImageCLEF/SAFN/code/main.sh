#!/bin/bash

post='1'
repeat='1'

task=('I->P' 'P->I' 'I->C' 'C->I' 'C->P' 'P->C')
source=('i' 'p' 'i' 'c' 'c' 'p')
target=('p' 'i' 'c' 'i' 'p' 'c')

data_root='/data/da/data/ImageCLEF'
snapshot='/data/da/vanilla/ImageCLEF/IAFN/snapshot'
result='/home/xuruijia/yjh/domain_adaptation/vanilla/ImageCLEF/IAFN/result'
epoch=100
gpu_id='6'

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
