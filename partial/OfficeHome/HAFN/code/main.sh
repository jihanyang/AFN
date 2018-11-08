#!/bin/bash

post='1'
repeat='1'

task=('Ar->Cl' 'Ar->Pr' 'Ar->Rw' 'Cl->Ar' 'Cl->Pr' 'Cl->Rw' 'Pr->Ar' 'Pr->Cl' 'Pr->Rw' 'Rw->Ar' 'Rw->Cl' 'Rw->Pr')
source=('Art' 'Art' 'Art' 'Clipart' 'Clipart' 'Clipart' 'Product' 'Product' 'Product' 'Real_World' 'Real_World' 'Real_World')
target=('Clipart' 'Product' 'Real_World' 'Art' 'Product' 'Real_World' 'Art' 'Clipart' 'Real_World' 'Art' 'Clipart' 'Product')

data_root='/data/da/data/OfficeHome/OfficeHomeDataset_10072016'
snapshot='/data/da/partial/OfficeHome/HAFN/snapshot'
result='/home/xuruijia/yjh/domain_adaptation/partial/OfficeHome/HAFN/result'
epoch=120
gpu_id='2'

for((index=0; index < 12; index++))
do
    echo ">> traning task ${index} : ${task[index]}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
        --data_root ${data_root} \
        --snapshot ${snapshot} \
        --post ${post} \
        --repeat ${repeat} \
        --epoch ${epoch} \
        --task ${task[index]} \
        --source ${source[index]} \
        --target ${target[index]}
    
    echo ">> testing task ${index} : ${task[index]}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python eval.py \
        --post ${post} \
        --data_root ${data_root} \
        --snapshot ${snapshot} \
        --result ${result} \
        --epoch ${epoch} \
        --task ${task[index]} \
        --target ${target[index]} \
        --repeat ${repeat}
done





