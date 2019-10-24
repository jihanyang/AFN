# AFN

PyTorch implementation for *Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation*.

## Requirements

- Platform : Linux

- Hardware : Nvidia GPU
- Others: 
  - CUDA 9.0.176
  - PyTorch 0.4.1
  - tqdm

## Datasets

Please follow the *README.md* in subfolder `Data` to organize datasets

## Training and Evaluation

- Make sure you have organized datasets and satisfied the requirements.

- According to the hierarchy in following block, enter corresponding setting ,dataset and method folder.
- Modify parameters:  `data_root`,  `result` and `snapshot`  in  `main.sh`，and can switch model through changing `model`. 
- If you want to run the IAFN+ENT mothod on *Office-31* or *ImageCLEF-DA*, you have to modify the command `CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \` to `CUDA_VISIBLE_DEVICES=${gpu_id} python train_ent.py \`
- run `bash main.sh` in your terminal

```latex
.
├── README.md
├── data
│   ├── ImageCLEF
│   ├── Office31
│   ├── OfficeHome
│   ├── README.md
│   └── Visda2017
├── partial
│   ├── OfficeHome
│   │   ├── HAFN
│   │   └── SAFN
│   ├── README.md
│   └── Visda2017
│       ├── HAFN
│       └── SAFN
├── resources
└── vanilla
    ├── ImageCLEF
    │   ├── HAFN
    │   └── SAFN
    ├── Office31
    │   ├── HAFN
    │   └── SAFN
    ├── README.md
    └── Visda2017
        ├── HAFN
        └── SAFN
```

Here are some description of parameters :

- `data_root` : the directory of data.
- `snapshot` : the directory to store and load state dicts.
- `result` : the directory that store evaluating results.
- `post` : distinguish each experiment.
- `repeat` : distinguish each repeated result in a experiment.
- `gpu_id` : the GPU ID to run experiments.
- `model` : switch model between `resnet101` and `resnet50`



## Citation

If you use AFN in your research, please consider citing:

```Latex
@InProceedings{Xu_2019_ICCV,
author = {Xu, Ruijia and Li, Guanbin and Yang, Jihan and Lin, Liang},
title = {Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```