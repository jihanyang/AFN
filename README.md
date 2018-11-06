# AFN

PyTorch implementation for *Unsupervised Domain Adaptation: An Adaptive Feature Norm Approach*.

## Requirements

- Platform : Linux

- Hardware : Nvidia GPU
- Others: 
  - CUDA 9.0.176
  - PyTorch 0.4.1
  - tqdm

## Datasets

Please follow the *README.md* to organize the datasets

## Trainning and Evaluation

- Make sure you have organized datasets and satisfied the requirements.

- According to the hierarchy in following block, enter corresponding setting ,datasets and method folder.
- Modify parameters: `data_root`,  `result` and `snapshot`  in  `main.sh` (You don't need to change any other parameters for reproduction)
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
│   │   └── IAFN
│   ├── README.md
│   └── Visda2017
│       ├── HAFN
│       └── IAFN
├── resources
└── vanilla
    ├── ImageCLEF
    │   ├── HAFN
    │   └── IAFN
    ├── Office31
    │   ├── HAFN
    │   └── IAFN
    ├── README.md
    └── Visda2017
        ├── HAFN
        └── IAFN
```

Here are some description for parameters :

- `data_root` : the directory of data.
- `snapshot` : the directory to store and load state dicts.
- `result` : the directory that store evaluating results.
- `post` : distinguish each experiment.
- `repeat` : distinguish each repeated result in a experiment.
- `gpu_id` : the GPU ID to run experiments.

