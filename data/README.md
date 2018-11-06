# Datesets

We use ImageCLEF-DA, VisDA2017, Office-31 and Office-Home in our experiments.

## ImageCLEF-DA

We use ImageCLEF-DA published [here](https://github.com/thuml/Xlearn/tree/master/caffe), and replace the absolute path in `*List.txt` to relative path.

Please reorganize the images and label lists as following directory tree.

```latex
.
|-- b
|-- bList.txt
|-- c
|-- cList.txt
|-- i
|-- iList.txt
|-- p
`-- pList.txt
```

## Office-31

You can download the origin image [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/). Our label files are replaced the absolute path  [here](https://github.com/thuml/Xlearn/tree/master/pytorch/data/office) to relative path.

Please reorganize the images and label lists as following directory tree.

```latex
.
|-- amazon
|   |-- images
|   `-- label.txt
|-- dslr
|   |-- images
|   `-- label.txt
`-- webcam
    |-- images
    `-- label.txt
```

## Office-Home

You can download the origin image [here](http://hemanthdv.org/OfficeHome-Dataset/). Our label files are replaced the absolute path  [here](https://github.com/thuml/PADA/tree/master/pytorch/data/office-home) to relative path.

Please reorganize the images and label lists as following directory tree.

```latex
.
|-- Art
|-- Art_shared.txt
|-- Art.txt
|-- Clipart
|-- Clipart_shared.txt
|-- Clipart.txt
|-- Product
|-- Product_shared.txt
|-- Product.txt
|-- Real_World
|-- Real_World_shared.txt
`-- Real_World.txt
```

## VisDA2017

You can download this dataset [here](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification). We also announce the `validation6_list.txt` under the partial setting and label lists used in *sample size* experiment.

```latex
.
|-- train
|-- train_list.txt
|-- validation
|-- validation6_list.txt
|-- validation_list25.txt
|-- validation_list50.txt
|-- validation_list75.txt
`-- validation_list.txt
```

