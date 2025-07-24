<div align="center">
    <h2>
        DSP-EntCD: A Knowledge-Freezing, Entropy-Guided Remote Sensing Change Detection Network with Domain-Specific Pretraining
    </h2>
</div>
<br>


## Introduction

The repository for this project is the code implementation of the paper DSP-EntCD: A Knowledge-Freezing, Entropy-Guided Remote Sensing Change Detection Network with Domain-Specific Pretraining.

If you find this project helpful, please give us a star ⭐️.

## Installation

### Environment Setting

<details open>

Run the following command to install dependencies.

If you only use the model code, this step is not needed.

```shell
pip install -r requirements.txt
```

## Dataset Preparation

<details open>

### Remote Sensing Change Detection Dataset

#### WHU-CD Dataset

- Dataset Download: [WHU-CD Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html)。

#### LEIVR-CD Dataset

- Dataset Download: [LEVIR-CD Dataset](https://chenhao.in/LEVIR/)。

#### Organization

You need to organize the data set into the following format:

```
${DATASET_ROOT} # dataset root dir
├── train
    ├── t1
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── t2
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        └── 0001.tif
        └── 0002.tif
        └── ...
├── val
    ├── t1
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── t2
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        ├── 0001.tif
        └── 0002.tif
        └── ...
├── test
    ├── t1
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── t2
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        └── 0001.tif
        └── 0002.tif
        └── ...
```

## Model Training and Inference

All hyperparameters for model training and inference are located in the `utils/path_hyperparameter.py` file, with corresponding comments for explanation.

## License

This project is licensed under the [Apache 2.0 License](LICENSE)。
