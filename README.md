# Spectraformer

<p align="center">
<a href="#Features">Features</a> • <a href="#install">Installation</a> • <a href="#usage">Usage</a> • <a href="#benchmark">Algorithms</a> • <a href="#License">License</a>
<br>
</p>

*Spectraformer* a unified random feature framework for transformer for approximating and learning the kernel function in linearized attention of the Transformer. It allows for the combination of any weight matrix with any component function. This repository is the official implementation of Spectraformer

![spectraformer framework](./resources/framework.png)

## Features
Spectraformer evaluates different combinations of weight matrices and component functions in the Transformer in three textual tasks in the LRA benchmark.

The component functions we currently cover are checked by green ticks
![spectraformer component functions](./resources/component_functions.png)

The weight matrices we currently cover are checked by green ticks
![spectraformer weight matrices](./resources/weight_matrices.png)

## Installation

### Preparing the Code
To install requirements in a conda environment:
<!-- https://medium.com/@crismunozv/installing-custom-python-version-in-vertex-ai-eb9b1463e023 -->
<!-- Can also use python=3.12 -->
```
conda create -y -n spectraformer python=3.12
conda activate spectraformer
conda install torchquad -c conda-forge
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

<!-- If cannot install transformers -->
<!-- https://github.com/huggingface/transformers/issues/2831 -->
<!-- curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
Then reinstall transformers -->

Note: Specific requirements for data preprocessing are not included here.

### Preparing the Dataset

Processed files can be downloaded [here](https://drive.google.com/drive/folders/1rE0SjpeFKPFtgmWWjYCoIMz91UozHWWC?usp=sharing), or processed with the following steps:

1. Requirements
```
tensorboard>=2.3.0
tensorflow>=2.3.1
tensorflow-datasets>=4.0.1
```
2. Download [the TFDS files for pathfinder](https://storage.cloud.google.com/long-range-arena/pathfinder_tfds.gz) and then set _PATHFINER_TFDS_PATH to the unzipped directory (following https://github.com/google-research/long-range-arena/issues/11)
3. Download [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) (7.7 GB).
4. Unzip `lra-release` and put under `./data/`.
```
cd data
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar zxvf lra-release.gz 
```
5. Create a directory `lra_processed` under `./data/`.
```
mkdir lra_processed
cd ..
```
6.The directory structure would be (assuming the root dir is `code`)
```
./data/lra-processed
./data/long-range-arena-main
./data/lra_release
```
7. Create train, dev, and test dataset pickle files for each task.
```
cd preprocess
python create_pathfinder.py
python create_listops.py
python create_retrieval.py
python create_text.py
python create_cifar10.py
```

Note: most source code comes from [LRA repo](https://github.com/google-research/long-range-arena).

## Usage

Modify the configuration in `config.py` and run
```
python main.py --mode train --attn skyformer --task lra-text
```
- mode: `train`, `eval`
- attn: `softmax`, `nystrom`, `linformer`, `reformer`, `perfromer`, `informer`, `bigbird`,  `kernelized`, `skyformer`
- feat: `trigrf`, `posrf`, `oprf`, `gerf`, `saderf`
- kernel_type: `gaus`, `orf`, `scrf`, `sorf`, `rom`, `sgq`, `qmc`, `mm`, `fastfood_fixed`, `fastfood_learnable`
- task: `lra-listops`, `lra-pathfinder`, `lra-retrieval`, `lra-text`, `lra-image`

To run experiments on GCP
```
pip install --upgrade google-cloud-storage

python main.py --mode eval --attn skyformer --task lra-text --bucket_name kernelized-transformer-code --blob_path kernelized-transformer/data/lra_processed

python main.py --mode eval --random 792 --attn performer --feat orf --kernel_type geomrf --task glue-cola
```

## Algorithms

|                             |                   | Accuracy (%) |           |           |           | Time (hour) |          |          |          | Memory (GB) |          |          |          |
|-----------------------------|:-----------------:|:------------:|:---------:|:---------:|:---------:|:-----------:|:--------:|:--------:|:--------:|:-----------:|:--------:|:--------:|:--------:|
| Paper                       |       Model       |       L      |     T     |     R     |     Mu    |      L      |     T    |     R    |    Mu    |      L      |     T    |     R    |    Mu    |
| _Choromanski et al. 2021_   |    _PosRF-ORF_    |    _37.35_   |  _61.60_  |  _80.53_  |  _59.83_  |    _1.30_   |  _2.84_  |  _2.89_  |  _2.34_  |    _1.17_   |  _2.31_  |  _2.10_  |  _1.86_  |
|                             |     PosRF-SORF    |     22.98    |   63.31   |   65.52   |   50.60   |     1.29    |   2.81   |   2.83   |   2.31   |     1.17    |   2.31   |   2.10   |   1.86   |
|                             |     PosRF-QMC     |     37.50    |   60.41   |   80.71   |   59.54   |     1.30    |   2.84   |   2.89   |   2.34   |     1.17    |   2.31   |   2.10   |   1.86   |
|                             |     PosRF-SGQ     |     37.45    |   62.68   |   78.37   |   59.50   |     1.30    |   2.83   |   2.89   |   2.34   |     1.17    |   2.31   |   2.10   |   1.86   |
| **This paper**              |    **PosRF-MM**   |   **38.05**  | **61.85** | **80.67** | **60.19** |   **1.31**  | **2.84** | **2.89** | **2.35** |   **1.17**  | **2.31** | **2.10** | **1.86** |
| _Chowdhury et al. 2022_     | _PosRF-FastFoodL_ |    _24.95_   |  _64.76_  |  _76.37_  |  _55.36_  |    _2.69_   |  _5.59_  |  _5.61_  |  _4.63_  |    _0.78_   |  _1.56_  |  _1.53_  |  _1.29_  |
| _Likhosherstov et al. 2022_ |     _OPRF-ORF_    |    _37.50_   |  _59.35_  |  _80.90_  |  _59.25_  |    _1.61_   |  _3.45_  |  _3.50_  |  _2.86_  |    _1.36_   |  _2.71_  |  _2.56_  |  _2.21_  |
|                             |     OPRF-SORF     |     32.11    |   64.34   |   77.47   |   57.97   |     1.57    |   3.37   |   3.41   |   2.78   |     1.36    |   2.71   |   2.56   |   2.21   |
|                             |      OPRF-QMC     |     38.41    |   60.32   |   80.80   |   59.84   |     1.61    |   3.46   |   3.51   |   2.86   |     1.36    |   2.71   |   2.56   |   2.21   |
|                             |      OPRF-MM      |     38.71    |   60.39   |   80.45   |   59.85   |     1.61    |   3.46   |   3.51   |   2.86   |     1.36    |   2.71   |   2.56   |   2.21   |
|                             |      OPRF-SGQ     |     22.53    |   61.34   |   79.29   |   54.39   |     1.60    |   3.45   |   3.47   |   2.84   |     1.36    |   2.71   |   2.56   |   2.21   |
|                             |   OPRF-FastFoodL  |     37.40    |   64.04   |   78.32   |   59.92   |     2.81    |   5.84   |   5.86   |   4.84   |     0.85    |   1.69   |   1.67   |   1.40   |
| _Likhosherstov et al. 2023_ |    _SADERF-ORF_   |    _37.35_   |  _61.37_  |  _80.87_  |  _59.86_  |    _1.61_   |  _3.52_  |  _3.58_  |  _2.90_  |    _1.44_   |  _2.86_  |  _2.69_  |  _2.33_  |
|                             |    SADERF-SORF    |     32.31    |   64.49   |   76.43   |   57.74   |     1.58    |   3.44   |   3.49   |   2.84   |     1.44    |   2.86   |   2.69   |   2.33   |
|                             |     SADERF-QMC    |     37.70    |   59.89   |   80.65   |   59.41   |     1.61    |   3.52   |   3.58   |   2.90   |     1.44    |   2.86   |   2.69   |   2.33   |
|                             |     SADERF-MM     |     38.00    |   60.80   |   80.48   |   59.76   |     1.61    |   3.52   |   3.58   |   2.90   |     1.44    |   2.86   |   2.69   |   2.33   |
|                             |     SADERF-SGQ    |     36.79    |   63.55   |   77.31   |   59.22   |     1.61    |   3.52   |   3.57   |   2.90   |     1.44    |   2.86   |   2.69   |   2.33   |
|                             |  SADERF-FastFoodL |     28.63    |   64.64   |   77.61   |   56.96   |     2.81    |   5.89   |   5.92   |   4.87   |     0.92    |   1.83   |   1.79   |   1.51   |


**References**

Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. Rethinking attention with performers. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=Ua6zuk0WRH.

Sankalan Pal Chowdhury, Adamos Solomou, Kumar Avinava Dubey, and Mrinmaya Sachan. Learning the transformer kernel. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=tLIBAEYjcv.

Valerii Likhosherstov, Krzysztof M Choromanski, Kumar Avinava Dubey, Frederick Liu, Tamas Sarlos, and Adrian Weller. Chefs'random tables: Non-trigonometric random features. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 34559–34573. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/df2d62b96a4003203450cf89cd338bb7-Paper-Conference.pdf.

Valerii Likhosherstov, Krzysztof Marcin Choromanski, Kumar Avinava Dubey, Frederick Liu, Tamas Sarlos, and Adrian Weller. Dense-exponential random features: Sharp positive estimators of the gaussian kernel. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=S0xrBMFihS.


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>