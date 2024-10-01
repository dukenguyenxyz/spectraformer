# Spectraformer

<p align="center">
<a href="#Features">Features</a> • <a href="#install">Installation</a> • <a href="#usage">Usage</a> • <a href="#benchmark">Algorithms</a> • <a href="#License">License</a>
<br>
</p>

*Spectraformer* a unified random feature framework for transformer for approximating and learning the kernel function in linearized attention of the Transformer. It allows for the combination of any weight matrix with any component function. This repository is the official implementation of Spectraformer

<!-- ![spectraformer framework](./resources/framework.png) -->
<img src="resources/framework.png" alt="spectraformer framework">

## Features
Spectraformer evaluates different combinations of weight matrices and component functions in the Transformer in three textual tasks in the LRA benchmark.

The component functions we currently cover are checked by green ticks
<!-- ![spectraformer component functions](./resources/component_functions.png) -->
<img src="resources/component_functions.png" alt="spectraformer component functions">

The weight matrices we currently cover are checked by green ticks
<!-- ![spectraformer weight matrices](./resources/weight_matrices.png) -->
<img src="resources/weight_matrices.png" alt="spectraformer weight matrices">

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
```

## Research
To incorporation a new component function or weight matrix, please satisfy the following requirement and follow the instruction.

### Component Function

**Requirement**: Component function `f` must satisfy $\mathbb{E}_\omega[f(x)f(y)] = K(x,y)$

**Code implementation**
- Add the new component function `f` to `src/models/component_functions.py`, the arguments should include data (the input), and other optional parameters.
- Import `f` and add a new entry to `FastAttention.comp_functions[f_name] = f` in `src/models/attention_performer.py` (line 176)

### Weight Matrix

**Requirement**: Weight matrix `W` must either be an approximator or a learner, as an approximator. As an approximator, it must provide unbiased or nearly unbiased estimation of a kernel k i.e., 

$$\mathbb{E}_\omega[f(x)f(y)] = K(x,y), W = [\omega_1, .. \omega_s]^T, f = TrigRF$$

A learner simply needs to parameterize a distribution `p`.

**Code implementation**
- Add the new weight matrix w to `src/models/weight_matrix_approx.py` or `src/models/weight_matrix_learner.py`, the arguments should include `nb_rows` (number of rows), `nb_cols` (number of columns) and `device`.
- Import `w` and add the if clause and `w` function call in `src/models/attention_performer.py` (line 208)

## Algorithms

|                  |              | Accuracy (\%) $\uparrow$ |              |       |      | Time (hour) $\downarrow$ |      |       |      | Memory (GB) $\downarrow$ |      |       |
|------------------|:------------:|:------------------------:|:------------:|:-----:|:----:|:------------------------:|:----:|:-----:|:----:|:------------------------:|:----:|:-----:|
|                  |       L      |             T            |       R      | $\mu$ |   L  |             T            |   R  | $\mu$ |   L  |             T            |   R  | $\mu$ |
| OPRF-FastFoodL   | 37.55 (0.48) |       64.41 (0.62)       | 77.70 (0.33) | 59.89 | 1.07 |           2.07           | 2.12 |  1.75 | 0.86 |           1.72           | 1.68 |  1.42 |
| OPRF-MM          | 38.08 (0.53) |       60.40 (0.85)       | 81.09 (0.18) | 59.86 | 0.68 |           1.26           | 1.24 |  1.06 | 1.36 |           2.71           | 2.56 |  2.21 |
| PosRF-MM         | 37.06 (0.37) |       61.87 (1.79)       | 80.58 (0.53) | 59.84 | 0.56 |           1.05           | 1.06 |  0.89 | 1.17 |           2.31           | 2.10 |  1.86 |
| OPRF-ORF         | 38.34 (0.22) |       60.16 (0.79)       | 80.88 (0.17) | 59.80 | 0.68 |           1.26           | 1.25 |  1.06 | 1.36 |           2.71           | 2.56 |  2.21 |
| SADERF-QMC       | 37.37 (0.38) |       61.14 (1.48)       | 80.84 (0.11) | 59.78 | 0.68 |           1.25           | 1.29 |  1.07 | 1.44 |           2.86           | 2.69 |  2.33 |
| PosRF-QMC        | 37.11 (0.09) |       61.69 (0.96)       | 80.55 (0.13) | 59.78 | 0.56 |           1.05           | 1.05 |  0.89 | 1.17 |           2.31           | 2.10 |  1.86 |
| SADERF-MM        | 37.10 (0.22) |       60.68 (1.88)       | 81.13 (0.17) | 59.64 | 0.68 |           1.25           | 1.29 |  1.07 | 1.44 |           2.86           | 2.69 |  2.33 |
| SADERF-ORF       | 37.10 (0.19) |       60.39 (2.08)       | 81.05 (0.22) | 59.51 | 0.68 |           1.25           | 1.28 |  1.07 | 1.44 |           2.86           | 2.69 |  2.33 |
| OPRF-QMC         | 37.69 (0.62) |       59.94 (0.59)       | 80.38 (0.49) | 59.34 | 0.68 |           1.26           | 1.26 |  1.07 | 1.36 |           2.71           | 2.56 |  2.21 |
| SADERF-SGQ       | 37.11 (0.21) |       62.46 (0.54)       | 78.38 (0.25) | 59.32 | 0.68 |           1.25           | 1.27 |  1.07 | 1.44 |           2.86           | 2.69 |  2.33 |
| SADERF-FastFoodL | 36.02 (1.38) |       64.63 (0.18)       | 76.99 (0.61) | 59.21 | 1.07 |           2.08           | 2.16 |  1.77 | 0.92 |           1.84           | 1.80 |  1.52 |
| OPRF-SGQ         | 37.10 (0.23) |       61.25 (0.54)       | 78.69 (0.54) | 59.01 | 0.67 |           1.26           | 1.25 |  1.06 | 1.36 |           2.71           | 2.56 |  2.21 |
| PosRF-ORF        | 34.35 (5.96) |       60.30 (0.97)       | 80.45 (0.22) | 58.37 | 0.56 |           1.05           | 1.06 |  0.89 | 1.17 |           2.31           | 2.10 |  1.86 |
| PosRF-FastFoodL  | 33.46 (3.70) |       64.65 (0.36)       | 76.95 (0.48) | 58.35 | 1.02 |           1.98           | 2.03 |  1.68 | 0.79 |           1.57           | 1.53 |  1.30 |
| SADERF-SORF      | 33.30 (0.98) |       64.70 (0.36)       | 74.71 (1.90) | 57.57 | 0.68 |           1.24           | 1.28 |  1.07 | 1.44 |           2.86           | 2.69 |  2.33 |
| PosRF-SGQ        | 28.64 (7.54) |       62.38 (0.53)       | 78.28 (0.20) | 56.43 | 0.56 |           1.05           | 1.05 |  0.89 | 1.17 |           2.31           | 2.10 |  1.86 |
| OPRF-SORF        | 27.91 (3.26) |       64.76 (0.66)       | 75.92 (1.74) | 56.20 | 0.67 |           1.26           | 1.24 |  1.06 | 1.36 |           2.71           | 2.56 |  2.21 |
| PosRF-SORF       | 21.27 (6.65) |       62.99 (0.40)       | 67.10 (1.11) | 50.45 | 0.56 |           1.05           | 1.05 |  0.89 | 1.17 |           2.31           | 2.10 |  1.86 |


**References**

Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. Rethinking attention with performers. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=Ua6zuk0WRH.

Sankalan Pal Chowdhury, Adamos Solomou, Kumar Avinava Dubey, and Mrinmaya Sachan. Learning the transformer kernel. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=tLIBAEYjcv.

Valerii Likhosherstov, Krzysztof M Choromanski, Kumar Avinava Dubey, Frederick Liu, Tamas Sarlos, and Adrian Weller. Chefs'random tables: Non-trigonometric random features. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 34559–34573. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/df2d62b96a4003203450cf89cd338bb7-Paper-Conference.pdf.

Valerii Likhosherstov, Krzysztof Marcin Choromanski, Kumar Avinava Dubey, Frederick Liu, Tamas Sarlos, and Adrian Weller. Dense-exponential random features: Sharp positive estimators of the gaussian kernel. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=S0xrBMFihS.


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>
