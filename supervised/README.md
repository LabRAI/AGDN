# Supervised Learning

This directory contains the supervised AGDN and GatedGCN training, testing, and TSP search pipeline.

## Quick Start

Train and test from this directory:

```shell
# train
python main.py --train

# test
python main.py --test
```

Run the search scripts after generating model predictions:

```shell
cd ./search

# TSP100
bash ./new-solve-100.sh 0.1 6 10 0 30 3 0 0

# TSP200
bash ./new-solve-200.sh 0 5 30 0 100 2 1 1

# TSP500
bash ./new-solve-500.sh 0 5 30 0 100 2 1 1

# TSP1000
bash ./new-solve-1000.sh 0 5 10 0 150 3 1 1
```

Before search, update the model name in the solve script and set `Max_City_Num` and `Max_Inst_Num` in `search/code/include/TSP_IO.h` to match the problem size and number of instances.

Get search results:

```shell
python get_results.py -n 200
```

## AGDN Hyperparameters

```text
TSP100
batch_size: 64
rescale: 2.0
hidden_dim: 128
lr: 1e-4
wd: 1.0

TSP200
batch_size: 64
rescale: 2.0
hidden_dim: 128
lr: 1e-4
wd: 1.0

TSP500
batch_size: 32
rescale: 4.0
hidden_dim: 128
lr: 5e-5
wd: 1.0

TSP1000
batch_size: 16
rescale: 4.0
hidden_dim: 128
lr: 5e-5
wd: 1.0
```
## GatedGCN Hyperparameters

```text
TSP100
batch_size: 64
rescale: 2.0
hidden_dim: 128
lr: 1e-4
wd: 1.0

TSP200
batch_size: 32
rescale: 2.0
hidden_dim: 128
lr: 1e-4
wd: 1.0

TSP500
batch_size: 16
rescale: 4.0
hidden_dim: 128
lr: 5e-5
wd: 1.0

TSP1000
batch_size: 4
rescale: 4.0
hidden_dim: 128
lr: 5e-5
wd: 1.0
```
