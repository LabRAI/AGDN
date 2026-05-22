# Anisotropic Graph Diffusion

## Quick Start

run train and test

```shell
# train
python main.py --train
# test
python main.py --test
```

search

Remember:

1. modify model name in solve.sh
2. modify `IO.h` Max_City_Num equals to problem_num, Max_Inst_Num equals to instance_num

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

get results

```shell
python get_results.py -n 200
```

agd hyperparams

```md
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

gcn hyperparams

```md
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
