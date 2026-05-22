# Unsupervised Learning

This directory contains the unsupervised AGDN training and heatmap generation pipeline.

## TSP-100

```shell
# train
python train.py \
  --num_of_nodes 100 \
  --EPOCHS 300 \
  --batch_size 32 \
  --temperature 3.5 \
  --C1_penalty 20.0 \
  --nlayers 2 \
  --hidden 64 \
  --rescale 1.0 \
  --moment 1 \
  --model agd \
  --device 1

# test
python loadmodel.py \
  --num_of_nodes 100 \
  --batch_size 256 \
  --temperature 3.5 \
  --nlayers 2 \
  --hidden 64 \
  --rescale 1.0 \
  --moment 1 \
  --topk 10 \
  --model agd \
  --device 1

# search
# Modify search settings first:
# - set double Param_T = 0.01
# - set int Rec_Num = 10 to match topk
cd ./Search
bash ./new-solve-100.sh 0.1 6 10 0 30 3 0 0
```

## TSP-200

```shell
# train
python train.py \
  --num_of_nodes 200 \
  --EPOCHS 300 \
  --batch_size 32 \
  --temperature 3.5 \
  --C1_penalty 20.0 \
  --nlayers 2 \
  --hidden 64 \
  --rescale 2.0 \
  --moment 1 \
  --lr 5e-3 \
  --stepsize 20 \
  --topk 20 \
  --model agd \
  --device 0 > train200-agd-ani.log

# generate heatmap
python loadmodel.py \
  --num_of_nodes 200 \
  --batch_size 128 \
  --temperature 3.5 \
  --nlayers 2 \
  --hidden 64 \
  --rescale 2.0 \
  --moment 1 \
  --topk 20 \
  --model agd \
  --device 1

# search
# Modify the model name in solve.sh and the node count in IO.h first.
cd ./Search
bash ./new-solve-200.sh 0 5 30 0 100 2 1 1
```

## TSP-500

```shell
# train
python train.py \
  --num_of_nodes 500 \
  --EPOCHS 300 \
  --batch_size 64 \
  --temperature 3.5 \
  --C1_penalty 10.0 \
  --nlayers 2 \
  --hidden 64 \
  --lr 3e-3 \
  --rescale 4.0 \
  --stepsize 20 \
  --model agd \
  --device 1 > train500-agd-ani.log

# generate heatmap
python loadmodel.py \
  --num_of_nodes 500 \
  --batch_size 128 \
  --temperature 3.5 \
  --nlayers 2 \
  --hidden 64 \
  --rescale 4.0 \
  --moment 1 \
  --model agd \
  --device 1

# search
# Modify the model name in solve.sh and the node count in IO.h first.
cd ./Search
bash ./new-solve-500.sh 0 5 30 0 100 2 1 1
```

## TSP-1000

```shell
# train
python train.py \
  --num_of_nodes 1000 \
  --EPOCHS 300 \
  --batch_size 64 \
  --temperature 3.5 \
  --nlayers 2 \
  --hidden 128 \
  --rescale 4.0 \
  --C1_penalty 10.0 \
  --lr 3e-3 \
  --stepsize 20 \
  --model agd \
  --device 2 > train1000-agd-ani.log

# generate heatmap
python loadmodel.py \
  --num_of_nodes 1000 \
  --batch_size 128 \
  --temperature 3.5 \
  --nlayers 2 \
  --hidden 128 \
  --rescale 4.0 \
  --moment 1 \
  --model agd \
  --device 1

# search
# Modify the model name in solve.sh and the node count in IO.h first.
cd ./Search
bash ./new-solve-1000.sh 0 5 10 0 150 3 1 1
```

