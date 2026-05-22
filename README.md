# AGDN: Learning to Solve Traveling Salesman Problem with Anisotropic Graph Diffusion Network

Official implementation of **AGDN: Learning to Solve Traveling Salesman Problem with Anisotropic Graph Diffusion Network**.

**Authors:** Bolin Shen, Ziwei Huang, Zhiguang Cao, and Yushun Dong

**Venue:** Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, August 9--13, 2026, Jeju Island, Republic of Korea.

This repository contains the supervised and unsupervised learning pipelines for solving Euclidean Traveling Salesman Problem (TSP) instances with Anisotropic Graph Diffusion Network (AGDN), together with the search scripts used to decode model predictions into tours.

## Quick Start

Clone the repository and install the Python dependencies:

```shell
git clone https://github.com/LabRAI/AGDN.git
cd AGDN
pip install -r requirements.txt
```

Run the supervised pipeline:

```shell
cd supervised
python main.py --train
python main.py --test
```

Run the supervised search step:

```shell
cd supervised/search
bash ./new-solve-100.sh 0.1 6 10 0 30 3 0 0
```

Run the unsupervised pipeline:

```shell
cd unsupervised
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
  --device 0
```

See [supervised/README.md](supervised/README.md) and [unsupervised/README.md](unsupervised/README.md) for the full training, inference, search, and hyperparameter commands.

## Requirements

The code was developed for Python 3 with PyTorch and CUDA-enabled GPU training. Install the main Python dependencies with:

```shell
pip install -r requirements.txt
```

The search component under `supervised/search` also requires a C++ compiler such as `g++`.

## Repository Structure

```text
AGDN-Official/
├── supervised/       # Supervised AGDN/GatedGCN training, testing, and TSP search
├── unsupervised/     # Unsupervised AGDN training and heatmap generation
├── requirements.txt  # Python dependencies
└── README.md         # Project overview
```

## Resources

This repository builds on and benefits from the following open-source projects:

- GatedGCN: [chaitjo/graph-convnet-tsp](https://github.com/chaitjo/graph-convnet-tsp)
- Unsupervised learning: [yimengmin/UTSP](https://github.com/yimengmin/UTSP)
- MCTS for TSP: [Spider-scnu/Monte-Carlo-tree-search-for-TSP](https://github.com/Spider-scnu/Monte-Carlo-tree-search-for-TSP)

We thank the authors and contributors for releasing their code.

## Citation

```
To be updated.
```
