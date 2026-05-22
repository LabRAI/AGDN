import argparse

import torch

from trainer import TSP_Trainer, TSP_Tester
from utils import load_config


def run_train():
    cfg = load_config(mode='train')
    torch.manual_seed(cfg.seed)
    trainer = TSP_Trainer(cfg)
    trainer.train()


def run_test():
    cfg = load_config(mode='test')
    torch.manual_seed(cfg.seed)
    tester = TSP_Tester(cfg)
    tester.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train model')
    parser.add_argument('--test', action='store_true', help='test model')
    args = parser.parse_args()
    if args.train:
        run_train()
    elif args.test:
        run_test()
    else:
        raise NotImplementedError
