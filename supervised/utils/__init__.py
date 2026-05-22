from torch.utils.data import DataLoader

from utils.config import load_config
from utils.dataset import TSPDataset
from utils.dataset_distribution import TSPDataset_Distribution


def get_dataloader(cfg, dataset_type):
    if cfg.problem_type == 'normal':
        dataset = TSPDataset(cfg, dataset_type)
    elif cfg.problem_type == 'distribution':
        dataset = TSPDataset_Distribution(cfg, dataset_type)
    else:
        raise NotImplementedError
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    return loader
