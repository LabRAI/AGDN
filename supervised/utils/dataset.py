import numpy as np
import torch
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
from tqdm import tqdm


class TSPDataset(Dataset):
    def __init__(self, cfg, dataset_type):
        super(TSPDataset, self).__init__()
        assert cfg.problem_type == 'normal'
        print(f'Initializing Normal Dataset')
        self.cfg = cfg
        self.data_name, self.sol_name = self.get_filename(cfg, dataset_type)
        self.batch_size = cfg.batch_size
        # process raw data
        self.data, self.target_nodes = self.process_raw()
        # process tour to target edge
        self.target_edges = self.process_target_nodes(self.target_nodes)
        # agd use dist_matrix, gcn use adj_matrix
        if cfg.model_name == 'agd':
            self.graph = self.process_dist(self.data)
            print(f'AGD use dist_matrix with sparse_ratio: {cfg.sparse_ratio}, temperature: {cfg.temperature}')
        elif cfg.model_name == 'gcn':
            self.graph = self.process_adj(self.data, cfg.sparse_ratio)
            print(f'GCN use adj_matrix with sparse_ratio: {cfg.sparse_ratio}')
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        node = self.data[idx]
        graph = self.graph[idx]
        target_nodes = self.target_nodes[idx]
        target_edges = self.target_edges[idx]
        return {
            'nodes': node,
            'graph': graph,
            'target_nodes': target_nodes,
            'target_edges': target_edges,
        }

    def get_filename(self, cfg, dataset_type):
        if dataset_type == 'train':
            data_name = cfg.train_data_path.format('instance', cfg.problem_size)
            sol_name = cfg.train_data_path.format('sol', cfg.problem_size)
        elif dataset_type == 'test':
            data_name = cfg.test_data_path.format('instance', cfg.problem_size)
            sol_name = cfg.test_data_path.format('sol', cfg.problem_size)
        else:
            raise ValueError(f'Invalid dataset_type: {dataset_type}')
        print(f'Loading data: {data_name}')
        return data_name, sol_name

    def process_raw(self):
        data = np.load(self.data_name)
        sample_num = data.shape[0]
        mean = np.mean(data, axis=1)
        data = self.cfg.rescale * (data - mean.reshape((sample_num, 1, 2)))
        target_nodes = np.load(self.sol_name)
        return torch.from_numpy(data).float(), torch.from_numpy(target_nodes).long()

    def process_dist(self, data):
        dist_matrix = torch.zeros([data.shape[0], data.shape[1], data.shape[1]])
        for i in tqdm(range(data.shape[0]), desc="data processing dist"):
            dist_matrix[i] = torch.from_numpy(distance_matrix(data[i], data[i]))
        return dist_matrix

    def process_adj(self, data, sparse_ratio):
        batch_size, n = data.shape[0], data.shape[1]
        dist_matrix = torch.zeros([batch_size, n, n])
        for i in tqdm(range(data.shape[0]), desc="data processing dist"):
            dist_matrix[i] = torch.from_numpy(distance_matrix(data[i], data[i]))
        adj_matrix = torch.zeros_like(dist_matrix, dtype=torch.long)
        k = int(n * sparse_ratio)
        for i in tqdm(range(batch_size), desc="data processing adj"):
            for j in range(n):
                row = dist_matrix[i, j]
                threshold = torch.topk(row, k, largest=False).values[-1]
                adj_matrix[i, j] = (row <= threshold).long()
        return adj_matrix

    def target_node_to_edge(self, target_node):
        num_nodes = len(target_node)
        target_edge = np.zeros((num_nodes, num_nodes))
        for idx in range(len(target_node) - 1):
            i = target_node[idx]
            j = target_node[idx + 1]
            target_edge[i][j] = 1
            target_edge[j][i] = 1
        # Add final connection
        target_edge[j][target_node[0]] = 1
        target_edge[target_node[0]][j] = 1
        return target_edge

    def process_target_nodes(self, target_nodes):
        target_nodes = target_nodes[:, :-1]
        target_edges = torch.zeros(target_nodes.shape[0], target_nodes.shape[1], target_nodes.shape[1])
        for idx in tqdm(range(len(target_nodes)), desc="data processing target"):
            target_edges[idx] = torch.from_numpy(self.target_node_to_edge(target_nodes[idx]))
        return target_edges.long()


class TSPDataset_adj(Dataset):
    def __init__(self, cfg, dataset_type='train'):
        super(TSPDataset_adj, self).__init__()
        print('Dataset using adj_matrix with sparse: ', cfg.sparse_ratio)
        self.cfg = cfg
        self.data_name, self.sol_name = self.get_filename(cfg, dataset_type)
        self.batch_size = cfg.batch_size
        self.data, self.target_nodes = self.process_raw()
        self.target_edges = self.process_target_nodes(self.target_nodes)
        self.adj_matrix = self.process_adj(self.data, sparse_ratio=cfg.sparse_ratio)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        node = self.data[idx]
        adj_matrix = self.adj_matrix[idx]
        target_nodes = self.target_nodes[idx]
        target_edges = self.target_edges[idx]
        return {
            'nodes': node,
            'dist_matrix': adj_matrix,
            'target_nodes': target_nodes,
            'target_edges': target_edges,
        }

    def get_filename(self, cfg, dataset_type='train'):
        if dataset_type == 'train':
            data_name = cfg.train_data_path.format('instance', cfg.problem_size)
            sol_name = cfg.train_data_path.format('sol', cfg.problem_size)
        elif dataset_type == 'val':
            data_name = cfg.val_data_path.format('instance', cfg.problem_size)
            sol_name = cfg.val_data_path.format('sol', cfg.problem_size)
        elif dataset_type == 'test':
            data_name = cfg.test_data_path.format('instance', cfg.problem_size)
            sol_name = cfg.test_data_path.format('sol', cfg.problem_size)
        else:
            raise ValueError(f'Invalid mode: {dataset_type}')
        print(f'Loading data: {data_name}')
        return data_name, sol_name

    def process_raw(self):
        data = np.load(self.data_name)
        sample_num = data.shape[0]
        mean = np.mean(data, axis=1)
        data = self.cfg.rescale * (data - mean.reshape((sample_num, 1, 2)))
        target_nodes = np.load(self.sol_name)
        return torch.from_numpy(data).float(), torch.from_numpy(target_nodes).long()

    def process_adj(self, data, sparse_ratio):
        batch_size, n = data.shape[0], data.shape[1]
        dist_matrix = torch.zeros([batch_size, n, n])
        for i in tqdm(range(data.shape[0]), desc="data processing dist"):
            dist_matrix[i] = torch.from_numpy(distance_matrix(data[i], data[i]))
        adj_matrix = torch.zeros_like(dist_matrix, dtype=torch.long)
        k = int(n * sparse_ratio)
        for i in tqdm(range(batch_size), desc="data processing adj"):
            for j in range(n):
                row = dist_matrix[i, j]
                threshold = torch.topk(row, k, largest=False).values[-1]
                adj_matrix[i, j] = (row <= threshold).long()
        return adj_matrix

    def target_node_to_edge(self, target_node):
        num_nodes = len(target_node)
        target_edge = np.zeros((num_nodes, num_nodes))
        for idx in range(len(target_node) - 1):
            i = target_node[idx]
            j = target_node[idx + 1]
            target_edge[i][j] = 1
            target_edge[j][i] = 1
        # Add final connection
        target_edge[j][target_node[0]] = 1
        target_edge[target_node[0]][j] = 1
        return target_edge

    def process_target_nodes(self, target_nodes):
        target_nodes = target_nodes[:, :-1]
        target_edges = torch.zeros(target_nodes.shape[0], target_nodes.shape[1], target_nodes.shape[1])
        for idx in tqdm(range(len(target_nodes)), desc="data processing target"):
            target_edges[idx] = torch.from_numpy(self.target_node_to_edge(target_nodes[idx]))
        return target_edges.long()
