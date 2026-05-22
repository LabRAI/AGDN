from pathlib import Path

import numpy as np
import torch
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
from tqdm import tqdm


class TSPDataset_Distribution(Dataset):
    def __init__(self, cfg, dataset_type):
        super(TSPDataset_Distribution, self).__init__()
        assert cfg.problem_type == 'distribution'
        assert dataset_type == 'test', 'only support test'
        print(f'Initializing Different Distribution Dataset')
        self.cfg = cfg
        self.data_name, self.sol_name = self.get_filename(cfg)
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

    def get_filename(self, cfg):
        problem = cfg.problem
        distribution = cfg.distribution
        size = cfg.problem_size
        data_dir = cfg.data_dir

        assert problem == "tsp"
        assert size in (100, 1000, 5000, 10000)
        assert distribution in ('uniform', 'clustered1', 'clustered2', 'explosion', 'implosion')

        if size == 100:
            baseline = "Gurobi"
        elif size == 1000:
            baseline = "LKH3_runs10"
        elif size == 5000:
            baseline = "LKH3_runs1"
        elif size == 10000:
            baseline = "LKH3_runs1"
        else:
            raise ValueError(f"Invalid size {size}")

        print(f'Loading dataset with distribution: {distribution}, baseline: {baseline}')

        instance_root = Path(data_dir)
        instance_dir = f"distribution/instances/{problem}{size}/"
        instance_name = f"{problem}{size}_{distribution}.txt"
        instance_file = instance_root.joinpath(instance_dir).joinpath(instance_name)

        solution_root = Path(data_dir)
        solution_dir = f"distribution/solutions/{problem}{size}/{problem}{size}_{distribution}/"
        solution_name = f"{baseline}.txt"
        solution_file = solution_root.joinpath(solution_dir).joinpath(solution_name)

        return instance_file, solution_file

    def process_raw(self):
        tsp_instances = self.read_tsp_instances_from_file(self.data_name)
        baseline_tours, baseline_lens, _ = self.read_solutions_from_file(self.sol_name)
        tour_first_col = baseline_tours[:, 0].unsqueeze(1)  # a[:, 0] shape: [batch], unsqueeze to [batch, 1]
        baseline_tours = torch.cat((baseline_tours, tour_first_col), dim=1)
        return tsp_instances, baseline_tours

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

    def read_solutions_from_file(self, file_path):
        tour_storage = []
        tour_len_storage = []
        ellapsed_time_storage = []
        with open(file_path, 'r', encoding='utf8') as read_file:
            line_text = read_file.readline()
            while line_text:
                tour_text, tour_len_text, ellapsed_time_text = line_text.strip().split(" ")

                tour = [int(val) for val in tour_text.split(",")]
                tour_storage.append(tour)

                tour_len = float(tour_len_text)
                tour_len_storage.append(tour_len)

                ellapsed_time = float(ellapsed_time_text)
                ellapsed_time_storage.append(ellapsed_time)

                line_text = read_file.readline()

        tours = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tour_storage], batch_first=True,
                                                padding_value=0)
        tour_lens = torch.tensor(tour_len_storage)
        time_consumptions = torch.tensor(ellapsed_time_storage)
        return tours, tour_lens, time_consumptions

    def read_tsp_instances_from_file(self, file_path):
        """
        read instances from the given file (should follow the rules in write_tsp_instances_to_file())
        :param file_path: the input data path
        :return: a (num, size, 2) tensor in cpu, multiple tsp instances
        """
        tsp_instances = []
        with open(file_path, 'r', encoding='utf8') as read_file:
            line_text = read_file.readline()
            while line_text:
                splitted_text = line_text.strip().split(" ")
                tsp_instance = []
                for node_text in splitted_text:
                    tsp_instance.append([float(val) for val in node_text.split(",")])
                tsp_instances.append(tsp_instance)
                line_text = read_file.readline()
        return torch.Tensor(tsp_instances)


if __name__ == "__main__":
    ...
