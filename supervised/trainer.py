import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from models import create_model
from utils import get_dataloader


class TSP_Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = create_model(cfg)
        self.train_loader = get_dataloader(cfg, dataset_type='train')
        self.epochs = cfg.epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.step_size, gamma=cfg.step_decay)
        self.device = int(cfg.device) if cfg.device >= 0 else 'cpu'
        self.model.to(self.device)

    def train(self):
        self.model.train()
        for i in tqdm(range(1, self.epochs + 1), desc='Training'):
            loss = self.train_step(self.train_loader)
            print('Epoch: {}, Loss: {}'.format(i, loss))
            if i % self.cfg.ckpt_step == 0 and i != 0:
                self.save_model(i)

    def train_step(self, loader):
        total_loss = 0
        num_batches = 0
        for batch in loader:
            nodes = batch['nodes'].to(self.device)
            graph = batch['graph'].to(self.device)
            target_edges = batch['target_edges'].to(self.device)
            # forward
            h = self.model(nodes, graph)
            # calculate loss
            loss = self.loss_node(h, target_edges)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            print('Batch: {}, Loss: {}'.format(num_batches, loss.item()))
        self.scheduler.step()
        return total_loss / num_batches

    def save_model(self, i):
        cfg = self.cfg
        save_path = f'{cfg.output_dir}/models/TSP{cfg.problem_size}/'
        save_name = f'{cfg.model_name}_{cfg.layer_name}_{cfg.encoder_layer_num}_{cfg.K}_{cfg.sparse_ratio}_{i}.pt'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(), save_path + save_name)
        print(f'Model saved to {save_path}{save_name}')

    def _get_heatmap(self, output):
        """
        :param output: [batch, n, n, 2]
        :return heat_map: [batch, n, n]
        """
        return output[:, :, :, 1]

    def _weighted_bce_loss(self, pred, target, weight):
        loss = F.binary_cross_entropy(pred, target, reduction='none')
        weighted_loss = loss * (weight[1] * target + weight[0] * (1 - target))
        return weighted_loss.mean()

    def loss_node(self, node_pred, target_edges):
        """
        :param node_pred: [b, n, n]
        :param target_edges: [b, n, n]
        :return loss:
        """
        tmp_target_edges = target_edges.cpu().numpy().flatten()
        cw = compute_class_weight('balanced', classes=np.unique(tmp_target_edges), y=tmp_target_edges)
        loss = F.cross_entropy(node_pred.permute(0, 3, 1, 2), target_edges,
                               weight=torch.tensor(cw, dtype=torch.float).to(node_pred.device),
                               reduction='mean')
        return loss


class TSP_Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = int(cfg.device) if cfg.device >= 0 else 'cpu'
        self.test_loader = get_dataloader(cfg, dataset_type='test')
        self.test_size = len(self.test_loader.dataset)
        self.test_nodes = cfg.problem_size
        self.topk = cfg.topk
        model = create_model(cfg)
        self.model = self.load_model(model)
        self.model.to(self.device)

    def inference(self):
        self.model.eval()
        heatmap_idx, heatmap_val, save_nodes, save_tour = self.inference_steps(self.test_loader)
        self._save_inference(heatmap_idx, heatmap_val, save_nodes, save_tour)

    def inference_mixscore(self):
        print('Infer Heatmap')
        for batch in tqdm(self.test_loader, desc='Inference Heatmap'):
            batch_size = batch['nodes'].shape[0]
            nodes = batch['nodes'].to(self.device)
            graph = batch['graph'].to(self.device)
            # forward
            h, MS = self.model(nodes, graph)
            # heatmap = self._get_heatmap(h)
            break
        return MS

    def inference_steps(self, loader):
        heatmap_idx = np.zeros((self.test_size, self.test_nodes, self.topk))
        heatmap_val = np.zeros((self.test_size, self.test_nodes, self.topk))
        save_nodes = np.zeros((self.test_size, self.test_nodes, 2))
        save_tour = np.zeros((self.test_size, self.test_nodes + 1))
        count = 0
        inference_time = 0.0
        for batch in tqdm(loader, desc='Inference'):
            batch_size = batch['nodes'].shape[0]
            nodes = batch['nodes'].to(self.device)
            graph = batch['graph'].to(self.device)
            target_nodes = batch['target_nodes'].to(self.device)
            # forward
            t0 = time.time()
            h = self.model(nodes, graph)
            t1 = time.time()
            inference_time += t1 - t0
            # get heatmap
            heatmap = self._get_heatmap(h)
            # save
            pred_idx = torch.topk(heatmap, self.topk, dim=2).indices
            pred_val = torch.topk(heatmap, self.topk, dim=2).values
            heatmap_idx[count:batch_size + count] = pred_idx.detach().cpu().numpy()
            heatmap_val[count:batch_size + count] = pred_val.detach().cpu().numpy()
            save_nodes[count:batch_size + count] = nodes.detach().cpu().numpy()
            save_tour[count:batch_size + count] = target_nodes.detach().cpu().numpy()
            count = count + batch_size
        print("Inference Time: {:.5f}s".format(inference_time))
        return heatmap_idx, heatmap_val, save_nodes, save_tour

    def load_model(self, model):
        cfg = self.cfg
        load_model_path = f'{cfg.output_dir}/models/'
        load_model_name = cfg.load_model_name
        print(f'Loading model from {load_model_name}')
        model.load_state_dict(torch.load(load_model_path + load_model_name))
        return model

    def _get_heatmap(self, output):
        """
        :param output: [batch, n, n, 2]
        :return heat_map: [batch, n, n]
        """
        return output[:, :, :, 1]

    def _save_inference(self, heatmap_idx, heatmap_val, save_nodes, save_tour):
        Q = save_nodes
        A = save_tour
        C = heatmap_idx
        V = heatmap_val
        inference_dir = './search/heatmap/'
        if not os.path.exists(inference_dir):
            os.makedirs(inference_dir)
        inference_name = f"heatmap_{self.cfg.model_name}_{self.cfg.problem_type}_TSP{self.test_nodes}_{heatmap_idx.shape[0]}.txt"
        with open(inference_dir + inference_name, "w") as f:
            for i in tqdm(range(Q.shape[0]), desc='Inference writing'):
                for j in range(Q.shape[1]):
                    f.write(str(Q[i][j][0]) + " " + str(Q[i][j][1]) + " ")
                f.write("output ")
                for j in range(A.shape[1]):
                    f.write(str(int(A[i][j] + 1)) + " ")
                f.write("indices ")
                for j in range(C.shape[1]):
                    for k in range(self.topk):
                        if C[i][j][k] == j:
                            f.write("-1" + " ")
                        else:
                            f.write(str(int(C[i][j][k] + 1)) + " ")
                f.write("value ")
                for j in range(V.shape[1]):
                    for k in range(self.topk):
                        f.write(str(V[i][j][k]) + " ")
                f.write("\n")
                if i == heatmap_idx.shape[0] - 1:
                    break
        print('Inference saved in ', inference_dir + inference_name)
