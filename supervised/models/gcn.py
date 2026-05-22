import torch
import torch.nn.functional as F
from torch import nn


class GatedGCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
            gated: Whether to use edge gating (True/False)
        """
        super(GatedGCNLayer, self).__init__()
        print('Using GatedGCN Layer')
        self.hidden_dim = hidden_dim
        self.aggregation = "max"
        self.norm = "batch"
        self.learn_norm = True
        self.track_norm = False
        self.gated = True

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=self.learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=self.learn_norm, track_running_stats=self.track_norm)
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=self.learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=self.learn_norm, track_running_stats=self.track_norm)
        }.get(self.norm, None)

    def forward(self, h, e, graph):
        """
        Args:
            h: Input node features (B x V x H)
            e: Input edge features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
        Returns:
            Updated node and edge features
        """
        batch_size, num_nodes, hidden_dim = h.shape
        h_in = h
        e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # Update node features
        h = Uh + self.aggregate(Vh, graph, gates)  # B x V x H

        # Normalize node features
        h = self.norm_h(
            h.view(batch_size * num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h

        # Normalize edge features
        e = self.norm_e(
            e.view(batch_size * num_nodes * num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        h = h_in + h
        e = e_in + e

        return h, e

    def aggregate(self, Vh, graph, gates):
        """
        Args:
            Vh: Neighborhood features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
            gates: Edge gates (B x V x V x H)
        Returns:
            Aggregated neighborhood features (B x V x H)
        """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H

        # Enforce graph structure through masking
        # 1 represents connect
        # Vh[graph.unsqueeze(-1).expand_as(Vh).bool()] = 0
        mask = graph.unsqueeze(-1).expand_as(Vh).bool()
        Vh = torch.where(mask, Vh, torch.tensor(0.0))

        if self.aggregation == "mean":
            return torch.sum(Vh, dim=2) / torch.sum(1 - graph, dim=2).unsqueeze(-1).type_as(Vh)

        elif self.aggregation == "max":
            return torch.max(Vh, dim=2)[0]

        else:
            return torch.sum(Vh, dim=2)


class Encoder_GatedGCN(nn.Module):
    def __init__(self, encoder_layer_num, hidden_dim, **cfg):
        super(Encoder_GatedGCN, self).__init__()
        print('initializing GatedGCN Encoder')
        self.init_embed_edges = nn.Embedding(2, hidden_dim)
        self.init_embed_nodes = nn.Linear(2, hidden_dim)
        self.project_node_emb = nn.Linear(hidden_dim, hidden_dim)
        self.project_graph_emb = nn.Linear(hidden_dim, hidden_dim)
        self.pred_heatmap = nn.Linear(hidden_dim, 2)

        self.layers = nn.ModuleList([
            GatedGCNLayer(hidden_dim)
            for _ in range(encoder_layer_num)
        ])

    def node_to_edge(self, x):
        Ux = self.project_node_emb(x)
        Gx = self.project_graph_emb(x.mean(dim=1))
        edge_embeddings = F.relu(Ux[:, :, None, :] + Ux[:, None, :, :] + Gx[:, None, None, :])
        return edge_embeddings

    def forward(self, x, adj_matrix):
        """
        Args:
            x: Input node features change (B x V x H) to (B x V x 2)
            graph: Graph adjacency matrices (B x V x V)
        Returns:
            Updated node features (B x V x H)
        """
        # Embed node features
        x = self.init_embed_nodes(x)
        # Embed edge features
        e = self.init_embed_edges(adj_matrix.type(torch.long))

        for layer in self.layers:
            x, e = layer(x, e, adj_matrix)
        edge_embeddings = self.node_to_edge(x)
        heatmap = self.pred_heatmap(edge_embeddings)
        return heatmap
