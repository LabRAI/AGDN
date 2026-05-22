import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def _preprocess(dist_matrix, temperature):
    """
    Re-weight dist_matrix by temperature as weight_matrix
    We implement exponential function, and gaussian kernel
    Param
    dist_matrix: [batch, n, n]
    """
    # exponential function, temp=3.5
    weight_matrix = torch.exp(-1. * dist_matrix / temperature)
    # gaussian kernel, temp=1.0, 2.0
    # weight_matrix = torch.exp(-0.5 * dist_matrix ** 2 / (temperature ** 2))
    # mask diagonal
    num_of_nodes = dist_matrix.shape[1]
    mask = torch.ones(num_of_nodes, num_of_nodes).to(dist_matrix.device)
    mask.fill_diagonal_(0)
    weight_matrix = weight_matrix * mask
    return weight_matrix


def _transition_sym(weight_matrix):
    """
    Calculate the transition matrix, Normalize Weight matrix by degree.
    adj_matrix: [batch, n, n]
    """
    batch_size, num_nodes, _ = weight_matrix.shape
    device = weight_matrix.device
    # add self-loop
    I = torch.eye(num_nodes, device=device).unsqueeze(0)  # Shape: [1, n, n]
    A_tilde = weight_matrix + I  # Shape: [batch, n, n]
    # degree matrix
    degrees = A_tilde.sum(dim=2)  # Shape: [batch, n]
    D_inv_sqrt = 1.0 / torch.sqrt(degrees)  # Shape: [batch, n]
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0  # handle 0
    # T = D^{-1/2} * A_tilde * D^{-1/2}
    T = D_inv_sqrt.unsqueeze(2) * A_tilde * D_inv_sqrt.unsqueeze(1)  # Shape: [batch, n, n]
    return T


def _transition_rw(weight_matrix):
    """
    Calculate the row-normalized transition matrix.
    Normalize Adjacency matrix such that each row sums to 1.
    adj_matrix: [batch, n, n]
    """
    batch_size, num_nodes, _ = weight_matrix.shape
    device = weight_matrix.device
    # add self-loop
    I = torch.eye(num_nodes, device=device).unsqueeze(0) * torch.max(weight_matrix)  # Shape: [1, n, n]
    A_tilde = weight_matrix + I  # Shape: [batch, n, n]
    # degree matrix (row sum)
    degrees = A_tilde.sum(dim=2)  # Shape: [batch, n]
    D_inv = 1.0 / degrees  # Shape: [batch, n]
    D_inv[torch.isinf(D_inv)] = 0.0  # handle 0
    # Row-normalized matrix: T = D^{-1} * A_tilde
    T = D_inv.unsqueeze(2) * A_tilde  # Shape: [batch, n, n]
    return T


def _sparsify(transition_matrix, top_k_ratio):
    """
    Sparsify the transition matrix by retaining only the top-k% elements in each row.
    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape [batch, node, node].
        top_k_ratio (float): Ratio of elements to retain in each row (default: 20%).
    Returns:
        torch.Tensor: Sparsified transition matrix with the same shape as the input.
    """
    batch, node, _ = transition_matrix.shape
    top_k = max(1, int(node * top_k_ratio))  # Ensure at least one element is retained

    # Sort the elements in each row and get the top-k threshold
    top_k_values, _ = torch.topk(transition_matrix, k=top_k, dim=-1)
    thresholds = top_k_values[..., -1].unsqueeze(-1)  # The k-th largest value in each row

    # Create a mask to keep elements greater than or equal to the threshold
    mask = transition_matrix >= thresholds

    # Apply the mask to sparsify the matrix
    sparsified_matrix = transition_matrix * mask

    return sparsified_matrix


def get_transition_matrix(dist_matrix, temperature=1.0, sparsify_ratio=0.7, norm='rw'):
    # 0. preprocess dist to weight matrix
    weight_matrix = _preprocess(dist_matrix, temperature)
    # 1. sparsification
    weight_matrix = _sparsify(weight_matrix, sparsify_ratio)
    # 2. weight matrix to transition matrix
    if norm == 'rw':
        transition_matrix = _transition_rw(weight_matrix)
    elif norm == 'sym':
        transition_matrix = _transition_sym(weight_matrix)
    else:
        raise ValueError(f'Norm {norm} not supported.')
    return transition_matrix


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        y_embed = x[:, :, 0]
        x_embed = x[:, :, 1]
        if self.normalize:
            # eps = 1e-6
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
        return pos


class ScalarEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        x_embed = x
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return pos_x


class AGDLayer(nn.Module):
    def __init__(self, hidden_dim, K=10, alpha=0.1):
        super().__init__()
        print('Using Graph Attention Diffusion layer')
        self.hidden_dim = hidden_dim
        self.K = K
        self.alpha = alpha  # teleport ratio
        self.beta = 0.5  # fusion ratio
        self.attn_hidden_dim = 16  # attn dim
        self.lin_hi = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hj = nn.Linear(hidden_dim, hidden_dim)
        self.lin_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin_ei = nn.Linear(hidden_dim, hidden_dim)
        self.lin_ej = nn.Linear(hidden_dim, hidden_dim)
        self.lin_e = nn.Linear(hidden_dim, hidden_dim)
        self.filter_i = nn.Linear(hidden_dim, hidden_dim)
        self.filter_j = nn.Linear(hidden_dim, hidden_dim)
        self.norm_x = nn.BatchNorm1d(hidden_dim, affine=True)
        self.norm_e = nn.BatchNorm1d(hidden_dim, affine=True)
        self.param_ppr_i = Parameter(torch.Tensor(K + 1, 1))
        self.param_ppr_j = Parameter(torch.Tensor(K + 1, 1))
        self.lin_attn_Ti = nn.Sequential(
            nn.Linear(2, self.attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.attn_hidden_dim, 1),
            nn.Softmax(dim=-1)
        )
        self.lin_attn_Tj = nn.Sequential(
            nn.Linear(2, self.attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.attn_hidden_dim, 1),
            nn.Softmax(dim=-1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # init by PPR
        for k in range(self.K + 1):
            self.param_ppr_i.data[k] = self.alpha * (1 - self.alpha) ** k
            self.param_ppr_j.data[k] = self.alpha * (1 - self.alpha) ** k
        self.param_ppr_i.data[-1] = (1 - self.alpha) ** self.K
        self.param_ppr_j.data[-1] = (1 - self.alpha) ** self.K

    def prop_T(self, hi, hj, T):
        # concatenate
        attn_input_Ti = torch.stack([torch.matmul(hi, hj.transpose(-1, -2)) / (self.hidden_dim ** 0.5), T], dim=-1)
        attn_input_Tj = torch.stack(
            [torch.matmul(hj, hi.transpose(-1, -2)) / (self.hidden_dim ** 0.5), T.transpose(-1, -2)], dim=-1)
        # attn score
        attn_Ti = self.lin_attn_Ti(attn_input_Ti).squeeze(-1)
        attn_Tj = self.lin_attn_Tj(attn_input_Tj).squeeze(-1)
        # transition fuse
        Ti = self.beta * T + (1 - self.beta) * attn_Ti
        Tj = self.beta * T.transpose(-1, -2) + (1 - self.beta) * attn_Tj
        return Ti, Tj

    def prop_h(self, hi, hj, Ti, Tj):
        hi = hi * (self.param_ppr_i[0])
        hj = hj * (self.param_ppr_j[0])
        hi_ = hi
        hj_ = hj
        for k in range(self.K):
            hi_ = F.relu(self.filter_i(torch.bmm(Ti, hi_)))
            hi = hi + self.param_ppr_i[k + 1] * hi_
            hj_ = F.relu(self.filter_j(torch.bmm(Tj, hj_)))
            hj = hj + self.param_ppr_j[k + 1] * hj_
        return hi, hj

    def aggr_h(self, hi, hj):
        # self attention
        attn = torch.matmul(hi, hj.transpose(-1, -2)) / (self.hidden_dim ** 0.5)  # [batch, node, node]
        attn = F.softmax(attn, dim=-1)  # [batch, node, node]
        # Weighted aggregation
        hj = torch.einsum('bij,bjc->bic', attn, hj)  # Shape: [batch, node, out_channels]
        # aggr h
        h_prime = self.lin_h(torch.cat([hi, hj], dim=-1))
        return h_prime

    def aggr_e(self, hi, hj, e):
        ei = self.lin_ei(hi)
        ej = self.lin_ej(hj)
        e = self.lin_e(e)
        e = ei.unsqueeze(1) + ej.unsqueeze(2) + e
        attn = torch.matmul(ei, ej.transpose(-1, -2)) / (self.hidden_dim ** 0.5)  # [batch, node, node]
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)  # [batch, node, node, 1]
        # attention score on edge embedding
        e_prime = attn * e  # [batch, node, node, embedding]
        return e_prime

    def aggr_T(self, Ti, Tj):
        T_prime = self.beta * Ti + (1 - self.beta) * Tj.transpose(-1, -2)
        return T_prime

    def forward(self, x, e, T, mode='node'):
        batch_size, num_nodes, hidden_dim = x.shape
        self.num_nodes = num_nodes
        x_in, e_in = x, e
        # map into anisotropy
        hi = self.lin_hi(x)
        hj = self.lin_hj(x)
        # prop T
        Ti, Tj = self.prop_T(hi, hj, T)
        # prop h
        hi, hj = self.prop_h(hi, hj, Ti, Tj)
        # aggr h
        h_prime = self.aggr_h(hi, hj)
        # aggr e
        # e_prime = self.aggr_e(hi, hj, e)
        # aggr T
        # T_prime = self.aggr_T(Ti, Tj)
        # norm
        h_prime = self.norm_x(
            h_prime.view(-1, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim)
        # e_prime = self.norm_e(
        #     e_prime.view(-1, hidden_dim)
        # ).view(batch_size, num_nodes, num_nodes, hidden_dim)
        # dropout
        h_prime = F.dropout(h_prime, p=0.5, training=self.training)
        # e_prime = F.dropout(e_prime, p=0.5, training=self.training)
        # non-linear
        h_prime = F.relu(h_prime)
        # e_prime = F.relu(e_prime)
        # residual
        h_prime = x_in + h_prime
        # e_prime = e_in + e_prime
        # return h, e
        return h_prime, None


class Encoder_AGD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        print('Initializing Anisotropic Graph Diffusion Model...Fixing')
        self.hidden_dim = hidden_dim
        # Modify
        self.sparse_ratio = 0.7
        self.temperature = 0.7
        self.transition_norm = 'rw'
        self.K = 5
        self.teleport_ratio = 0.1
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
        self.node_pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
        self.layers = nn.ModuleList([
            AGDLayer(hidden_dim, K=self.K, alpha=self.teleport_ratio) for _ in range(n_layers)
        ])
        self.lin_node1 = nn.Linear(hidden_dim * (1 + n_layers), hidden_dim)
        self.lin_node2 = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.prob = nn.Softmax(dim=1)

    def node_to_heatmap(self, h):
        e = F.relu(h[:, :, None, :] + h[:, None, :, :] + h.mean(dim=1)[:, None, None, :])
        attn = torch.matmul(h, h.transpose(-1, -2)) / (self.hidden_dim ** 0.5)  # [batch, node, node]
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)  # [batch, node, node, 1]
        hm = attn * e
        return hm

    def forward(self, x, dist_matrix, mode='node'):
        T = get_transition_matrix(dist_matrix, temperature=self.temperature,
                                  sparsify_ratio=self.sparse_ratio, norm=self.transition_norm)
        h = self.node_embed(self.node_pos_embed(x))
        # e = self.edge_embed(self.edge_pos_embed(T))
        e = None
        hs = [h]
        for i, layer in enumerate(self.layers):
            h, e = layer(h, e, T, mode=mode)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        h = self.lin_node1(h)
        h = self.bn(h.view(-1, self.hidden_dim)).view(h.shape)
        h = F.leaky_relu(h)
        h = self.lin_node2(h)
        h = self.prob(h)
        return h
