from typing import Union, Optional, Callable

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_scatter import scatter
from torch_sparse import SparseTensor

from torch_geometric.nn import LEConv
from torch_geometric.utils import softmax
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import add_remaining_self_loops

class ASAPooling(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5,
                 GNN: Optional[Callable] = None, dropout: float = 0.0,
                 negative_slope: float = 0.2, add_self_loops: bool = False,
                 **kwargs):
        super(ASAPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN = GNN
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels,
                                         **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        if self.GNN is not None:
            self.gnn_intra_cluster.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """"""
        N = x.size(0)

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        x_pool = x
        if self.GNN is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)

        print(f"edge_index {edge_index.shape}")
        print(f"x_pool {x_pool.shape}")
        x_pool_j = x_pool[edge_index[0]]
        x_q = scatter(x_pool_j, edge_index[1], dim=0, reduce='max')
        x_q = self.lin(x_q)[edge_index[1]]

        score = self.att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=N)

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout, training=self.training)

        v_j = x[edge_index[0]] * score.view(-1, 1)
        x = scatter(v_j, edge_index[1], dim=0, reduce='add')

        # Cluster selection.
        fitness = self.gnn_score(x, edge_index).sigmoid().view(-1)
        perm = topk(fitness, self.ratio, batch)
        x = x[perm] * fitness[perm].view(-1, 1)
        batch = batch[perm]

        # Graph coarsening.
        row, col = edge_index
        A = SparseTensor(row=row, col=col, value=edge_weight,
                         sparse_sizes=(N, N))
        S = SparseTensor(row=row, col=col, value=score, sparse_sizes=(N, N))
        S = S[:, perm]

        A = S.t() @ A @ S

        if self.add_self_loops:
            A = A.fill_diag(1.)
        else:
            A = A.remove_diag()

        row, col, edge_weight = A.coo()
        edge_index = torch.stack([row, col], dim=0)

        return x, edge_index, edge_weight, batch, perm, S

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__,
                                         self.in_channels, self.ratio)