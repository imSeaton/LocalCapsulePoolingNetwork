import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add
from torch_sparse import spspmm, coalesce
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import MessagePassing


# Todo；这里的GCNlayer是没有squash的
class SparseGCNConv(MessagePassing):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(SparseGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None, improved=False):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        # ToDo 在归一化的时候是否需要再添加自环
        # fill_value = 1 if not improved else 2
        # edge_index, edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        # 计算节点度数
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt==float('inf')] = 0
        # D^(1/2)*A*D^(1/2)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight, add_loop=True):
        r"""
            稀疏矩阵模式下的GCN
            add_loop: 如果为True，则会删除edge_index的自环，并且添加新的自环
            input:
                    x: (num_of_nodes, hidden)
                    edge_index: (2, num_of_edges)
                    edge_weight: (num_of_edges)
        """
        x = torch.matmul(x, self.weight)
        # batch中graph中节点的总数
        N = x.size(0)
        # 添加自环
        # ToDO: 这里需要考虑，是否需要删除之前edge_index对角线上的元素, 然后重新添加自环
        if add_loop:
            # 删除原先的自环
            edge_index, edge_weight = remove_self_loops(edge_index=edge_index, edge_attr=edge_weight)
            if self.improved:
                fill_value = 2
            else:
                fill_value = 1
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value=fill_value, num_nodes=N)

        edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.type)

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)