import torch
import torch.nn as nn
import torch.nn.functional as F
# from DenseGCNConv import DenseGCNConv
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import dense_to_sparse
from utils import StAS, squash, transfer_c_index


class GlobalCapsulePooling_2(nn.Module):
    def __init__(self, num_targets, in_channels, out_channels, dropout, num_routes):
        super(GlobalCapsulePooling_2, self).__init__()
        self.num_target_cap = num_targets
        self.prim_cap_dim = in_channels
        self.target_cap_dim = out_channels
        self.dropout = nn.Dropout(p=dropout)
        self.num_routes = num_routes
        self.bn_feat = nn.BatchNorm1d(self.prim_cap_dim)
        # 用GCN将数据映射到不同的空间中，进行胶囊网络
        for i in range(self.num_target_cap):
            # self.convs.append(SparseGCNConv(self.prim_cap_dim, self.target_cap_dim, improved=False, bias=True))
            self.convs.append(GCNConv(in_channels=in_channels, out_channels=out_channels))

    def forward(self, x, edge_index, edge_weight=None, batch=None, final_layer=True):
        """
        :param x:                   (num_graphs * num_primary_caps, hidden)
        :param edge_index:          (2, num_edges)
        :param edge_weight:         (num_edges)
        :param batch:               (num_nodes)
        :param final_layer:         bool mark whether the layer is the last layer
        :return:
                                    if final_layer == False:
                                        output: out:            (num_graph * num_target_cap, hidden)
                                                edge_index:     (2, new_edges)
                                                edge_weight:    (num_edges, )
                                                batch:          (num_nodes, )
                                    elif final_layer == True:
                                        output: out:            (num_graph, num_target_cap)
        """
        # shape of x: (num_graphs * num_primary_caps, hidden)
        x = self.bn_feat(x)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index[1], dtype=x.dtype, device=edge_index.device, requires_grad=False)
        u_hat = []
        # 这里的GCN不对
        for i, conv in enumerate(self.convs):
            # ToDo: 这里使用了全局池化S_T*A*S的自环
            temp = conv(x, edge_index, edge_weight)
            u_hat.append(temp)
        # (num_graphs * num_primary_caps, num_target_cap, hidden)
        u_hat = torch.stack(u_hat, dim=1)
        # (num_graphs * num_primary_caps, )
        node_degrees = scatter_add(src=edge_weight, index=edge_index[0], dim=0)
        # (num_graphs * num_primary_caps, 1)
        b = node_degrees.unsqueeze(dim=-1)
        # (num_graphs * num_primary_caps, num_target_cap, 1)
        b = b.repeat(1, self.num_target_cap).unsqueeze(dim=-1)
        # (num_graphs * num_primary_caps, num_target_cap, hidden)
        temp_u_hat = u_hat.detach()
        for i in range(self.num_routes - 1):
            c = F.softmax(b, dim=1)
            # (num_graph, num_target_caps, hidden)
            s = scatter_add((c*temp_u_hat), batch, dim=0)
            v = squash(s, dim=-1)
            # (num_graph * num_primary_cap, num_target_caps, hidden)
            v = v[batch]
            # (num_graph * num_primary_cap, num_target_cap, 1)
            u_produce_v = (temp_u_hat * v).sum(dim=-1, keepdim=True)
            b = b + u_produce_v
        # (num_graph * num_primary_cap, num_target_cap, 1)
        c = F.softmax(b, dim=1)
        s = scatter_add((c*u_hat), batch, dim=0)
        # 残差连接
        x_mean = scatter_mean(x, batch, dim=0)
        x_mean = x_mean.unsqueeze(dim=1)
        s = s + x_mean
        v = squash(s, dim=-1)
        if not final_layer:
            c = c.squeeze(dim=-1)
            # 将c转换为每个graph独立的cluster_assignment_matrix
            # (num_nodes, self.num_target_cap) --> (num_nodes, num_graph * num_target_cap)
            c = transfer_c_index(c=c, batch=batch)
            # S_index: (2, num_new_edges)
            # S_value: (num_new_edges)
            S_index, S_value = dense_to_sparse(c)
            # 这里需要考虑N 和 kN分别为多少
            # ToDo: 这里未去除自环
            # ToDo 这里需要将shape为(num_nodes, 10)的C转换为(num_nodes, num_graph*10)的C 才能够进行STAS

            edge_index, edge_weight = StAS(edge_index, edge_weight, S_index, S_value, x.device, x.shape[0], 10)
            num_graphs = v.shape[0]

            # (num_graph, num_target_cap, hidden) --> (num_graph * num_target_cap, hidden)
            out = v.view(-1, self.target_cap_dim)
            # (num_nodes,)
            batch = torch.Tensor([i for i in range(num_graphs) for j in range(self.num_target_cap)], device=x.device)
            return out, edge_index, edge_weight, batch

        else:
            out = v.norm(-1)
            # (batch, num_target_cap)
            return out