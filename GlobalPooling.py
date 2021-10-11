"""
    A global pooling mechanism using capsule network with structure information directing the routing
    between low-level capsules and high-level capsules

    input:
            x shape: (num_of_all_nodes, hidden)
            edge_index: (2, num_of_edges)
            weight: (num_of_edges)
    output:
            x shape: (num_of_target, hidden)
            edge_index: (2, num_of_new_edges)
            weight: (num_of_new_edges)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from SparseGCNConv import SparseGCNConv
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from utils import StAS, squash, transfer_c_index


class GlobalCapsulePooling(nn.Module):
    def __init__(self, num_targets, in_channels, out_channels, dropout, num_routes=3, final_layer=False):
        super(GlobalCapsulePooling, self).__init__()
        self.num_target_cap = num_targets
        self.prim_cap_dim = in_channels
        self.target_cap_dim = out_channels
        self.dropout = nn.Dropout(p=dropout)
        self.num_routes = num_routes
        self.final_layer = final_layer
        self.bn_feat = nn.BatchNorm1d(self.prim_cap_dim)
        self.convs = torch.nn.ModuleList()
        # 用GCN将数据映射到不同的空间中，进行胶囊网络
        for i in range(self.num_target_cap):
            # self.convs.append(SparseGCNConv(self.prim_cap_dim, self.target_cap_dim, improved=False, bias=True))
            self.convs.append(GCNConv(in_channels=in_channels, out_channels=out_channels))

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
            将一个batch中所有的低层节点并行处理，让每个graph 生成最终的高层节点

            input： x:               (num_nodes, hidden)
                    edge_index:     (2, num_nodes)
                    edge_weight:    (num_nodes)
                    batch:          (num_nodes)
                    final_layer:    bool  mark whether the layer is the last layer

            if final_layer == False:
                output: out:            (num_graph * num_target_cap, hidden)
                        edge_index:     (2, new_edges)
                        edge_weight:    (num_edges, )
                        batch:          (num_nodes, )
            elif final_layer == True:
                output: out:            (num_graph, num_target_cap)
        """
        # 这里的num_nodes是不是一个loader中的nodes
        # (num_nodes, hidden)
        x = self.bn_feat(x)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index[1], dtype=x.dtype, device=edge_index.device, requires_grad=False)
        u_hat = []
        for i, conv in enumerate(self.convs):
            # ToDo: 这里局部池化之后，S_T*A*S邻接矩阵的自环被设置为1
            temp = conv(x, edge_index, edge_weight)
            u_hat.append(temp)
        # (num_of_nodes, num_target_cap, hidden)
        u_hat = torch.stack(u_hat, dim=1)

        # (num_nodes, )
        node_degrees = scatter_add(src=edge_weight, index=edge_index[0], dim=0)
        # (num_nodes, 1)
        b = node_degrees.unsqueeze(dim=-1)
        # (num_nodes, self.num_target_cap, 1)
        b = b.repeat(1, self.num_target_cap).unsqueeze(dim=-1)
        # (num_of_nodes, num_target_cap, hidden)
        temp_u_hat = u_hat.detach()
        for i in range(self.num_routes - 1):
            # ToDo： 初始部分的节点度数作为初始路由系数，但是每个节点在多个通道的节点度数都相同，没有太大的区分度
            # 这部分softmax是所有节点对于每个通道的权重进行softmax
            # (num_nodes, self.num_target_cap, 1)
            c = F.softmax(b, dim=1)
            # 加权求和
            # (num_graph, num_target_cap, hidden)
            s = scatter_add((c * temp_u_hat), batch, dim=0)
            v = squash(s, dim=-1)
            # (num_nodes, num_target_cap, hidden)
            v = v[batch]
            # (num_of_nodes, self.num_target_cap, 1)
            u_produce_v = (temp_u_hat * v).sum(dim=-1, keepdim=True)
            # ToDo: 这部分修正每个节点再各个通道的权重，
            b = b + u_produce_v
        # (num_nodes, self.num_target_cap, 1)
        c = F.softmax(b, dim=1)
        # (num_graphs, self.num_target_cap, hidden)
        s = scatter_add((c * u_hat), batch, dim=0)
        # 残差连接
        # (num_nodes, self.num_target_cap, hidden)
        # (num_graphs, hidden)
        x_mean = scatter_mean(x, batch, dim=0)
        # (num_grpahs, 1, hidden)
        x_mean = x_mean.unsqueeze(dim=1)
        s = s + x_mean
        # (num_graph, self.num_target_cap, hidden)
        v = squash(s, dim=-1)
        # 1.(num_nodes, 10)
        # 2.(num_nodes, self.num_target_cap)
        if not self.final_layer:
            c = c.squeeze(dim=-1)
            # 将c转换为每个graph独立的cluster_assignment_matrix
            # (num_nodes, self.num_target_cap) --> (num_nodes, num_graph * num_target_cap)
            c = transfer_c_index(c=c, batch=batch)
            N = c.size(0)
            kN = c.size(1)
            # ToDo:这里有很多的0
            S_index, S_value = dense_to_sparse(c)
            # 这里需要考虑N 和 kN分别为多少
            # ToDo: 这里未去除自环
            # ToDo 这里需要将shape为(num_nodes, 10)的C转换为(num_nodes, num_graph*10)的C 才能够进行STAS
            dense_new_edge_index = to_dense_adj(S_index, edge_attr=S_value)[0]
            edge_index, edge_weight = StAS(edge_index, edge_weight, S_index, S_value, x.device, N, kN)

            num_graphs = v.shape[0]

            # (num_graph, num_target_cap, hidden) --> (num_graph * num_target_cap, hidden)
            out = v.view(-1, self.target_cap_dim)
            # (num_nodes,)
            batch = torch.tensor([i for i in range(num_graphs) for j in range(self.num_target_cap)], dtype=batch.dtype).to(x.device)
            # print(f"edge_index.shape {edge_index.shape}")
            # print(f"edge_weight.shape {edge_weight.shape}")
            # print(f"batch.shape {batch.shape}")
            return out, edge_index, edge_weight, batch

        else:
            out = v.norm(-1, dim=-1)
            # (batch, num_target_cap)
            return out