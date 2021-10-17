import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax
from torch_geometric.nn.pool.topk_pool import topk
from utils import squash_1, squash_2, graph_connectivity
from SparseGCNConv import SparseGCNConv
from torch_geometric.nn import GCNConv

class LocalCapsulePooling(nn.Module):
    """输入一个batch的graph，对其进行池化，选出重要的cluster"""

    def __init__(self, hidden, ratio, dropout_att=0, dataset_name='DD'):
        super(LocalCapsulePooling, self).__init__()
        self.in_channels = hidden
        self.out_channels = hidden
        self.ratio = ratio
        # self.negative_slope = negative_slope
        self.dropout_att = dropout_att
        # self.gcn_transform = GCNConv(self.in_channels, self.out_channels)
        self.gcn_transform = GCNConv(self.in_channels, self.out_channels)
        self.bn_feat = nn.BatchNorm1d(hidden)
        self.bn_feat_2 = nn.BatchNorm1d(hidden)
        self.score_add = nn.Linear(self.in_channels, 1)
        if dataset_name in ['DD', 'MUTAG', 'NCI109', 'NCI1', 'ENZYMES', 'FRANKENSTEIN', 'REDDIT-BINARY',]:
            self.squash = squash_1
        elif dataset_name in ['PROTEINS',  'IMDB-BINARY', 'IMDB-MULTI',  'REDDIT-MULTI', 'COLLAB']:
            self.squash = squash_2
        else:
            print('Wrong Dataset')
        self.reset_parameters()

    def reset_parameters(self):
        self.gcn_transform.reset_parameters()
        self.bn_feat.reset_parameters()
        self.score_add.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # ToDO: 这里可以看需要是否删除
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        fill_value = 1
        # x = self.bn_feat(x)
        # 一个batch中每个graph中节点的数量
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        # ToDo: 这部分自环还是需要考虑的，添加之后会影响每个节点到高层的连接
        # 添加自环
        edge_index, edge_weight = add_remaining_self_loops(edge_index=edge_index, edge_weight=edge_weight,
                                                           fill_value=fill_value, num_nodes=num_nodes.sum())
        # 节点总数
        N = x.size(0)
        """用胶囊网络进行生成高层cluster特征和cluster assignment matrix"""
        # 一层GCN
        # ToDo
        x_pool_j = self.gcn_transform(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x_pool_j = self.bn_feat(x_pool_j)
        x_pool_j = self.squash(x_pool_j, dim=-1)
        # print(f"x_pool_j.grad {x_pool_j.requires_grad} \t {x_pool_j.grad}\t\t 'x_pool_j[0][:5]' {x_pool_j[0][:2].data}")
        # x_pool_j = F.relu(x_pool_j)
        # print(f"x.shape after GCN {x_pool_j.shape}")
        # cluster内节点特征
        # E * F
        # x_pool_j = x
        x_pool_j = x_pool_j[edge_index[1]]

        # 用胶囊网络进行聚合
        # 加权并聚合cluster内节点，如果edge_weight为None，则设置初始权重为1
        if edge_weight is None:
            edge_weight = edge_index.new_ones(edge_index.shape[1], dtype=x.dtype, device=x.device)
        # score 为每个低层节点到高层节点的隶属程度
        b_ij = edge_weight.clone().detach()
        # 动态路由
        num_routing = 3
        x_pool_j_detach = x_pool_j.detach()
        for i in range(num_routing - 1):
            # 对一个节点属于的所有cluster对softmax
            c_ij = softmax(b_ij, edge_index[1], num_nodes=N)
            # score = softmax(score, edge_index[0], num_nodes=N)
            # 加权表征
            #  (E, 1)
            # *(E, F)
            # =(E, F)
            x_pool_j_weighted = c_ij.unsqueeze(dim=-1) * x_pool_j_detach
            # 聚合cluster内节点表征
            # N * F
            cluster_representation = scatter_add(x_pool_j_weighted, edge_index[0], dim=0)
            cluster_representation = self.squash(cluster_representation)
            # cluster_representation = F.relu(cluster_representation)
            # E * F
            cluster_representation_per_edge = cluster_representation[edge_index[0]]
            # cluster 特征与node特征做注意力
            # shape of score: (E)
            # score += (cluster_representation * x_pool_j_detach).sum(dim=1)
            b_ij += (cluster_representation_per_edge * x_pool_j_detach).sum(dim=-1)
        # (E)
        c_ij = softmax(b_ij, edge_index[1], num_nodes=N)
        x_pool_j_weighted = c_ij.unsqueeze(dim=-1) * x_pool_j
        cluster_representation = scatter_add(x_pool_j_weighted, edge_index[0], dim=0)
        # ToDo 这里可以考虑加一个好的残差连接
        # cluster_representation += x
        # (N, F)
        cluster_representation = self.bn_feat_2(cluster_representation)
        cluster_representation = self.squash(cluster_representation)
        # cluster_representation = F.relu(cluster_representation)
        # print(f"cluster_representation.shape {cluster_representation.shape}")

        """计算cluster 得分"""
        # lenth of cluster_representation_vector
        # (N)
        cluster_score = cluster_representation.norm(dim=-1)
        degree = scatter_add(src=edge_weight, index=edge_index[0], dim=0)

        # (ratio*N)
        perm = topk(x=cluster_score, ratio=self.ratio, batch=batch)
        # 选择topk个节点特征
        x = cluster_representation[perm]
        batch = batch[perm]
        # edge_index: (2, E) 根据 perm 进行S_T*A*S --> （2, E')
        edge_index, edge_weight, S_index, S_value = graph_connectivity(
            device=x.device,
            perm=perm,
            edge_index=edge_index,
            edge_weight=edge_weight,
            score=c_ij,
            ratio=self.ratio,
            batch=batch,
            N=N)

        return x, edge_index, edge_weight, batch, S_index, S_value, perm

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__, self.in_channels, self.ratio)
