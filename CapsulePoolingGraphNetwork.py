import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from LocalPooling import LocalPooling
from torch_scatter import scatter_mean, scatter_max, scatter_add, scatter_softmax
from utils import squash_1, squash_2, readout_1, readout_2, sparse_to_dense, F_norm_loss, F_norm_x_loss
from disentangle import LinearDisentangle
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops


class CapsulePoolingGraphNetwork(nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.5, dropout_att=0.5,
                 local_pool_mode='mode_1',
                 readout_mode='XU',
                 dataset_name=''):
        super(CapsulePoolingGraphNetwork, self).__init__()
        self.hidden = hidden
        self.ratio = ratio
        self.dropout_att = dropout_att
        self.num_layers = num_layers
        self.local_pooling_mode = local_pool_mode
        self.readout_mode = readout_mode
        # if self.readout_mode == "X":
        #     self.readout = readout_1
        # elif self.readout_mode == "XU":
        #     self.readout = self.ReadoutAttenLinear
        if dataset_name in ['DD', 'NCI109', 'NCI1', 'MUTAG', 'ENZYMES', 'FRANKENSTEIN', 'REDDIT-BINARY']:
            self.squash = squash_1
            from utils import squash_1 as squash
        elif dataset_name in ['PROTEINS',  'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-MULTI', 'COLLAB']:
            self.squash = squash_2
            from utils import squash_2 as squash
        else:
            print('Wrong Dataset')
        # print(f"self.squash {self.squash}")
        self.dataset_name = dataset_name
        # Todo 这里需要判断是否需要参数化形式


        self.bn_disen_0 = nn.BatchNorm1d(dataset.num_features)
        self.disentangle_num = 4
        self.disen = torch.nn.ModuleList()
        for i in range(self.disentangle_num):
            self.disen.append(LinearDisentangle(dataset.num_features, hidden // self.disentangle_num))
        self.bn_disen_1 = nn.BatchNorm1d(hidden)
        # Local Pooling
        # self.conv1 = GCNConv(hidden, hidden)
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.pool1 = LocalPooling(hidden, ratio, dropout_att, local_pool_mode=self.local_pooling_mode, dataset_name=self.dataset_name)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden))
        self.convs.append(self.conv1)
        self.pools.append(self.pool1)
        for i in range(1, num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.pools.append(LocalPooling(hidden, ratio, dropout_att, local_pool_mode=self.local_pooling_mode, dataset_name=self.dataset_name))
            self.bns.append(nn.BatchNorm1d(hidden))
        # readout中的参数化向量
        self.readout_atten_linear_lst = []
        for i in range(num_layers):
            self.readout_atten_linear_lst.append(nn.Parameter(torch.Tensor(hidden, 1)))

        # self.lin1 = Linear(1 * hidden, hidden)  # 3*hidden due to readout layer
        self.lin1 = Linear(3*hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.bn_disen_0.reset_parameters()
        self.bn_disen_1.reset_parameters()
        for conv, pool, bn in zip(self.convs, self.pools, self.bns):
            conv.reset_parameters()
            pool.reset_parameters()
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for i in range(self.num_layers):
            # uniform(self.hidden, self.readout_atten_linear_lst[i])
            torch.nn.init.xavier_normal_(self.readout_atten_linear_lst[i])


    def forward(self, data):
        """
        :input:  (DataLoader)
        :return: (batch, probablilties)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 初次创建edge_weight
        edge_weight = x.new_ones(edge_index[0].size(0), dtype=x.dtype, device=x.device)
        x = self.bn_disen_0(x)
        # disentangle
        # out = []
        # for i, disen in enumerate(self.disen):
        #     temp = F.relu(disen(x))
        #     temp = F.dropout(temp, p=self.dropout_att, training=self.training)
        #     out.append(temp)
        # x = torch.cat(out, dim=-1)
        #
        # batch_normalization
        # x = self.bn_disen_1(x)
        # 两种不同的readout机制：1、拼接所有层的结果， 2、加和所有层的结果
        graph_representation = x.new_zeros(batch.max().item()+1, 3*self.hidden)
        # 两种辅助训练损失：1、S与X的自分布相同 2、XtX各层之间每个维度多节点之间的分布相同
        SSt_XXt_loss = x.new_zeros(1)
        x_loss = x.new_zeros(1)

        # 给第一次GCN加上自环
        edge_index, edge_weight = remove_self_loops(edge_index=edge_index, edge_attr=edge_weight)
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        edge_index, edge_weight = add_remaining_self_loops(edge_index=edge_index, edge_weight=edge_weight,
                                                           fill_value=1, num_nodes=num_nodes.sum())
        pooled_x = x
        pooled_edge_index = edge_index
        for i in range(self.num_layers):
            if i == 0:
                print(f"x.shape {x.shape}")
                print(f"self.convs[0] {self.convs[0]}")
            x = self.squash(self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight))
            # x = F.relu(self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight))
            N = x.shape[0]
            x_l_above = x
            edge_index_l_above, edge_weight_l_above = edge_index, edge_weight
            # 池化后的graph，节点为 self.ratio * num_nodes
            # x: (new_num_nodes, hidden)
            x, edge_index, edge_weight, batch, S_index, S_value, perm = self.pools[i](x=x, edge_index=edge_index, edge_weight=edge_weight,
                                                           batch=batch)
            if i == 1:
                pooled_x = x
                pooled_edge_index = edge_index
            third_x = x
            third_edge_index = edge_index
            # BN层
            # x = self.bns[i](x)
            #   SSt与XXt的F范数损失
            kN = x.shape[0]
            temp_SSt_XXt_loss = F_norm_loss(S_index=S_index, S_value=S_value, X=x_l_above, N=N, kN=kN,
                                            edge_index=edge_index, edge_weight=edge_weight)
            SSt_XXt_loss = SSt_XXt_loss + temp_SSt_XXt_loss
            # 每层的权重
            temp_norm_x_loss = F_norm_x_loss(x_l_above, x)
            x_loss = x_loss + temp_norm_x_loss
            #   AXu作为权重 -->(num_nodes, 1)
            #   加权聚合 --> (num_graphs, hidden)
            # graph_representation = graph_representation + readout_1(x, batch)
            # (num_graphs, 3 * hidden)
            graph_representation = graph_representation + self.ReadoutAttenLinear(x, edge_index, edge_weight, batch, i)
        x = F.relu(self.lin1(graph_representation))
        x = F.dropout(x, p=self.dropout_att, training=self.training)
        x = self.lin2(x)
        # (num_graphs, num_classes)
        out = F.log_softmax(x, dim=-1)
        # return out, SSt_XXt_loss
        return out, x_loss, third_x, third_edge_index
        # return out, x_loss, pooled_x, pooled_edge_index

    def ReadoutAttenLinear(self, x, edge_index, edge_weight, batch, i):
        # batch个graph构成的邻接矩阵
        # A = sparse_to_dense(edge_index, edge_weight, m=x.shape[0], n=x.shape[0]).to(x.device)
        #   Xu
        #   (num_nodes, hidden) @ (hidden, 1)
        #   ->(num_nodes, 1)
        self.readout_atten_linear_lst[i] = self.readout_atten_linear_lst[i].to(x.device)
        # node_attens = A @ x @ self.readout_atten_linear_lst[i].to(x.device)
        # (num_nodes, 1)
        node_attens = x @ self.readout_atten_linear_lst[i].to(x.device)
        # 加权求和
        # (num_nodes, hidden)
        nodes_wegihted = node_attens * x
        x_weighted_sum = scatter_add(src=nodes_wegihted, index=batch, dim=0)
        x_mean = scatter_mean(x, batch, dim=0)
        x_max, _ = scatter_max(x, batch, dim=0)
        # (num_graphs, 3 * hidden)
        return torch.cat((x_weighted_sum, x_mean, x_max), dim=-1)

    def __repr__(self):
        return self.__class__.__name__