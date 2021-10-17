import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from LocalCapsulePooling import LocalCapsulePooling
from torch_scatter import scatter_mean, scatter_max, scatter_add, scatter_softmax
from utils import squash_1, squash_2, common_readout, readout_2, sparse_to_dense, get_loss_stability
from disentangle import LinearDisentangle
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops


class LocalCapsulePoolingNetwork(nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.5, dropout_att=0.5,
                 readout_mode='TAR',
                 dataset_name=''):
        super(LocalCapsulePoolingNetwork, self).__init__()
        self.hidden = hidden
        self.ratio = ratio
        self.dropout_att = dropout_att
        self.num_layers = num_layers
        self.readout_mode = readout_mode
        if dataset_name in ['DD', 'NCI109', 'NCI1', 'MUTAG', 'ENZYMES', 'FRANKENSTEIN', 'REDDIT-BINARY']:
            self.squash = squash_1
            from utils import squash_1 as squash
        elif dataset_name in ['PROTEINS',  'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-MULTI', 'COLLAB']:
            self.squash = squash_2
            from utils import squash_2 as squash
        else:
            print('Wrong Dataset')
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
        self.pool1 = LocalCapsulePooling(hidden, ratio, dropout_att, dataset_name=self.dataset_name)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden))
        self.convs.append(self.conv1)
        self.pools.append(self.pool1)
        for i in range(1, num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.pools.append(LocalCapsulePooling(hidden, ratio, dropout_att,  dataset_name=self.dataset_name))
            self.bns.append(nn.BatchNorm1d(hidden))
        # readout中的参数化向量
        self.task_aware_readout_linear_lst = []
        for i in range(num_layers):
            self.task_aware_readout_linear_lst.append(nn.Parameter(torch.Tensor(hidden, 1)))

        # 3*hidden due to readout layer
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
        if self.readout_mode == "TAR":
            for i in range(self.num_layers):
                # uniform(self.hidden, self.readout_atten_linear_lst[i])
                torch.nn.init.xavier_normal_(self.task_aware_readout_linear_lst[i])


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
        graph_representation = x.new_zeros(batch.max().item()+1, 3*self.hidden)
        # 池化前后分布相同
        loss_stability = x.new_zeros(1)

        # 给第一次GCN加上自环
        edge_index, edge_weight = remove_self_loops(edge_index=edge_index, edge_attr=edge_weight)
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        edge_index, edge_weight = add_remaining_self_loops(edge_index=edge_index, edge_weight=edge_weight,
                                                           fill_value=1, num_nodes=num_nodes.sum())
        # third_x is used to draw the pooled edge_index of graph
        third_x = x
        third_edge_index = edge_index
        for i in range(self.num_layers):
            x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight)
            # x = self.bns[i](x)
            x = self.squash(x)
            # x = self.squash(self.bns[i](self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight)))
            # x = F.relu(self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight))
            N = x.shape[0]
            x_l_above = x
            edge_index_l_above, edge_weight_l_above = edge_index, edge_weight
            # 池化后的graph，节点为 self.ratio * num_nodes
            # x: (new_num_nodes, hidden)
            x, edge_index, edge_weight, batch, S_index, S_value, perm = self.pools[i](x=x, edge_index=edge_index, edge_weight=edge_weight,
                                                           batch=batch)
            # # BN层
            # x = self.bns[i](x)
            # third_x is used to draw the pooled edge_index of graph
            if i == 2:
                third_x = x
                third_edge_index = edge_index
            #   SSt与XXt的F范数损失
            kN = x.shape[0]
            # 每层的权重
            temp_loss_stability = get_loss_stability(x_l_above, x)
            loss_stability = loss_stability + temp_loss_stability
            # (num_graphs, 3 * hidden)
            if self.readout_mode == "Common":
                graph_representation = graph_representation + common_readout(x, batch)
            elif self.readout_mode == "TAR":
                graph_representation = graph_representation + self.ReadoutAwareReadout(x, edge_index, edge_weight, batch, i)

        x = F.relu(self.lin1(graph_representation))
        x = F.dropout(x, p=self.dropout_att, training=self.training)
        x = self.lin2(x)
        # (num_graphs, num_classes)
        out = F.log_softmax(x, dim=-1)
        return out, loss_stability, third_x, third_edge_index

    def ReadoutAwareReadout(self, x, edge_index, edge_weight, batch, i):
        # batch个graph构成的邻接矩阵
        self.task_aware_readout_linear_lst[i] = self.task_aware_readout_linear_lst[i].to(x.device)
        # node_attens = A @ x @ self.readout_atten_linear_lst[i].to(x.device)
        # (num_nodes, 1)
        node_attens = x @ self.task_aware_readout_linear_lst[i]
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