import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
from torch_geometric.nn import GCNConv, DenseGCNConv, dense_diff_pool, TopKPooling, SAGPooling, GINConv
# from ASAP import ASAPooling
from DenseGCNConv import DenseGCNConv as DGCNConv
import torch.nn.functional as F
from utils import dense_readout, readout_1
from torch_scatter import scatter_mean, scatter_max
from ASAP.asap_pool import ASAP_Pooling



# Reimplementation for setting Frobenius Norm as L2-Norm (For CUDA Compatibility)
def dense_mincut_pool(x, adj, s, mask=None):
    EPS = 1e-15
    r"""MinCUt pooling operator from the `"Mincut Pooling in Graph Neural
    Networks" <https://arxiv.org/abs/1907.00481>`_ paper
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    # out_adj => Very Very dense adjacency Matrix...
    # MinCUT regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s)) + EPS
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / (torch.norm(ss, dim=(-1, -2), keepdim=True, p=2) + EPS) -
        i_s / torch.norm(i_s), dim=(-1, -2), p=2)
    ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.

    ######### NORMALIZE TERM ##################

    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss


def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out


class MinCutPool(torch.nn.Module):
    def __init__(self, dataset, hidden=128, ratio=0.5, dropout_att=0.5):
        super(MinCutPool, self).__init__()
        # self.args = args
        # self.num_features = args.num_features
        # self.nhid = args.num_hidden
        # self.ln = args.ln
        self.dataset = dataset
        self.hidden = hidden
        self.ratio = ratio
        self.dropout_att = dropout_att
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset[0].x.shape[0]
        # Encoder
        self.conv_1 = DGCNConv(self.num_features, self.hidden)
        self.conv_2 = DGCNConv(self.hidden, self.hidden)

        # Pooling
        pool_out_dim_1 = int(self.num_nodes * self.ratio)
        self.pool_1 = nn.Linear(self.hidden, pool_out_dim_1)

        self.conv_3 = DGCNConv(self.hidden, self.hidden)
        pool_out_dim_2 = int(pool_out_dim_1*self.ratio)
        self.pool_2 = nn.Linear(self.hidden, pool_out_dim_2)

        self.conv_4 = DGCNConv(self.hidden, self.hidden)
        pool_out_dim_3 = int(pool_out_dim_2 * self.ratio)
        self.pool_3 = nn.Linear(self.hidden, pool_out_dim_3)

        self.lin_1 = nn.Linear(3 * hidden, hidden)
        self.lin_2 = nn.Linear(hidden, dataset.num_classes)
        # # Decoder
        # self.conv3 = GCNConv(self.nhid, self.nhid)
        # self.conv4 = GCNConv(self.nhid, self.nhid)
        # self.conv5 = GCNConv(self.nhid, self.num_features)

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.conv_3.reset_parameters()
        self.conv_4.reset_parameters()
        self.pool_1.reset_parameters()
        self.pool_2.reset_parameters()
        self.pool_3.reset_parameters()
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

    def forward(self, data):
        mc_loss = 0
        o_loss = 0
        x, adj, mask = data.x, data.adj, data.mask
        graph_representation = x.new_zeros(x.shape[0], 3 * self.hidden)
        x = F.tanh(self.conv_1(x, adj))
        x = F.tanh(self.conv_2(x, adj))
        # x, mask = to_dense_batch(x, batch)

        # adj = to_dense_adj(edge_index, batch)
        # print(f"x: {x.shape}")
        # print(f"adj: {adj.shape}")
        # print(f"self.pool_1 {self.pool_1}")
        # print(f"mask.shape {mask.shape}")
        s = self.pool_1(x)
        # print(f"s.shape {s.shape}")
        x, adj, temp_mc_loss, temp_o_loss = dense_mincut_pool(x, adj, s, mask)
        graph_representation = graph_representation + dense_readout(x)
        mc_loss += temp_o_loss
        o_loss += temp_o_loss
        # print(f"x.shape {x.shape}")
        x = F.tanh(self.conv_3(x, adj))
        # print(f"x.shape {x.shape}")
        # print(f"self.pool_2 {self.pool_2}")
        s = self.pool_2(x)
        # print(f"---------------------------")
        # print(f"x: {x.shape}")
        # print(f"adj: {adj.shape}")
        # print(f"self.s {s.shape}")

        x, adj, temp_mc_loss, temp_o_loss = dense_mincut_pool(x, adj, s)
        pooled_x = x
        pooled_edge_index = adj
        graph_representation = graph_representation + dense_readout(x)
        mc_loss += temp_o_loss
        o_loss += temp_o_loss
        x = F.tanh(self.conv_4(x, adj))
        s = self.pool_3(x)
        x, adj, temp_mc_loss, temp_o_loss = dense_mincut_pool(x, adj, s)
        third_x = x
        third_edge_index = adj
        graph_representation = graph_representation + dense_readout(x)
        mc_loss += temp_o_loss
        o_loss += temp_o_loss
        # (num_graph, num_classes)
        x = F.relu(self.lin_1(graph_representation))
        x = F.dropout(x, p=self.dropout_att, training=self.training)
        x = self.lin_2(x)
        # (num_graphs, num_classes)
        out = F.log_softmax(x, dim=-1)
        return out, mc_loss+o_loss, third_x, third_edge_index
        # return out, mc_loss + o_loss, pooled_x, pooled_edge_index



class DiffPool(torch.nn.Module):
    def __init__(self, dataset, hidden=128, ratio=0.5, dropout_att=0.5):
        super(DiffPool, self).__init__()

        self.dataset = dataset
        self.hidden = hidden
        self.ratio = ratio
        self.dropout_att = dropout_att
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset[0].x.shape[0]
        # Encoder
        self.conv_1 = DGCNConv(self.num_features, self.hidden)
        self.conv_2 = DGCNConv(self.hidden, self.hidden)

        # Pooling
        pool_out_dim_1 = int(self.num_nodes * self.ratio)
        self.pool_1 = nn.Linear(self.hidden, pool_out_dim_1)

        self.conv_3 = DGCNConv(self.hidden, self.hidden)
        pool_out_dim_2 = int(pool_out_dim_1 * self.ratio)
        self.pool_2 = nn.Linear(self.hidden, pool_out_dim_2)

        self.conv_4 = DGCNConv(self.hidden, self.hidden)
        pool_out_dim_3 = int(pool_out_dim_2 * self.ratio)
        self.pool_3 = nn.Linear(self.hidden, pool_out_dim_3)

        self.lin_1 = nn.Linear(3 * hidden, hidden)
        self.lin_2 = nn.Linear(hidden, dataset.num_classes)
        # # Decoder
        # self.conv3 = GCNConv(self.nhid, self.nhid)
        # self.conv4 = GCNConv(self.nhid, self.nhid)
        # self.conv5 = GCNConv(self.nhid, self.num_features)

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.conv_3.reset_parameters()
        self.conv_4.reset_parameters()
        self.pool_1.reset_parameters()
        self.pool_2.reset_parameters()
        self.pool_3.reset_parameters()
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

    def forward(self, data):
        l1 = 0
        e1 = 0
        x, adj, mask = data.x, data.adj, data.mask
        graph_representation = x.new_zeros(x.shape[0], 3 * self.hidden)
        x = F.tanh(self.conv_1(x, adj))
        x = F.tanh(self.conv_2(x, adj))
        # x, mask = to_dense_batch(x, batch)

        # adj = to_dense_adj(edge_index, batch)
        # print(f"x: {x.shape}")
        # print(f"adj: {adj.shape}")
        # print(f"self.pool_1 {self.pool_1}")
        # print(f"mask.shape {mask.shape}")
        s = self.pool_1(x)
        # print(f"s.shape {s.shape}")
        x, adj, temp_l1, temp_e1 = dense_diff_pool(x, adj, s, mask)
        graph_representation = graph_representation + dense_readout(x)
        l1 += temp_l1
        e1 += temp_e1
        # print(f"x.shape {x.shape}")
        x = F.tanh(self.conv_3(x, adj))
        # print(f"x.shape {x.shape}")
        # print(f"self.pool_2 {self.pool_2}")
        s = self.pool_2(x)
        # print(f"---------------------------")
        # print(f"x: {x.shape}")
        # print(f"adj: {adj.shape}")
        # print(f"self.s {s.shape}")

        x, adj, temp_l1, temp_e1 = dense_diff_pool(x, adj, s)

        pooled_x = x
        pooled_edge_index = adj
        graph_representation = graph_representation + dense_readout(x)
        l1 += temp_l1
        e1 += temp_e1

        x = F.tanh(self.conv_4(x, adj))
        s = self.pool_3(x)
        x, adj, temp_l1, temp_e1 = dense_diff_pool(x, adj, s)
        third_x = x
        third_edge_index = adj
        graph_representation = graph_representation + dense_readout(x)
        l1 += temp_l1
        e1 += temp_e1

        # (num_graph, num_classes)
        x = F.relu(self.lin_1(graph_representation))
        x = F.dropout(x, p=self.dropout_att, training=self.training)
        x = self.lin_2(x)
        # (num_graphs, num_classes)
        out = F.log_softmax(x, dim=-1)
        return out, l1+e1, third_x, third_edge_index
        # return out, l1 + e1, pooled_x, pooled_edge_index


class TopKPool(torch.nn.Module):
    def __init__(self, dataset, hidden=128, ratio=0.5, dropout_att=0.5):
        super(TopKPool, self).__init__()
        self.dataset = dataset
        self.hidden = hidden
        self.ratio = ratio
        self.dropout_att = dropout_att
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset[0].x.shape[0]

        # Encoder
        self.conv_1 = GCNConv(self.num_features, self.hidden)
        self.conv_2 = GCNConv(self.hidden, self.hidden)
        self.conv_3 = GCNConv(self.hidden, self.hidden)
        self.conv_4 = GCNConv(self.hidden, self.hidden)

        # Pooling
        self.pool_1 = TopKPooling(self.hidden, self.ratio)
        self.pool_2 = TopKPooling(self.hidden, self.ratio)
        self.pool_3 = TopKPooling(self.hidden, self.ratio)

        self.lin_1 = nn.Linear(3 * hidden, hidden)
        self.lin_2 = nn.Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.conv_3.reset_parameters()
        self.conv_4.reset_parameters()
        self.pool_1.reset_parameters()
        self.pool_2.reset_parameters()
        self.pool_3.reset_parameters()
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_representation = x.new_zeros(batch.max().item() + 1, 3 * self.hidden)
        x = F.tanh(self.conv_1(x, edge_index))
        x = F.tanh(self.conv_2(x, edge_index))
        # Pooling
        res = x
        x, edge_index, _, batch, perm, _ = self.pool_1(x, edge_index, None, batch)
        graph_representation = graph_representation + readout_1(x, batch)

        x = F.tanh(self.conv_3(x, edge_index))
        x, edge_index, _, batch, perm, _ = self.pool_2(x, edge_index, None, batch)
        pooled_x = x
        pooled_edge_index = edge_index
        graph_representation = graph_representation + readout_1(x, batch)

        x = F.tanh(self.conv_4(x, edge_index))
        x, edge_index, _, batch, perm, _ = self.pool_3(x, edge_index, None, batch)
        third_x = x
        third_edge_index = edge_index
        graph_representation = graph_representation + readout_1(x, batch)

        x = F.relu(self.lin_1(graph_representation))
        x = F.dropout(x, p=self.dropout_att, training=self.training)
        x = self.lin_2(x)
        # (num_graphs, num_classes)
        out = F.log_softmax(x, dim=-1)
        return out, 0, third_x, third_edge_index
        # return out, 0, pooled_x, pooled_edge_index


class SAGPool(torch.nn.Module):
    def __init__(self, dataset, hidden=128, ratio=0.5, dropout_att=0.5):
        super(SAGPool, self).__init__()
        self.dataset = dataset
        self.hidden = hidden
        self.ratio = ratio
        self.dropout_att = dropout_att
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset[0].x.shape[0]


        # Encoder
        self.conv_1 = GCNConv(self.num_features, self.hidden)
        self.conv_2 = GCNConv(self.hidden, self.hidden)
        self.conv_3 = GCNConv(self.hidden, self.hidden)
        self.conv_4 = GCNConv(self.hidden, self.hidden)

        # Pooling
        self.pool_1 = SAGPooling(self.hidden, self.ratio)
        self.pool_2 = SAGPooling(self.hidden, self.ratio)
        self.pool_3 = SAGPooling(self.hidden, self.ratio)


        self.lin_1 = nn.Linear(3 * hidden, hidden)
        self.lin_2 = nn.Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.conv_3.reset_parameters()
        self.conv_4.reset_parameters()
        self.pool_1.reset_parameters()
        self.pool_2.reset_parameters()
        self.pool_3.reset_parameters()
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_representation = x.new_zeros(batch.max().item()+1, 3 * self.hidden)
        x = F.tanh(self.conv_1(x, edge_index))
        x = F.tanh(self.conv_2(x, edge_index))
        # Pooling
        res = x
        x, edge_index, _, batch, perm, _ = self.pool_1(x, edge_index, None, batch)
        graph_representation = graph_representation + readout_1(x, batch)

        x = F.tanh(self.conv_3(x, edge_index))
        x, edge_index, _, batch, perm, _ = self.pool_2(x, edge_index, None, batch)
        pooled_x = x
        pooled_edge_index = edge_index
        graph_representation = graph_representation + readout_1(x, batch)

        x = F.tanh(self.conv_4(x, edge_index))
        x, edge_index, _, batch, perm, _ = self.pool_3(x, edge_index, None, batch)
        third_x = x
        third_edge_index = edge_index
        graph_representation = graph_representation + readout_1(x, batch)

        x = F.relu(self.lin_1(graph_representation))
        x = F.dropout(x, p=self.dropout_att, training=self.training)
        x = self.lin_2(x)
        # (num_graphs, num_classes)
        out = F.log_softmax(x, dim=-1)
        return out, 0, third_x, third_edge_index
        # return out, 0, pooled_x, pooled_edge_index


class SAGPoolG(torch.nn.Module):
    def __init__(self, dataset, hidden=128, ratio=0.5, dropout_att=0.5):
        super(SAGPoolG, self).__init__()
        self.dataset = dataset
        self.hidden = hidden
        self.ratio = ratio
        self.dropout_att = dropout_att
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset[0].x.shape[0]


        # Encoder
        self.conv_1 = GCNConv(self.num_features, self.hidden)
        self.conv_2 = GCNConv(self.hidden, self.hidden)
        self.conv_3 = GCNConv(self.hidden, self.hidden)
        self.conv_4 = GCNConv(self.hidden, self.hidden)

        # Pooling
        self.pool_1 = SAGPooling(self.hidden, self.ratio)



        self.lin_1 = nn.Linear(3*hidden, hidden)
        self.lin_2 = nn.Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.conv_3.reset_parameters()
        self.conv_4.reset_parameters()
        self.pool_1.reset_parameters()
        # self.pool_2.reset_parameters()
        # self.pool_3.reset_parameters()
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_representation = x.new_zeros(x.shape[0], self.hidden)
        x = F.tanh(self.conv_1(x, edge_index))
        x = F.tanh(self.conv_2(x, edge_index))
        # Pooling
        res = x
        # x, edge_index, _, batch, perm, _ = self.pool_1(x, edge_index, None, batch)
        graph_representation = graph_representation + x

        x = F.tanh(self.conv_3(x, edge_index))
        # x, edge_index, _, batch, perm, _ = self.pool_2(x, edge_index, None, batch)
        pooled_x = x
        pooled_edge_index = edge_index
        graph_representation = graph_representation + x

        x = F.tanh(self.conv_4(x, edge_index))
        graph_representation = graph_representation + x

        x, edge_index, _, batch, perm, _ = self.pool_1(graph_representation, edge_index, None, batch)
        third_x = x
        third_edge_index = edge_index
        # print(f"x.shape {x.shape}")
        # print(f"batch.shape {batch.shape}")
        graph_representation = readout_1(x, batch)
        x = F.relu(self.lin_1(graph_representation))
        x = F.dropout(x, p=self.dropout_att, training=self.training)
        x = self.lin_2(x)
        # (num_graphs, num_classes)
        out = F.log_softmax(x, dim=-1)
        return out, 0, third_x, third_edge_index
        # return out, 0, pooled_x, pooled_edge_index



class ASAP_Pool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, **kwargs):
        super(ASAP_Pool, self).__init__()
        if type(ratio)!=list:
            ratio = [ratio for i in range(num_layers)]
        self.num_layers = num_layers
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.pool1 = ASAP_Pooling(in_channels=hidden, ratio=ratio[0], **kwargs)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.pools.append(ASAP_Pooling(in_channels=hidden, ratio=ratio[i], **kwargs))
        self.lin1 = Linear(3*hidden, hidden) # 2*hidden due to readout layer
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.pool1.reset_parameters()
        for conv, pool in zip(self.convs, self.pools):
            conv.reset_parameters()
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_weight, batch, perm = self.pool1(x=x, edge_index=edge_index, edge_weight=None, batch=batch)
        xs = readout_1(x, batch)
        # for conv, pool in zip(self.convs, self.pools):
        #     x = F.relu(conv(x=x, edge_index=edge_index, edge_weight=edge_weight))
        #     x, edge_index, edge_weight, batch, perm = pool(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        #     xs += readout_1(x, batch)
        pooled_x = 0
        pooled_edge_index = 0
        for i in range(self.num_layers-1):
            x = F.relu(self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight))
            x, edge_index, edge_weight, batch, perm = self.pools[i](x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
            xs += readout_1(x, batch)
            if i == 0:
                pooled_x = x
                pooled_edge_index = edge_index
        third_x = x
        third_edge_index = edge_index
        x = F.relu(self.lin1(xs))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        out = F.log_softmax(x, dim=-1)
        return out, 0, third_x, third_edge_index
        # return out, 0, pooled_x, pooled_edge_index


class GIN(nn.Module):
    def __init__(self, dataset, hidden=128, dropout_att=0.5):
        super(GIN, self).__init__()
        self.hidden = hidden
        self.dropout_att = dropout_att
        seq_1 = nn.Sequential(nn.Linear(dataset.num_node_features, hidden),
                              nn.ReLU(),
                              nn.Linear(hidden, hidden))
        self.conv_1 = GINConv(seq_1)
        seq_2 = nn.Sequential(nn.Linear(hidden, hidden),
                              nn.ReLU(),
                              nn.Linear(hidden, hidden))
        self.conv_2 = GINConv(seq_2)
        self.conv_3 = GINConv(seq_2)
        self.conv_4 = GINConv(seq_2)
        self.lin_1 = nn.Linear(3 * hidden, hidden)
        self.lin_2 = nn.Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.conv_3.reset_parameters()
        self.conv_4.reset_parameters()
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_representation = x.new_zeros(batch.max().item() + 1, 3 * self.hidden)
        x = F.relu(self.conv_1(x, edge_index))
        x = F.relu(self.conv_2(x, edge_index))
        graph_representation += readout_1(x, batch)
        x = F.relu(self.conv_3(x, edge_index))
        graph_representation += readout_1(x, batch)
        x = F.relu(self.conv_4(x, edge_index))
        graph_representation += readout_1(x, batch)
        x = F.relu(self.lin_1(graph_representation))
        x = F.dropout(x, p=self.dropout_att, training=self.training)
        x = self.lin_2(x)
        # (num_graphs, num_classes)
        out = F.log_softmax(x, dim=-1)
        # out, aux_loss, high_level_x, high_level_edge_index
        return out, 0, None, None

class GCN(nn.Module):
    def __init__(self, dataset, hidden=128, dropout_att=0.5):
        super(GCN, self).__init__()
        self.hidden = hidden
        self.dropout_att = dropout_att
        self.conv_1 = GCNConv(dataset.num_node_features, hidden)
        self.conv_2 = GCNConv(hidden, hidden)
        self.conv_3 = GCNConv(hidden, hidden)
        self.conv_4 = GCNConv(hidden, hidden)
        self.lin_1 = nn.Linear(3 * hidden, hidden)
        self.lin_2 = nn.Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.conv_3.reset_parameters()
        self.conv_4.reset_parameters()
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_representation = x.new_zeros(batch.max().item() + 1, 3 * self.hidden)
        x = F.relu(self.conv_1(x, edge_index))
        # x = F.relu(self.conv_2(x, edge_index))
        graph_representation += readout_1(x, batch)
        x = F.relu(self.conv_3(x, edge_index))
        graph_representation += readout_1(x, batch)
        x = F.relu(self.conv_4(x, edge_index))
        graph_representation += readout_1(x, batch)
        x = F.relu(self.lin_1(graph_representation))
        x = F.dropout(x, p=self.dropout_att, training=self.training)
        x = self.lin_2(x)
        # (num_graphs, num_classes)
        out = F.log_softmax(x, dim=-1)
        # out, aux_loss, high_level_x, high_level_edge_index
        return out, 0, None, None