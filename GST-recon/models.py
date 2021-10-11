# MODEL DEFINITION

import torch
import torch.nn as nn

from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
from torch_geometric.nn import GCNConv, DenseGCNConv, dense_diff_pool, TopKPooling, SAGPooling
# from ASAP import ASAPooling
from asap_pool import ASAP_Pooling
from CapsulePool import CapsulePooling
import torch.nn.functional as F

from layers import P

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

class GST_AE(torch.nn.Module):
    def __init__(self, args, n_nodes):
        super(GST_AE, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.num_hidden
        self.ln = args.ln

        # Encoder
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        # Pooling
        self.pool = P(self.nhid, self.nhid, self.nhid, 1, int(n_nodes * self.args.ratio), ln=self.ln, args=self.args)

        # Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)

    def forward(self, nodes, edges, batch):
        x, edge_index = nodes, edges

        x = F.tanh(self.conv1(x, edge_index))
        x = F.tanh(self.conv2(x, edge_index))

        # Pooling
        x, attn = self.pool(x, batch, edge_index)
        k = x.shape[1]

        # Pooling Adjacency
        dense_adj = to_dense_adj(edge_index)
        # 225 x 225
        pool_adj = torch.bmm(torch.bmm(attn, dense_adj), attn.transpose(1, 2))
        # ind = torch.arange(k, device=pool_adj.device)
        # pool_adj[:, ind, ind] = 0
        # d = torch.einsum('ijk->ij', pool_adj)
        # d = torch.sqrt(d)[:, None] + 1e-15
        # pool_adj = (pool_adj / d) / d.transpose(1, 2)

        # Upsampling
        # 900 x 32
        x_out = torch.bmm(attn.transpose(1, 2), x)
        x_out = x_out.squeeze(0)
        # 900 x 900
        adj_out = torch.bmm(attn.transpose(1, 2), torch.bmm(pool_adj, attn))
        
        if self.args.recon_adj:

            edge_index = dense_to_sparse(adj_out.squeeze(0))[0]

        x = F.tanh(self.conv3(x_out, edge_index))
        x = F.tanh(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)

        return x, adj_out, 0, 0

class MINCUTPOOL_AE(torch.nn.Module):
    def __init__(self, args, n_nodes):
        super(MINCUTPOOL_AE, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.num_hidden
        self.ln = args.ln

        # Encoder
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        # Pooling
        self.pool = nn.Linear(self.nhid, int(n_nodes * self.args.ratio))
        
        # Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)

    def forward(self, nodes, edges, batch):
        x, edge_index = nodes, edges

        x = F.tanh(self.conv1(x, edge_index))
        x = F.tanh(self.conv2(x, edge_index))
        
        # Pooling
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index)

        s = self.pool(x)
        x, adj, mc_loss, o_loss = dense_mincut_pool(x, adj, s, mask)

        # Upsampling
        # 900 x 32
        s = torch.softmax(s, dim=-1)
        x_out = torch.bmm(s, x)
        x_out = x_out.squeeze(0)
        # 900 x 900
        adj_out = torch.bmm(s, torch.bmm(adj, s.transpose(1, 2)))
        
        if self.args.recon_adj:

            edge_index = dense_to_sparse(adj_out.squeeze(0))[0]

        x = F.tanh(self.conv3(x_out, edge_index))
        x = F.tanh(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)

        return x, adj_out, mc_loss, o_loss


class CapsulePool_AE(torch.nn.Module):
    def __init__(self, args, n_nodes):
        super(CapsulePool_AE, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.num_hidden
        self.ln = args.ln

        # Encoder
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        # Pooling
        # self.pool = nn.Linear(self.nhid, n_nodes // 4)
        self.pool = CapsulePooling(hidden=self.nhid, ratio=self.args.ratio, )
        # Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)

    def forward(self, nodes, edges, batch):
        x, edge_index = nodes, edges

        x = F.tanh(self.conv1(x, edge_index))
        x = F.tanh(self.conv2(x, edge_index))

        # Pooling
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index)

        # x, adj, mc_loss, o_loss = dense_mincut_pool(x, adj, s, mask)

        x, adj, batch, perm, s, x_loss = self.pool(x, edge_index, batch=batch)

        # Upsampling
        # 900 x 32
        # s = torch.softmax(s, dim=-1)
        # x = x.unsqueeze(dim=0)
        x = x.unsqueeze(dim=0)
        adj = adj.unsqueeze(dim=0)
        s = s.unsqueeze(dim=0)
        x_out = torch.bmm(s, x)
        x_out = x_out.squeeze(0)
        # 900 x 900
        adj_out = torch.bmm(s, torch.bmm(adj, s.transpose(1, 2)))

        if self.args.recon_adj:
            edge_index = dense_to_sparse(adj_out.squeeze(0))[0]

        x = F.tanh(self.conv3(x_out, edge_index))
        x = F.tanh(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)

        return x, adj_out, x_loss*0.00001, 0


class DIFFPOOL_AE(torch.nn.Module):
    def __init__(self, args, n_nodes):
        super(DIFFPOOL_AE, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.num_hidden
        self.ln = args.ln

        # Encoder
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        # Pooling
        self.pool = DenseGCNConv(self.nhid, int(n_nodes * self.args.ratio))
        # self.pool = DenseGCNConv(self.nhid, n_nodes //4)

        # Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)

    def forward(self, nodes, edges, batch):
        x, edge_index = nodes, edges

        x = F.tanh(self.conv1(x, edge_index))
        x = F.tanh(self.conv2(x, edge_index))

        # Pooling
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index)

        s = self.pool(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        # Upsampling
        # 900 x 32
        s = torch.softmax(s, dim=-1)
        x_out = torch.bmm(s, x)
        x_out = x_out.squeeze(0)
        # 900 x 900
        adj_out = torch.bmm(s, torch.bmm(adj, s.transpose(1, 2)))

        if self.args.recon_adj:

            edge_index = dense_to_sparse(adj_out.squeeze(0))[0]

        x = F.tanh(self.conv3(x_out, edge_index))
        x = F.tanh(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)

        return x, adj_out, 0.1 * l1, 0.1 * e1


class ASAP_AE(torch.nn.Module):
    def __init__(self, args, n_nodes):
        super(ASAP_AE, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.num_hidden
        self.ln = args.ln

        # Encoder
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)


        # Pooling
        self.pool = ASAP_Pooling(in_channels=self.nhid,
                               ratio=self.args.ratio,
                               )

        # Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)

    def forward(self, nodes, edges, batch):
        x, edge_index = nodes, edges

        x = F.tanh(self.conv1(x, edge_index))
        x = F.tanh(self.conv2(x, edge_index))

        # Pooling
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index)

        # s = self.pool(x, adj, mask)

        # x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        # edge_weight = x.new_ones(edge_index.shape)
        x, adj, batch, perm, s = self.pool(x, edge_index, batch=batch)


        # Upsampling
        # 900 x 32
        # s = torch.softmax(s, dim=-1)
        x = x.unsqueeze(dim=0)
        s = s.unsqueeze(dim=0)
        adj = adj.unsqueeze(dim=0)
        # print(f"x.shape {x.shape}   s.shape {s.shape}")
        x_out = torch.bmm(s, x)
        # (900, 32)
        x_out = x_out.squeeze(0)
        # 900 x 900
        # print(f"adj.shape {adj.shape}")
        # print(f"s.transpose(1, 2) {s.transpose(1, 2).shape}")
        adj_out = torch.bmm(s, torch.bmm(adj, s.transpose(1, 2)))

        if self.args.recon_adj:
            edge_index = dense_to_sparse(adj_out.squeeze(0))[0]

        x = F.tanh(self.conv3(x_out, edge_index))
        x = F.tanh(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)
        # print(f"x.shape {x.shape}")
        # print(f"adj_out.shape {adj_out.shape}")
        return x, adj_out, 0, 0

class TOPKPOOL_AE(torch.nn.Module):
    def __init__(self, args, n_nodes):
        super(TOPKPOOL_AE, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.num_hidden
        self.ln = args.ln

        # Encoder
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        # Pooling
        self.pool = TopKPooling(self.nhid, self.args.ratio)
        
        # Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)

    def forward(self, nodes, edges, batch):
        x, edge_index = nodes, edges

        x = F.tanh(self.conv1(x, edge_index))
        x = F.tanh(self.conv2(x, edge_index))
        
        # Pooling
        res = x
        x, edge_index, _, batch, perm, _ = self.pool(x, edge_index, None, batch)

        # Upsampling
        # 900 x 32
        x_out = torch.zeros_like(res)
        x_out[perm] = x

        # 900 x 900
        x = F.tanh(self.conv3(x_out, edge_index))
        x = F.tanh(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)

        return x, edge_index, 0, 0


class SAGPOOL_AE(torch.nn.Module):
    def __init__(self, args, n_nodes):
        super(SAGPOOL_AE, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.num_hidden
        self.ln = args.ln

        # Encoder
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        # Pooling
        self.pool = SAGPooling(self.nhid, self.args.ratio)

        # Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)

    def forward(self, nodes, edges, batch):
        x, edge_index = nodes, edges

        x = F.tanh(self.conv1(x, edge_index))
        x = F.tanh(self.conv2(x, edge_index))

        # Pooling
        res = x
        x, edge_index, _, batch, perm, _ = self.pool(x, edge_index, None, batch)

        # Upsampling
        # 900 x 32
        x_out = torch.zeros_like(res)
        x_out[perm] = x

        # 900 x 900
        x = F.tanh(self.conv3(x_out, edge_index))
        x = F.tanh(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)

        return x, edge_index, 0, 0