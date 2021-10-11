import torch
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
from torch_sparse import coalesce
from torch_sparse import transpose
from torch_sparse import spspmm
from torch_scatter import scatter_mean, scatter_max, scatter_add


def readout_1(x, edge_index, edge_weight, batch):
    x_sum = scatter_add(x, batch, dim=0)
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0)
    return torch.cat((x_sum, x_mean, x_max), dim=-1)
    # return x_sum


def readout_2(x, edge_index, edge_weight, batch):
    """
    :param x: shape: (num_nodes, hidden)
    :param edge_index: shape: (num_nodes)
    :param edge_weight: shape: (num_nodes)
    :param batch:  (num_nodes)
    以节点的度数作为权重给每个节点特征加权，将节点特征的加权和作为Q
    K == V == X
    graph_representation = Q @ K_T @ V
    :return:  (num_graphs, )
    """
    # (num_nodes, 1)
    degree = scatter_add(src=edge_weight, index=edge_index[0], dim=0).unsqueeze(dim=-1)
    # degree = F.softmax(degree, dim=0)
    # (num_nodes, hidden)
    x_weighted = degree * x
    # (num_graphs, hidden)
    Q = scatter_add(x_weighted, batch, dim=0)
    # (num_nodes, hidden)
    Q = Q[batch]
    # (num_nodes, hidden)
    K = x
    # (num_nodes, 1)
    QK = (Q * K).sum(dim=-1, keepdim=True)
    # (num_graphs, hidden)
    graph_representation_weighted_sum = scatter_add(src=QK*x, index=batch, dim=0)
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0)
    return torch.cat((graph_representation_weighted_sum, x_mean, x_max), dim=-1)
    # return graph_representation_weighted_sum




def squash_1(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


def squash_2(x, dim=-1):
    eps = 1e-8
    n = x.norm(dim=dim, keepdim=True)
    return (1 - 1 / (torch.exp(n) + eps)) * (x/(n+eps))


def StAS(index_A, value_A, index_S, value_S, device, N, kN):
    r"""StAS: a function which returns new edge weights for the pooled graph using the formula S^{T}AS"""

    index_A, value_A = coalesce(index_A, value_A, m=N, n=N)
    index_S, value_S = coalesce(index_S, value_S, m=N, n=kN)
    index_B, value_B = spspmm(index_A, value_A, index_S, value_S, N, N, kN)

    index_St, value_St = transpose(index_S, value_S, N, kN)
    index_B, value_B = coalesce(index_B, value_B, m=N, n=kN)
    # index_E, value_E = spspmm(index_St.cpu(), value_St.cpu(), index_B.cpu(), value_B.cpu(), kN, N, kN)
    index_E, value_E = spspmm(index_St, value_St, index_B, value_B, kN, N, kN)

    # return index_E.to(device), value_E.to(device)
    return index_E, value_E


def graph_connectivity(device, perm, edge_index, edge_weight, score, ratio, batch, N):
    r"""graph_connectivity: is a function which internally calls StAS func to maintain graph connectivity"""
    # kN

    kN = perm.size(0)
    # (kN, 1)
    perm2 = perm.view(-1, 1)
    # mask contains bool mask of edges which originate from perm (selected) nodes
    # 根据选择出的节点，选择邻接矩阵中的出射点，将其mask成true，其余的mask成false
    mask = (edge_index[0] == perm2).sum(0, dtype=torch.bool)

    # create the S
    # 在入射点中筛选出与s1中对应的点
    S0 = edge_index[1][mask].view(1, -1)
    # 在出射点中筛选出选中的点
    S1 = edge_index[0][mask].view(1, -1)
    # 表示第一行的节点和第二行的cluster有连接
    index_S = torch.cat([S0, S1], dim=0)
    # score是每个节点隶属于cluster的得分，如果是training阶段，则50%的节点得分被设置为0，其余节点得分将被放大
    # ToDo: 这里score用了detach(),是不是说明之前的score是有梯度回传的
    value_S = score[mask].detach().squeeze()

    # relabel for pooling ie: make S [N x kN]
    n_idx = torch.zeros(N, dtype=torch.long)
    # 对于节点总表中的N个节点，选出其中的kN个，并且顺序排序，其他未选择的设置为0
    # topk个点在n_idx中按perm中的循序排序，假设从小到大，则最小得分的点为0，最大得分的点为kn
    n_idx[perm] = torch.arange(perm.size(0))
    # index_S[1] 表示选出的kn 个cluster 的出射点
    # n_idx[index_S[1]]中的元素均为所选出的出射点的perm排序
    # index_S[0]：topk个入射点
    # index_S[1], topk个出射点的perm排序
    index_S[1] = n_idx[index_S[1]]

    # create A
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].size(0))
    else:
        value_A = edge_weight.clone()

    fill_value = 1
    # ToDo 这里的新生成的A需要对角线上的元素是1
    index_E, value_E = StAS(index_A, value_A, index_S, value_S, device, N, kN)
    # index_E, value_E = remove_self_loops(edge_index=index_E, edge_attr=value_E)
    # index_E, value_E = add_remaining_self_loops(edge_index=index_E, edge_weight=value_E, fill_value=fill_value,
    #                                             num_nodes=kN)

    return index_E, value_E, index_S, value_S


def margin_loss(scores, target, loss_lambda=0.5):
    target = F.one_hot(target, scores.size(1))
    v_mag = scores

    zero = torch.zeros(1)
    zero = zero.to(device=scores.device)
    m_plus = 0.9
    m_minus = 0.1

    max_l = torch.max(m_plus - v_mag, zero)**2
    max_r = torch.max(v_mag - m_minus, zero)**2
    T_c = target

    L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
    L_c = L_c.sum(dim=1)
    L_c = L_c.mean()
    return L_c


def get_num_nodes_per_graph(batch):
    # 计算一个batch中每个graph有多少个节点
    new_ones = batch.new_ones(batch.size(0))
    return scatter_add(new_ones, batch, dim=0)


def get_num_graphs(batch):
    # 返回一个batch中有多少个graph
    return (batch.max()+1).long().item()


def transfer_c_index(c, batch):
    """
    将shape为(num_nodes, num_target_cap)的cluster assignment matrix转换为(num_nodes, num_graph * num_target_cap)，
    其中每个graph中的c互不干涉，其余部分设置为0
    :param c: (num_nodes, num_target_cap) 其中，num_target_cap为高层节点的数量
    :param batch: (num_nodes, ) (0, 0, 1, 1, 2, 2...)
    :return: (num_nodes, num_graph * num_target_cap) 形式的稀疏矩阵
    """
    num_graphs = get_num_graphs(batch)
    num_nodes = c.size(0)
    num_target_caps = c.shape[1]
    new_c = torch.zeros((num_nodes, num_graphs * num_target_caps), device=c.device)
    caps_mask_index = torch.tensor([i for i in range(num_graphs) for j in range(num_target_caps)], device=c.device)
    caps_mask = (batch.view(-1, 1) == caps_mask_index)
    for i in range(c.shape[0]):
        mask = caps_mask[i]
        new_c[i][mask] = c[i]
    return new_c


def sparse_to_dense(edge_index, edge_weight, num_nodes):
    """
    将稀疏的邻接矩阵转变为稠密类型的，可以进行A@X的操作
    :param edge_index: (2, num_edges)
    :param edge_weight: (num_edges,)
    :return:  (num_nodes, num_nodes)
    """
    A = edge_weight.new_zeros(num_nodes, num_nodes)
    row, col = edge_index[0], edge_index[1]
    A[row, col] = edge_weight
    # shape of A: (num_nodes, num_nodes)
    return A


def F_norm_loss(S_index, S_value, X, N=4, kN=4, edge_index=None, edge_weight=None):
    """计算SS_T 与 XX_T 的F范数损失"""
    S_index, S_value = coalesce(S_index, S_value, N, kN)
    St_index, St_value = transpose(S_index, S_value, m=N, n=kN)
    # SS_T
    SSt_index, SSt_value = spspmm(S_index, S_value, St_index, St_value, N, kN, N)
    SSt = sparse_to_dense(SSt_index, SSt_value, N, N).to(X.device)
    # XX_T
    XXt = (X @ X.T).to(X.device)
    # 将SS_T 从稀疏转为稠密矩阵 (2, e) -> (N, N)
    F_matrix = SSt -XXt
    loss = F_matrix.norm(p=2)/N
    return loss


def sparse_to_dense(index, value, m, n):
    # (2, e) -> (N, N)
    adj = torch.zeros(m, n, dtype=value.dtype, device=value.device)
    row, col = index[0], index[1]
    adj[row, col] = value
    return adj


def F_norm_x_loss(x_l_above, x):
    x_l_above_merge = x_l_above.T @ x_l_above
    x_merge = x.T @ x
    F_matrix = x_l_above_merge - x_merge
    loss = F_matrix.norm(p=2)
    return loss