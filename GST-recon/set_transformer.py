import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.utils import to_dense_batch

def attention_visualize(A):
    import seaborn as sns
    import matplotlib.pyplot as plt
    A = A.squeeze(0).cpu().numpy()
    # ax = sns.heatmap(A, vmax=1.0, linewidths=.5)
    ax = sns.heatmap(A)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig("./results/clustering_visualization.jpg")
    plt.close()
    print("Save Done")
    return

def query_key_visualize(Q, K):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    Q = Q.squeeze(0).cpu().numpy()
    K = K.squeeze(0).cpu().numpy()
    pca = PCA(n_components=2)
    X = np.concatenate([Q, K])
    X = pca.fit_transform(X)
    labels = [0 for _ in range(len(Q))] + [1 for _ in range(len(K))]
    colors = ['red' if label == 0 else 'blue' for label in labels]
    plt.scatter(X[:, 0], X[:, 1], color=colors)
    plt.savefig("./results/query_key_visualization.jpg")
    plt.close()
    print("Save Done")
    return


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, attention_mask=None, graph=None, return_attention=False):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
            # Clustering
            A = torch.softmax(attention_mask + attention_score, 1)
        else:
            # Clustering
            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 1)

        # attention_visualize(A)
        # query_key_visualize(Q, K)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        # O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if return_attention:
            return O, A
        else:
            return O

class Graph_MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(Graph_MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = GCNConv(dim_K, dim_V)
        self.fc_v = GCNConv(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, attention_mask=None, graph=None, return_attention=False):
        Q = self.fc_q(Q)
        
        if graph is not None:

            (x, edge_index, batch) = graph

            K, V = self.fc_k(x, edge_index), self.fc_v(x, edge_index)

            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)

        else:

            raise ValueError("graph is not defined")

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        attention_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        # attention_score = Q_.bmm(K_.transpose(1,2))
        
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            # Clustering
            A = torch.softmax(attention_mask + attention_score, dim=1)
        else:
            # Clustering
            A = torch.softmax(attention_score, dim=1)

        # attention_visualize(A)
        # query_key_visualize(Q, K)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        # O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if return_attention:
            return O, A
        else:
            return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, mab='MAB'):
        super(SAB, self).__init__()

        if mab == 'MAB':
            self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
        elif mab == 'Graph_MAB':
            self.mab = Graph_MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, attention_mask=None, graph=None):
        return self.mab(X, X, attention_mask, graph)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, mab='MAB'):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        if mab == 'MAB':
            self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
            self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        elif mab == 'Graph_MAB':
            self.mab0 = Graph_MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
            self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, attention_mask=None, graph=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, attention_mask, graph)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, mab='MAB'):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        # nn.init.xavier_normal_(self.S, gain=0.1)

        if mab == 'MAB':
            self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        elif mab == 'Graph_MAB':
            self.mab = Graph_MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, attention_mask=None, graph=None, return_attention=False):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, attention_mask, graph, return_attention)