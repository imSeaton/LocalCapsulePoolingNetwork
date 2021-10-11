import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import random
import numpy as np

from tqdm import tqdm

from pygsp import graphs
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
from torch_geometric.nn import GCNConv, dense_mincut_pool
import torch.nn.functional as F

from layers import P
from models import *

def load_model(args, n_nodes):
    if args.model == 'MinCutPool':
        model = MINCUTPOOL_AE(args, n_nodes)
    elif args.model == 'DiffPool':
        model = DIFFPOOL_AE(args, n_nodes)
    elif args.model == 'TopkPool':
        model = TOPKPOOL_AE(args, n_nodes)
    elif args.model == 'SAGPool':
        model = SAGPOOL_AE(args, n_nodes)
    elif args.model == 'ASAP':
        model = ASAP_AE(args, n_nodes)
    elif args.model == 'CapsulePool':
        model = CapsulePool_AE(args, n_nodes)
    else:
        model = GST_AE(args, n_nodes)

    return model

def main(args):
    # if args.model == 'MinCutPool':
    #     args.device = 'cpu'
    # else:
    #     args.device = 'cuda'
    args.device = 'cuda'
    # args.device = 'cpu'
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # Random Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # HYPERPARAMETERS
    dataset = args.data

    # DATASET DEFINITION
    if dataset == 'ring':
        G = graphs.Ring(N=200)
    elif dataset == 'grid':
        G = graphs.Grid2d(N1=30, N2=30)
    elif dataset == 'smallgrid':
        G = graphs.Grid2d(N1=10, N2=10)
    elif dataset == 'logo':
        # Very Very Hard
        G = graphs.Logo()
    elif dataset == 'david':
        G = graphs.DavidSensorNet(N=200, seed=42)
    elif dataset == 'twomoon':
        G = graphs.TwoMoons('synthesized', sigmad=0.0, seed=42)
    elif dataset == 'community':
        G = graphs.Community(N=240, Nc=3, comm_sizes=[80,80,80], seed=42)

    X = G.coords.astype(np.float32)
    A = G.W
    y = np.zeros(X.shape[0])  # X[:,0] + X[:,1]
    n_nodes = A.shape[0]

    args.num_features = 2
    device = args.device

    model = load_model(args, n_nodes)

    print(model)
    model = model.to(device)

    coo = A.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    A = torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))
    X = torch.FloatTensor(X)
    batch = torch.LongTensor([0 for _ in range(X.shape[0])])
    A = A.coalesce().indices()

    ITER = 10000
    es_patience = 1000
    tol = 1e-5
    best_loss = 1000000
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    iterator = tqdm(range(ITER))
    criterion = nn.MSELoss()
    losses = []
    patience = es_patience
    for i in iterator:
        if args.test: break # Checkpoint should be saved beforehand
        X = X.to(device)
        A = A.to(device)
        batch = batch.to(device)

        X_out, A_out, mc_loss, o_loss = model(X, A, batch)

        if args.loss_adj:

            loss = criterion(to_dense_adj(A), A_out)

        else:

            loss = criterion(X, X_out)            

        loss = loss + mc_loss + o_loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())

        if i % 100 == 0:
            print("Iteration {}: Loss {}".format(i, loss.item()))
        
        if loss.item() + tol < best_loss:
            best_loss = loss.item()
            patience = es_patience
            torch.save(model.state_dict(), "checkpoints/best_{}_{}_{}.pth".format(args.data, args.model, args.ratio))
        else:
            patience -= 1
            if patience == 0:
                iterator.close()
                break

    # # For visualization (Evaluate)
    # model = load_model(args, n_nodes)
    # model = model.to(device)
    # model.load_state_dict(torch.load("checkpoints/best_{}_{}.pth".format(args.data, args.model)))
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        A = A.to(device)
        batch = batch.to(device)
        X_out, A_out, _, _ = model(X, A, batch)
        
        if args.loss_adj:

            loss = criterion(to_dense_adj(A), A_out)

        else:

            loss = criterion(X, X_out)            

    print("MSE Loss: ", criterion(X, X_out))
    # print("Adj Loss: ", criterion(to_dense_adj(A), A_out))

    X = X.cpu()
    X_out = X_out.cpu()

    # PLOTS
    plt.plot(losses)
    plt.title('Loss')
    plt.figure(figsize=(8, 4))
    pad = 0.1
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    colors = X[:, 0] + X[:, 1]
    plt.subplot(1, 2, 1)
    plt.scatter(*X[:, :2].T, c=colors, s=8, zorder=2)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Original')
    # plt.axvline(0, c='k', alpha=0.2)
    # plt.axhline(0, c='k', alpha=0.2)
    plt.subplot(1, 2, 2)
    plt.scatter(*X_out[:, :2].T, c=colors, s=8, zorder=2)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Reconstructed')
    # plt.axvline(0, c='k', alpha=0.2)
    # plt.axhline(0, c='k', alpha=0.2)
    plt.tight_layout()
    plt.savefig("results/autoencoder_{}_{}_{}.jpg".format(args.data, args.model, args.ratio))
    print("{} Saved.".format("results/autoencoder_{}_{}_{}.jpg".format(args.data, args.model, args.ratio)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GST')
    parser.add_argument('--data', default='grid', type=str,
                        choices=['ring', 'grid', 'smallgrid', 'logo', 'david', 'twomoon', 'community'],
                        help='dataset type')
    parser.add_argument('--seed', type=int, default=7, help='seed')
    parser.add_argument("--model", type=str, default='CapsulePool',
                        choices=['GST', 'MinCutPool', 'DiffPool', 'TopkPool', 'SAGPool', 'ASAP', 'CapsulePool'])
    parser.add_argument('--num-hidden', type=int, default=32, help='hidden size')

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ln", action='store_true')
    parser.add_argument("--test", action='store_true')

    parser.add_argument("--recon-adj", action='store_true')
    parser.add_argument("--loss-adj", action='store_true')

    parser.add_argument('--mab', default='MAB', type=str,
                        choices=['MAB', 'Graph_MAB'],
                        help='MAB type')
    parser.add_argument('--ratio', default=0.25, type=float,
                        help='Pool ratio')

    args = parser.parse_args()
    print(args)
    main(args)