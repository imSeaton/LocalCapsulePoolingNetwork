import sys
from pygsp import graphs
import numpy as np
from models import *
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser(description='CapsulePool with different Pool Ratio')
parser.add_argument('--data', default='grid', type=str,
                    help='dataset type')
parser.add_argument('--seed', type=int, default=7, help='seed')
parser.add_argument('--num-hidden', type=int, default=32, help='hidden size')

parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--ln", action='store_true')
parser.add_argument("--test", action='store_true')
# parser.add_argument("--ratio", default=0.9, help='Pool Ratio')
parser.add_argument("--recon_adj", default=False, help='loss of reconstruction adj')
parser.add_argument('--mab', default='MAB', type=str,
                        choices=['MAB', 'Graph_MAB'],
                        help='MAB type')
args = parser.parse_args()
# Random Seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set Device
device = 'cuda'

dataset = args.data
# load data
if dataset == 'ring':
    G = graphs.Ring(N=200)
elif dataset == 'grid':
    G = graphs.Grid2d(N1=30, N2=30)
elif dataset == 'twomoon':
    G = graphs.TwoMoons('synthesized', sigmad=0.0, seed=7)

X = G.coords.astype(np.float32)
print(X.shape)
A = G.W
y = np.zeros(X.shape[0])  # X[:,0] + X[:,1]
n_nodes = A.shape[0]

args.num_features = 2

# model_list = ['TopkPool', 'DiffPool', 'MinCutPool', 'GST']
ratio_list = ['0.1', '0.25',  '0.5', '0.8']
# model_dict = {'TopkPool': TOPKPOOL_AE,
#               'DiffPool': DIFFPOOL_AE,
#               'MinCutPool': MINCUTPOOL_AE,
#               'GST': GST_AE}



coo = A.tocoo()
values = coo.data
indices = np.vstack((coo.row, coo.col))

i = torch.LongTensor(indices)
v = torch.FloatTensor(values)

A = torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))
X = torch.FloatTensor(X)
batch = torch.LongTensor([0 for _ in range(X.shape[0])])
A = A.coalesce().indices()

criterion = nn.MSELoss()

preds = []
losses = []
for ratio in ratio_list:
    args.ratio = float(ratio)
    # print(f"args {args}")
    # print(f"args.ratio {args.ratio}")
    model = CapsulePool_AE(args, n_nodes)
    model = model.to(device)
    # parser.add_argument("--ratio", default=ratio, help='Pool Ratio')
    try:
        model.load_state_dict(torch.load("checkpoints/best_{}_{}_{}.pth".format(args.data, "CapsulePool", ratio)))
    except:
        print("Checkpoint of {} missing for data {}_{}".format('CapsulePool', args.data, args.ratio))
        assert False
    model.eval()
    with torch.no_grad():
        X, A, batch = X.to(device), A.to(device), batch.to(device)
        print(model)
        X_out, edge_index, _, _ = model(X, A, batch)
        loss = criterion(X, X_out)
        preds.append(X_out.cpu())
        losses.append(loss.item())

X = X.cpu()
# PLOTS
num_models = len(ratio_list)
plt.figure(figsize=(4*(num_models+1), 4))
pad = 0.1
x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
colors = X[:, 0] + X[:, 1]
plt.subplot(1, num_models+1, 1)
plt.scatter(*X[:, :2].T, c=colors, s=8, zorder=2)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('Original', y=-0.1)
if args.data == 'ring':
    plt.axvline(0, c='k', alpha=0.2)
    plt.axhline(0, c='k', alpha=0.2)
plt.xticks([]),plt.yticks([])
for i in range(num_models):
    plt.subplot(1, num_models+1, 2+i)
    X_out = preds[i]
    plt.scatter(*X_out[:, :2].T, c=colors, s=8, zorder=2)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("CapsulePool_" + ratio_list[i], y=-0.1)
    if args.data == 'ring':
        plt.axvline(0, c='k', alpha=0.2)
        plt.axhline(0, c='k', alpha=0.2)
    plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.savefig("results/total_figure_{}_with_different_pool_ratio.jpg".format(args.data))
print("{} Saved.".format("results/total_figure_{}_with_different_pool_ratio.jpg".format(args.data)))
print("Losses: ", losses)

