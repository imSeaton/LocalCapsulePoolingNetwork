import sys
from pygsp import graphs
import numpy as np
from models import *
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='ASAP and CapsulePool with different Pool Ratio')
parser.add_argument('--data', default='grid', type=str,
                    help='dataset type')
parser.add_argument('--seed', type=int, default=7, help='seed')
parser.add_argument('--num-hidden', type=int, default=32, help='hidden size')

parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--ln", action='store_true')
parser.add_argument("--test", action='store_true')
parser.add_argument("--ratio", default=0.5, help='Pool Ratio')
parser.add_argument("--recon_adj", default=False, help='loss of reconstruction adj')
parser.add_argument('--mab', default='MAB', type=str,
                        choices=['MAB', 'Graph_MAB'],
                        help='MAB type')
args = parser.parse_args()

# Set Device
device = 'cuda'

dataset = args.data
# load data
if dataset == 'ring':
    G = graphs.Ring(N=200)
elif dataset == 'grid':
    G = graphs.Grid2d(N1=30, N2=30)
elif dataset == 'twomoon':
    G = graphs.TwoMoons('synthesized', sigmad=0.0, seed=42)

X = G.coords.astype(np.float32)
print(X.shape)
A = G.W
y = np.zeros(X.shape[0])  # X[:,0] + X[:,1]
n_nodes = A.shape[0]

args.num_features = 2

model_list = ['ASAP', 'CapsulePool']
ratio_list = ['0.1', '0.25',  '0.5', '0.8']
model_dict = {'ASAP': ASAP_AE,
              'CapsulePool': CapsulePool_AE}



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
for k in range(len(model_list)):

    for ratio in ratio_list:

        args.ratio = float(ratio)
        # print(f"model_list[k] {model_list[k]}")
        model = model_dict[model_list[k]](args, n_nodes)
        model = model.to(device)
        try:
            model.load_state_dict(torch.load("checkpoints/best_{}_{}_{}.pth".format(args.data, model_list[k], ratio)))
        except:
            print("Checkpoint of {} missing for data {}_{}".format(model_list[k], args.data, args.ratio))
            assert False
        model.eval()
        with torch.no_grad():
            X, A, batch = X.to(device), A.to(device), batch.to(device)
            # print(model)
            X_out, edge_index, _, _ = model(X, A, batch)
            loss = criterion(X, X_out)
            preds.append(X_out.cpu())
            losses.append(loss.item())

# PLOTS
X = X.cpu()
num_models = len(ratio_list)
plt.figure(figsize=(4*(num_models+1), 4*2))
for k in range(len(model_list)):
    pad = 0.1
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    colors = X[:, 0] + X[:, 1]
    plt.subplot(len(model_list), num_models+1, 1+(num_models+1)*k)
    plt.scatter(*X[:, :2].T, c=colors, s=8, zorder=2)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Original', y=-0.1)
    if args.data == 'ring':
        plt.axvline(0, c='k', alpha=0.2)
        plt.axhline(0, c='k', alpha=0.2)
    plt.xticks([]),plt.yticks([])
    for i in range(num_models):
        # print(f"2+i+k {2+i+(num_models+1)*k}")
        print(f"(k+1, num_models+1, 2+i+(num_models+1)*k) {(len(model_list), num_models+1, 2+i+(num_models+1)*k)}")
        plt.subplot(len(model_list), num_models+1, 2+i+(num_models+1)*k)
        X_out = preds[i+k*num_models]
        plt.scatter(*X_out[:, :2].T, c=colors, s=8, zorder=2)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        method_name = model_list[k]
        if method_name == 'CapsulePool':
            method_name = 'LCP'

        plt.title(method_name + "_" + ratio_list[i], y=-0.1)
        if args.data == 'ring':
            plt.axvline(0, c='k', alpha=0.2)
            plt.axhline(0, c='k', alpha=0.2)
        plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.savefig("results/total_figure_{}_with_different_model_pool_ratio.jpg".format(args.data))
print("{} Saved.".format("results/total_figure_{}_with_different_pool_ratio.jpg".format(args.data)))
print("Losses: ", losses)

