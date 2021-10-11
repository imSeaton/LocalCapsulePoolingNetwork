import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import DenseDataLoader as DenseLoader
from data_processing import get_dataset
dataset = get_dataset('PROTEINS', sparse=False)
dataloader = DenseLoader(dataset, batch_size=2, shuffle=True)
for data in dataloader:
    print(data)