"""
    Disentangle function
"""
import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros


# linearly disentangle node representations
class LinearDisentangle(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearDisentangle, self).__init__()
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    # todo：这里初始化是什么需要靠考量清楚
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias