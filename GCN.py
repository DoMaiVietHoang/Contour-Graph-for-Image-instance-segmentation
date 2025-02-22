import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
class ContourMerge(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(ContourMerge, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for __ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.mlp = nn.Sequential(
            torch.nn.Linear(out_channels,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,2),
        )
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = torch.relu(x)
        x = self.convs[-1](x, edge_index)
        edgeFeature = (x[edge_index[0]] + x[edge_index[1]]) / 2
        Y = self.mlp(edgeFeature)
        return Y