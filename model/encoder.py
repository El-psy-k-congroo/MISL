import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import RGCNConv



class RGCN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_relations):
        super(RGCN, self).__init__()
        self.rgcn1 = RGCNConv(in_features, hidden_features, num_relations, num_bases=30)
        self.rgcn2 = RGCNConv(hidden_features, out_features, num_relations, num_bases=30) #

    def forward(self, x, edge_index, edge_type):
        x = torch.relu(self.rgcn1(x, edge_index, edge_type))
        x = torch.relu(self.rgcn2(x, edge_index, edge_type))
        return x




