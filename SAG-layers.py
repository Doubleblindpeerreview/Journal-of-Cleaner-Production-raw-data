from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch


class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.5,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        
        self.conv = Conv(in_channels,1)
        self.add_module('attention_layer',self.conv)
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = self.conv(x,edge_index).view(-1)

        perm = topk(score, self.ratio, batch)
        x = x[perm] * torch.tanh(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm