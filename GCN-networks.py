import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from SAG-layers import SAGPool
   
class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args = args
        
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.nhid)
        self.conv6 = GCNConv(self.nhid, self.nhid)
        self.conv7 = GCNConv(self.nhid, self.nhid)
        self.pool = SAGPool(self.nhid*7, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*14, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3= F.relu(self.conv3(x2, edge_index))
        x4= F.relu(self.conv4(x3, edge_index))
        x5= F.relu(self.conv5(x4, edge_index))
        x6= F.relu(self.conv6(x5, edge_index))
        x7= F.relu(self.conv7(x6, edge_index))
        
        x = torch.cat([x1,x2,x3,x4,x5,x6,x7],dim=1)
        
        x, edge_index, _, batch, _ = self.pool(x, edge_index, None, batch)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

