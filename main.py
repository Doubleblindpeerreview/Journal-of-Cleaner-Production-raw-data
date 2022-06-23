import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from GCN-networks import  Net
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split
from sklearn import metrics
import numpy
from torch.utils.data import Subset

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=' ')
parser.add_argument('--batch_size', type=int, default=' ')
parser.add_argument('--lr', type=float, default=' ',
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=' ')
parser.add_argument('--nhid', type=int, default=' ',
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=' ')
parser.add_argument('--dropout_ratio', type=float, default=' ')
parser.add_argument('--dataset', type=str, default=' ')
parser.add_argument('--epochs', type=int, default=' ',
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=' ',
                    help='patience for earlystopping')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'

dataset = TUDataset(os.path.join('./data',args.dataset),name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset)*(' '))
num_val = int(len(dataset)*' ')
num_test = len(dataset) - (num_training+num_val)
training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)


min_loss = 1e10
patience = 0

for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc,val_loss = test(model,val_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(),' .pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 

model = Net(args).to(args.device)
model.load_state_dict(torch.load('./.pth'))
test_acc,test_loss = test(model,test_loader)
print("Test accuarcy:{}".format(test_acc))


categories = [' ', ' ', ' ', ' ']
a = []
b = []
for data in test_loader:
    data = data.to(args.device)
    out = model(data)
    pred = out.max(dim=1)[1]
    pred = pred.tolist() 
    data.y = data.y.tolist() 
    a.append(pred)
    b.append(data.y)
    
print("Precision, Recall and F1-Score...")
print(metrics.classification_report(b, a, target_names=categories)) 
cm = metrics.confusion_matrix(b, a)
print(cm)
   

