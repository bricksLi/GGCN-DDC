import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import nn as gnn
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from collections import Counter



class GraphLayer(gnn.MessagePassing):

    def __init__(self, in_dim, out_dim, dropout=0.5,
                 act=torch.relu, bias=False, step=2):
        super(GraphLayer, self).__init__(aggr='add')
        self.step = step
        self.act = act
        self.encode = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, g):
        # x=>ht
        x = self.encode(x)
        x = self.act(x)
    
        a = self.propagate(edge_index=g.edge_index, x=x,
                           edge_attr=self.dropout(g.edge_attr))

        return a

    def message(self, x_j, edge_attr):
        return x_j * edge_attr.unsqueeze(-1)

    def update(self, inputs):
        return inputs

    def graph2batch(self, x, length):
        x_list = []
        for l in length:
            x_list.append(x[:l])
            x = x[l:]
        x = pad_sequence(x_list, batch_first=True)
        return x


class GAT_NET(torch.nn.Module):
    def __init__(self, features, hidden, heads=4):
        super(GAT_NET, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=4)

        self.gat2 = GATConv(hidden * heads, 1)

    def forward(self, x,g):

        edge_index = g.edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return x  #


class ReadoutLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5,
                 act=torch.relu, bias=False):
        super(ReadoutLayer, self).__init__()
        self.act = act
        self.bias = bias
        self.att = GAT_NET(in_dim, 16)

        self.emb = nn.Linear(in_dim, in_dim, bias=True)
        # self.mlp = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(out_dim, out_dim, bias=True)
        # )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask,g):

        att_ = self.att(x,g).sigmoid()
        att = self.graph2batch(att_, g.length)

        emb_ = self.act(self.emb(x))

        emb=self.graph2batch(emb_, g.length)
        x = att * emb
        x_=att_*emb_

        x = self.__max(x, mask) + self.__mean(x, mask)

        # x=self.mlp(x)
        return x,x_

    def graph2batch(self, x, length):
        x_list = []
        for l in length:
            x_list.append(x[:l])
            x = x[l:]
        x = pad_sequence(x_list, batch_first=True)
        return x

    def __max(self, x, mask):
        return (x + (mask - 1) * 1e9).max(1)[0]

    def __mean(self, x, mask):
        return (x * mask).sum(1) / mask.sum(1)


class Model(nn.Module):
    def __init__(self, num_words, num_classes, in_dim=300, hid_dim=96,
                 step=2, dropout=0.5, word2vec=None, freeze=True):
        super(Model, self).__init__()
        if word2vec is None:
            self.embed = nn.Embedding(num_words + 1, in_dim, num_words)
        else:
            self.embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(word2vec).float(), freeze, num_words)

        self.gcn = GraphLayer(in_dim, hid_dim, act=torch.tanh, dropout=dropout, step=step)

        self.read = ReadoutLayer(hid_dim, num_classes, act=torch.tanh, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, g):
        mask = self.get_mask(g)
        x = self.embed(g.x)

        x= self.gcn(x, g)
        x,z= self.read(x, mask,g)

        # z=F.normalize(z,p=2,dim=1)
        A_pred=self.dot_product_decode(z)
        A_ori=torch.sparse_coo_tensor(g.edge_index, g.edge_attr, torch.Size([g.x.shape[0], g.x.shape[0]])).to_dense()
        return A_pred,x,A_ori

    def dot_product_decode(self,Z):
        A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
        return A_pred

    def get_mask(self, g):
        mask = pad_sequence([torch.ones(l) for l in g.length], batch_first=True).unsqueeze(-1)

        if g.x.is_cuda: mask = mask.cuda()
        return mask