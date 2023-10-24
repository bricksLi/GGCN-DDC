import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn import metrics
from dataset_modify import get_data_loader
# from dataset_modify_lst import get_data_loader
# from dataset_ng_369 import get_data_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
import argparse
from config import *
from model_grunot_att_daegc_A_big import Model  #
from evaluation import eva  #
from collections import Counter
import matplotlib.pyplot as plt


class Texting_deagc(nn.Module):
    def __init__(self, num_words, hid_dim, word2vec,
                 num_clusters, freeze=True, v=1):
        super(Texting_deagc, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        self.embedding = Model(num_words, num_clusters,
                               hid_dim=hid_dim, word2vec=word2vec, freeze=freeze)
        self.embedding.load_state_dict(torch.load(args.path, map_location='cpu'))

        # self.reset_parameters()

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, hid_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, loader, device):
        A_preds, adj_labels, labels = [], [], []
        z = torch.empty((1, 96), requires_grad=True).to(device)
        for j, graph in enumerate(loader):
            graph = graph.to(device)
            targets = graph.y

            A_pred, z_, A_ori = model.embedding(graph)

            adj_labels.append(A_ori)

            A_preds.append(A_pred)

            labels.extend(targets.data.cpu())
            z = torch.cat((z, z_), dim=0)

        z = z[1:]

        q = self.get_Q(z)

        return A_preds, adj_labels, labels, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def get_graphs_emb(loader, model, device):
    y, z = [], []
    z = torch.empty((1, 96)).to(device)
    for i in range(len(loader)):
        for j, graph in enumerate(loader[i]):
            graph = graph.to(device)
            target = graph.y
            with torch.no_grad():
                _, z_, _ = model.embedding(graph)
                z = torch.cat((z, z_), dim=0)
            y.extend(target.data.cpu())

    z = z[1:]
    y = np.array(y).flatten()
    return z, y


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_td(cate, loader, model, optimizer, epoch, device, p):
    model.train() if cate == "train" else model.eval()

    loss_sum = 0.
    preds_t, labels_t = [], []
    # emb_temp=torch.empty((1, 96))
    for i in range(len(loader)):

        if epoch % args.update_interval == 0:

            A_preds, adj_labels, labels, z, Q = model(loader[i], device)
            p = target_distribution(Q.detach())

        else:
            A_preds, adj_labels, labels, z, Q = model(loader[i], device)

        labels_t.extend(labels)

        q = Q.detach().data.cpu().numpy().argmax(1)
        preds_t.extend(q)

        kl_loss = F.kl_div(Q.log(), p, reduction='batchmean')

        re_loss = torch.tensor(0., requires_grad=True).float().to(device)  # re-con loss

        for A_pred, A_ori in zip(A_preds, adj_labels):
            re_loss = re_loss + F.binary_cross_entropy(A_pred.view(-1), A_ori.view(-1))


        loss = 10 * kl_loss + re_loss / len(loader[i])

        if cate == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum = loss_sum + loss.data

    y_data = np.array(labels_t)
    y_data_pre = np.array(preds_t)

    acc, nmi, ari, f1 = eva(y_data, y_data_pre, epoch, cate)
    return acc, nmi, loss_sum, y_data_pre, y_data, p

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class MyDataLoader(object):

    def __init__(self, dataset, batch_size, mini_batch_size=0):
        self.total = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        if mini_batch_size == 0:
            self.mini_batch_size = self.batch_size

    def __getitem__(self, item):
        ceil = (item + 1) * self.batch_size
        sub_dataset = self.dataset[ceil - self.batch_size:ceil]
        # if ceil >= self.total:
        #     random.shuffle(self.dataset)
        return DataLoader(sub_dataset, batch_size=self.mini_batch_size,pin_memory=True, num_workers=7)

    def __len__(self):
        if self.total == 0: return 0
        return (self.total - 1) // self.batch_size + 1


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)  # random seed
    torch.manual_seed(SEED)
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='R8')  # 默认使用了R8数据集
    parser.add_argument('--best_epoch', type=int, default=35)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=8, type=int)
    parser.add_argument('--update_interval', default=2, type=int)  # [1,3,5]
    parser.add_argument('--hid_dim', default=96, type=int)
    # parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=0.)
    parser.add_argument('--batch_size', type=int, default=4936)
    parser.add_argument('--mini_batch_size', type=int, default=64)
    # parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--freeze', type=bool, default=True)
    parser.add_argument('--ng', default=3, type=int)  # 20ng
    args = parser.parse_args(args=[])
    print("load dataset")
    print(args)
    # params，参数设置
    start = 0
    args.path = f'./pretrain/pretexting_{args.dataset}_test_{args.best_epoch}.pkl'

    (train_loader, test_loader, valid_loader), word2vec = get_data_loader(args.dataset, args.batch_size,
                                                                          args.mini_batch_size)

    num_words = len(word2vec) - 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p1 = p2 = p3 = torch.tensor(0.).float().to(device)

    model = Texting_deagc(num_words, hid_dim=args.hid_dim, word2vec=word2vec, num_clusters=args.n_clusters)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("-" * 50)
    print(f"params: [start={start}, batch_size={args.batch_size}, lr={args.lr}, weight_decay={args.weight_decay}]")
    print("-" * 50)
    print(model)
    print("-" * 50)
    print(args.dataset)

    # 训练模型
    best_acc = 0.

    print("----------------Initial pretraining--------------------")

    model = model.to(device)
    Z_pre, y = get_graphs_emb(train_loader, model, device)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(Z_pre.cpu().numpy())

    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    acc, nmi, ari, f1 = eva(y, y_pred, cate="pretrain")
    test_acc = []
    acc_lst = []
    nmi_lst = []
    for epoch in range(start, args.max_epoch):  # start training
        t1 = time.time()
        model = model.to(device)

        acc_train, _, train_loss, _, _, p1 = train_td("train", train_loader, model, optimizer, epoch, device, p1)

        acc_valid, _, valid_loss, _, _, p2 = train_td("valid", valid_loader, model, optimizer, epoch, device, p2)

        acc_test, nmi_test, test_loss, preds, labels, p3 = train_td("test", test_loader, model, optimizer, epoch,
                                                                    device, p3)

        acc_lst.append(acc_test)
        nmi_lst.append(nmi_test)

        if best_acc < acc_test:
            best_acc = acc_test

        cost = time.time() - t1
        print((f"epoch={epoch:03d}, cost={cost:.2f}, "
               f"train:[{train_loss:.4f}, {acc_train * 100:.2f}%], "
               f"valid:[{valid_loss:.4f}, {acc_valid * 100:.2f}%], "
               f"test:[{test_loss:.4f}, {acc_test * 100:.2f}%], "
               f"best_acc={best_acc * 100:.2f}%"))

    m = np.array(acc_lst)
    n = np.array(nmi_lst)
    print("Test Acc, NMI, Ari and F1-Score...")
    acc, nmi, ari, f1 = eva(labels, preds, cate="last_test")
