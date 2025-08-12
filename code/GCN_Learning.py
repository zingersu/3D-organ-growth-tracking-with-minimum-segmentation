import time
import numpy as np
import os
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def GCN_train(target_coordinates, sample_count, graph_path):

    adj, features, labels, idx_train, idx_val, idx_test = load_data(path=graph_path, dataset="a", sample_count=sample_count)

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()               # GraphConvolution forward
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

    def test():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        f1 = open(graph_path + "\\" + "label_test.txt", 'w')
        output_labels_train_list = labels[idx_train].tolist()
        print(','.join(str(label) for label in output_labels_train_list), end=',', file=f1, sep='')
        output_labels_test_list = output[idx_test].max(1)[1].type_as(labels[idx_test]).tolist()
        print(','.join(str(label) for label in output_labels_test_list), end='\n', file=f1, sep='')
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()

    with open(graph_path + "\\" + "coordinate_matrix.txt", 'r') as f:
        coordinates = [line.strip().split() for line in f]

    with open(graph_path + "\\" + "label_test.txt", 'r') as f:
        numbers = [num for num in f.read().strip().split(',')]

    for i in range(len(coordinates)):
        coordinates[i].append(numbers[i])

    coordinates = np.array(coordinates)
    coordinates = np.hstack((coordinates[:, :3].astype(float), coordinates[:, 3:].astype(int)))

    coordinates = coordinates[sample_count:, :]
    output_len = len(coordinates)

    if output_len == sample_count:
        train_label = coordinates[:, -1:]
        coordinates = np.hstack((target_coordinates, train_label))
        return coordinates
    elif output_len <= 2048:
        return coordinates


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)