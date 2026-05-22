# %%
import time
import argparse
import numpy as np
import torch
from models.ICGNN import ICGNN
from dataset import Dataset, get_coauthor_dataset, get_amazon_dataset, load_social_network
import os
from utils import get_ics_value

from loguru import logger

import torch_geometric.utils as U

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action = 'store_true',
                    default = False, help = 'debug mode')
parser.add_argument('--seed', type = int, default = 11, help = 'Random seed.')
parser.add_argument("--label_rate", type = float, default = 0.05,
                    help = 'rate of labeled data')
parser.add_argument('--noise', type = str, default = 'uniform', choices = ['uniform', 'pair'],
                    help = 'type of noises')
parser.add_argument('--dataset', type = str, default = "cora",
                    choices = ['cora', 'citeseer', 'pubmed', 'dblp', 'cs', 'photo', 'computers', 'blogcatalog'],
                    help = 'dataset')
parser.add_argument('--ptb_rate', type = float, default = 0.2,
                    help = "noise ptb_rate")

parser.add_argument('--weight_decay', type = float, default = 5e-4,
                    help = 'Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type = int, default = 128,
                    help = 'Number of hidden units.')
parser.add_argument('--edge_hidden', type = int, default = 64,
                    help = 'Number of hidden units of MLP graph constructor')
parser.add_argument('--dropout', type = float, default = 0.5,
                    help = 'Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type = int, default = 200,
                    help = 'Number of epochs to train.')
parser.add_argument('--lr', type = float, default = 0.001,
                    help = 'Initial learning rate.')
parser.add_argument('--t_small', type = float, default = 0.05,
                    help = 'threshold of eliminating the edges')
parser.add_argument('--p_u', type = float, default = 0.8,
                    help = 'threshold of adding pseudo labels')
parser.add_argument("--n_p", type = int, default = 50,
                    help = 'number of positive pairs per node')
parser.add_argument("--n_n", type = int, default = 50,
                    help = 'number of negitive pairs per node')
parser.add_argument('--normalize_features', type = bool, default = True)
parser.add_argument("--K", type = int, default = 50,
                    help = 'number of KNN search for each node')

parser.add_argument('--sep_gmm', type = bool, default = True)
parser.add_argument('--temp', type = float, default = 0.2)
parser.add_argument('--feat_dim', type = int, default = 32,
                    help = 'feature dimension of projector')
parser.add_argument('--reassign', type = int, default = 1,
                    help = 'epoch interval of reassign confidence')
parser.add_argument('--pseudo_loss_weight', type = float, default = 1.0,
                    help = 'pseudo labeling loss weight')
parser.add_argument('--rec_loss_weight', type = float, default = 0.03,
                    help = 'weight of loss of edge predictor')
parser.add_argument('--scale1', type = float)
parser.add_argument('--warmup_epochs', type = int, default = 15,
                    help = 'warmup epochs for noise detection')

parser.add_argument('--pagerank_prob', type = float, default = 0.85,
                    help = "random walk probability")
parser.add_argument("--local_conflict_weight", type = float, default = 0.8,
                    help = 'knn conflict value weight')
parser.add_argument("--pseudo_type", type = str, default = 'one',
                    help = 'pseudo labeling loss type')
parser.add_argument("--ics_type", type = str, default = 'global_and_knn',
                    help = 'noise detect type')
parser.add_argument("--runs", type = int, default = 5)

args = parser.parse_known_args()[0]


logger.add(
    f'./logs/{args.dataset}/{args.dataset}_{args.noise}_{args.ptb_rate}_{args.label_rate}.log',
    level = 'INFO')
logger.warning(vars(args))

args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.device = device
print(device)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)
np.random.seed(15)  # Here the random seed is to split the train/val/test data

# %%
if args.dataset == 'dblp':
    from torch_geometric.datasets import CitationFull
    import torch_geometric.utils as utils
    dataset = CitationFull('./data', 'dblp')
elif args.dataset == "cora" or args.dataset == 'citeseer' or args.dataset == 'pubmed':
    data = Dataset(root = './data', name = args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train = idx_train[:int(args.label_rate * adj.shape[0])]
elif args.dataset == "cs" or args.dataset == "physics":
    dataset = get_coauthor_dataset(args.dataset, args.normalize_features)
elif args.dataset == "computers" or args.dataset == "photo":
    dataset = get_amazon_dataset(args.dataset, args.normalize_features)

if args.dataset == "dblp" or args.dataset == "cs" or args.dataset == "computers" or args.dataset == "photo":
    adj = U.to_scipy_sparse_matrix(dataset.data.edge_index)
    features = dataset.data.x.numpy()
    labels = dataset.data.y.numpy()
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    idx_test = idx[:int(0.8 * len(labels))]
    idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
    idx_train = idx[int(0.9 * len(labels)):int((0.9 + args.label_rate) * len(labels))]

# %% add noise to the labels
from utils import noisify_with_P

ptb = args.ptb_rate
nclass = labels.max() + 1
train_labels = labels[idx_train]
noise_y, P = noisify_with_P(train_labels, nclass, ptb, 10, args.noise)
noise_labels = labels.copy()
noise_labels[idx_train] = noise_y

# Get indices of noisy labels
idx_noisy = np.where(labels != noise_labels)[0]
idx_noisy = np.intersect1d(idx_noisy, idx_train)
# Get indices of clean labels
idx_clean = np.where(labels == noise_labels)[0]
idx_clean = np.intersect1d(idx_clean, idx_train)

# calculating the Personalized PageRank Matrix
ppr_file = os.path.join('.', 'data',
                        f'{args.dataset}_{args.noise}_{args.ptb_rate}_{args.label_rate}_ppr.pt')
if os.path.exists(ppr_file):
    Pi = torch.load(ppr_file)
else:
    pr_prob = 1 - args.pagerank_prob
    A = torch.from_numpy(adj.todense()).float()
    A_hat = A + torch.eye(A.size(0))  # add self-loop
    D = torch.diag(torch.sum(A_hat, 1))
    D = D.inverse().sqrt()
    A_hat = torch.mm(torch.mm(D, A_hat), D)
    Pi = pr_prob * ((torch.eye(A.size(0)) - (1 - pr_prob) * A_hat).inverse())
    torch.save(Pi, ppr_file)

gpr_matrix = []  # the class-level influence distribution
for iter_c in range(nclass):
    class_mask = noise_labels[idx_train] == iter_c
    selected_idx = torch.LongTensor(idx_train) * torch.LongTensor(class_mask)
    selected_idx = selected_idx[torch.nonzero(selected_idx).squeeze(1)]
    iter_Pi = Pi[selected_idx]
    iter_gpr = torch.mean(iter_Pi, dim = 0).squeeze()
    gpr_matrix.append(iter_gpr)

temp_gpr = torch.stack(gpr_matrix, dim = 0)
temp_gpr = temp_gpr.transpose(0, 1)
gpr = temp_gpr

ics_list = get_ics_value(args, Pi, gpr, idx_train, noise_labels)


idx_train, idx_clean, idx_noisy = torch.LongTensor(idx_train), \
                                  torch.LongTensor(idx_clean), \
                                  torch.LongTensor(idx_noisy)

test_accs = []

for seed in range(args.seed, args.seed + args.runs):
    logger.info('testing with seed {:d}'.format(seed))

    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    icgnn = ICGNN(args, device)
    icgnn.fit(features, adj, noise_labels, labels, idx_train,
              idx_val, idx_test, ics_list, idx_clean, idx_noisy, Pi)

    logger.info("=====test set accuracy=======")
    test_acc = icgnn.test(idx_test)
    logger.info("===================================")

    test_accs.append(test_acc)
# %%
logger.info("=====accuracy over 5 runs=======")
logger.info('test_acc: [mean:{:.6f}][std:{:.6f}]'.format(
    np.mean(test_accs),
    np.std(test_accs),
))
logger.info("===================================")
