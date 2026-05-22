# %%
import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import torch_geometric.utils as utils
import scipy.sparse as sp
from models.GCN import GCN
from utils import accuracy, sparse_mx_to_torch_sparse_tensor, get_ics_value

from loguru import logger
import pickle


class ICGNN:
    def __init__(self, args, device):

        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_edge_index = None
        self.best_pred_graph = None
        self.weights = None
        self.estimator = None
        self.model = None
        self.pred_edge_index = None
        self.nclass = None
        self.scale1 = args.scale1
        self.targets_pseudo = None
        self.contrast_mode = 'all'
        self.T = args.temp
        self.pseudo_type = args.pseudo_type

    def fit(self, features, adj, labels, clean_labels, idx_train, idx_val, idx_test, ics_list, idx_clean, idx_noisy,
            Pi_matrix):
        args = self.args

        self.structure_ics = torch.FloatTensor(ics_list).to(args.device)
        self.Pi_matrix = Pi_matrix.to(args.device)
        self.idx_train, self.idx_val, self.idx_test = torch.LongTensor(idx_train).to(args.device), torch.LongTensor(
            idx_val).to(args.device), torch.LongTensor(idx_test).to(args.device)
        self.idx_clean, self.idx_noisy = torch.LongTensor(idx_clean).to(args.device), torch.LongTensor(idx_noisy).to(
            args.device)

        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        self.adj = torch.from_numpy(adj.todense()).float().to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        labels = torch.LongTensor(np.array(labels)).to(self.device)

        self.edge_index = edge_index
        self.features = features
        self.labels = labels

        self.clean_labels = torch.LongTensor(np.array(clean_labels)).to(self.device)
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(idx_train))).to(self.device)

        self.args.nfeat, self.args.nclass = features.shape[1], labels.max().item() + 1
        self.nclass = self.args.nclass
        #
        self.model = GCN(nfeat = features.shape[1],
                         nhid = self.args.hidden,
                         nclass = labels.max().item() + 1,
                         self_loop = True,
                         dropout = self.args.dropout, device = self.device).to(self.device)

        self.estimator = EstimateAdj(features.shape[1], args, idx_train, device = self.device).to(self.device)
        self.pred_edge_index = self.KNN(edge_index, features, self.args.K, idx_train)

        self.optimizer = optim.Adam(
            list(self.estimator.parameters()) + list(self.model.parameters()),
            lr = args.lr, weight_decay = args.weight_decay)

        self.warm_up = True
        # Train model
        t_total = time.time()

        # Assign Confidence
        self.pseudo_labeling()

        for epoch in range(args.epochs):
            if (epoch % args.reassign) == 0:
                self.pseudo_labeling(epoch)
            self.train(epoch, features, edge_index, self.idx_train, self.idx_val, self.idx_test)

        logger.info("Optimization Finished!")
        logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        logger.info("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

        logger.info("=====validation set accuracy=======")
        self.test(self.idx_val)
        logger.info("===================================")

    def train(self, epoch, features, edge_index, idx_train, idx_val, idx_test):
        args = self.args

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        # obtain representations and rec loss of the estimator
        representations, rec_loss = self.estimator(edge_index, features)

        pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim = 1)
        origin_w = torch.cat([torch.ones(edge_index.shape[1]), torch.zeros(self.pred_edge_index.shape[1])]).to(
            self.device)

        predictor_weights, _ = self.estimator.get_estimated_weigths(pred_edge_index, representations, origin_w)
        edge_remain_idx = torch.where(predictor_weights != 0)[0].detach()
        predictor_weights = predictor_weights[edge_remain_idx]
        pred_edge_index = pred_edge_index[:, edge_remain_idx]

        logits, feat = self.model(features,
                                  pred_edge_index,
                                  predictor_weights,
                                  return_feature = True)

        with torch.no_grad():
            def comb(p1, p2, lam):
                return (1 - lam) * p1 + lam * p2

            def sharpen_distribution(prob_dist, temperature):
                if temperature == 1.0:
                    return prob_dist
                logits = torch.log(prob_dist)
                sharpened_logits = logits / temperature
                sharpened_probs = torch.softmax(sharpened_logits, dim = 1)
                return sharpened_probs

            prob = F.softmax(logits.detach(), dim = 1)

            targets_onehot_noise = F.one_hot(self.labels, self.nclass).float()
            train_confidence = self.confidence[self.idx_train].unsqueeze(1)

            global_neighbors_smoothing = torch.mm(self.Pi_matrix, prob)
            global_neighbors_smoothing = sharpen_distribution(global_neighbors_smoothing, self.T)
            targets_corrected = comb(global_neighbors_smoothing[self.idx_train], targets_onehot_noise[self.idx_train],
                                     train_confidence * self.scale1)

            # pseudo labeling
            targets_pseudo = global_neighbors_smoothing
            max_prob, targets_pseudo_2 = prob.max(dim = 1)
            targets_pseudo_2 = F.one_hot(targets_pseudo_2, self.nclass).float()

        def CE(logits, targets):
            return - (targets * F.log_softmax(logits, dim = 1)).sum(-1).mean()

        cls_loss = (1 - self.estimated_noise_ratio) * CE(logits[self.idx_train], targets_corrected) + \
                   self.estimated_noise_ratio * CE(logits[self.idx_train], targets_onehot_noise[self.idx_train])

        filter_condition = max_prob[self.idx_unlabel] > self.args.p_u
        idx_add = self.idx_unlabel[filter_condition]
        if self.pseudo_type == 'mix':
            pseudo_loss = 0.5 * CE(logits[idx_add], targets_pseudo[idx_add]) + 0.5 * CE(logits[idx_add],
                                                                                        targets_pseudo_2[idx_add])
        elif self.pseudo_type == 'one':
            pseudo_loss = CE(logits[self.idx_unlabel], targets_pseudo[self.idx_unlabel])
        elif self.pseudo_type == 'no':
            pseudo_loss = torch.zeros((1,), device = self.device).mean()

        loss = cls_loss + \
               pseudo_loss * self.args.pseudo_loss_weight + \
               rec_loss * self.args.rec_loss_weight

        loss.backward()

        self.optimizer.step()

        acc_train = accuracy(logits[idx_train].detach(), self.labels[idx_train])

        # Evaluate validation set performance separately,
        self.model.eval()
        outputs = self.model(features,
                             pred_edge_index,
                             predictor_weights)
        # output = outputs[-1]
        acc_val = accuracy(outputs[idx_val], self.labels[idx_val])

        if acc_val > self.best_val_acc:
            acc_test = accuracy(outputs[idx_test], self.labels[idx_test])
            self.best_val_acc = acc_val
            self.best_pred_graph = predictor_weights.detach()
            self.best_edge_index = pred_edge_index.detach()
            self.weights = deepcopy(self.model.state_dict())

            logger.info(
                'saving current graph/gcn, epoch: {:d}, best_val_acc: {:.4f}, best_test_acc: {:.4f}, edge shape: {:d}'.format(
                    epoch, self.best_val_acc.item(), acc_test.item(), pred_edge_index.shape[1]))

    def test(self, idx_test):
        """Evaluate the performance of ICGNN on test set
        """
        features = self.features
        labels = self.labels

        self.model.eval()
        estimated_weights = self.best_pred_graph
        pred_edge_index = self.best_edge_index
        outputs = self.model(features,
                             pred_edge_index,
                             estimated_weights)
        # output = outputs[-1]
        acc_test = accuracy(outputs[idx_test], labels[idx_test])

        logger.info("\tClassifer results: accuracy= {:.4f}".format(acc_test.item()))

        return float(acc_test)

    def extract_features(self, model):
        model.eval()

        representations, rec_loss = self.estimator(self.edge_index, self.features)
        if self.warm_up:
            predictor_weights, _ = self.estimator.get_estimated_weigths(self.edge_index, representations)
            out, feat = model(self.features, self.edge_index, predictor_weights, return_feature = True)
        else:
            pred_edge_index = torch.cat([self.edge_index, self.pred_edge_index], dim = 1)
            origin_w = torch.cat([torch.ones(self.edge_index.shape[1]), torch.zeros(self.pred_edge_index.shape[1])]).to(
                self.device)

            predictor_weights, _ = self.estimator.get_estimated_weigths(pred_edge_index, representations, origin_w)
            edge_remain_idx = torch.where(predictor_weights != 0)[0].detach()
            predictor_weights = predictor_weights[edge_remain_idx]
            pred_edge_index = pred_edge_index[:, edge_remain_idx]

            out, feat = model(self.features, pred_edge_index, predictor_weights, return_feature = True)
        features = F.normalize(feat, dim = 1)
        predict_labels = F.softmax(out, dim = 1)
        all_labels = self.labels

        return features, predict_labels, all_labels

    # Calculate structure-level and attribute-level ICS
    def pseudo_labeling(self, epoch = -1):
        features, predict_labels, all_labels = self.extract_features(self.model)

        lamba = 0.5 + (self.args.local_conflict_weight - 0.5) * max(epoch / self.args.warmup_epochs, 1)
        if self.args.ics_type == 'global_and_knn':
            self.attribute_ics = self.construct_knn(features, K = 5)
            self.confidence = self.noise_detect(labels = self.labels,
                                                ics_list = (lamba * self.attribute_ics + (1 - lamba) * self.structure_ics))
        elif self.args.ics_type == 'global_and_knn_all':
            self.attribute_ics = self.construct_knn(features, K = 10, use_all = True)
            self.confidence = self.noise_detect(labels = self.labels,
                                                ics_list = (lamba * self.attribute_ics + (1 - lamba) * self.structure_ics))

        estimated_noise_ratio = (self.confidence[self.idx_train] > 0.5).float().mean().item()
        self.estimated_noise_ratio = estimated_noise_ratio
        if self.args.scale1 is None:
            self.scale1 = estimated_noise_ratio


    def construct_knn(self, features, K = 10, use_all = False):
        if use_all:
            curr_query = features
        else:
            curr_query = features[self.idx_train]

        adj_matrix = torch.mm(curr_query, curr_query.transpose(0, 1))
        knn_weight, knnG = torch.topk(adj_matrix, k = K + 1)

        num_nodes = len(knnG)
        row = []
        col = []
        weights = torch.flatten(F.softmax(knn_weight[:, 1:], dim = 1))

        for node, neighbors in enumerate(knnG):
            row.append(torch.LongTensor([node] * len(neighbors[1:])))
            col.append(neighbors[1:])

        row = torch.concat(row, dim = 0).to(self.device)
        col = torch.concat(col, dim = 0).to(self.device)
        adj = torch.sparse_coo_tensor(torch.vstack([row, col]), torch.ones(len(row)).to(self.device),
                                      size = (num_nodes, num_nodes))
        self.adj_with_weight = torch.sparse_coo_tensor(torch.vstack([row, col]), weights.to(self.device),
                                                       size = (num_nodes, num_nodes))
        adj = adj.to_dense()

        labels = self.labels[self.idx_train]

        pr_prob = 1 - self.args.pagerank_prob
        A = adj.to(self.device)
        A_hat = A + torch.eye(A.size(0)).to(self.device)  # add self-loop
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        Pi = pr_prob * ((torch.eye(A.size(0)).to(self.device) - (1 - pr_prob) * A_hat).inverse())

        gpr_matrix = []  # the class-level influence distribution
        for iter_c in range(self.nclass):
            class_mask = (labels == iter_c)
            if use_all:
                selected_idx = self.idx_train * class_mask
                selected_idx = selected_idx[torch.nonzero(selected_idx).squeeze(1)]
                iter_Pi = Pi[selected_idx]
            else:
                iter_Pi = Pi[class_mask]
            iter_gpr = torch.mean(iter_Pi, dim = 0).squeeze()
            gpr_matrix.append(iter_gpr)

        temp_gpr = torch.stack(gpr_matrix, dim = 0)
        temp_gpr = temp_gpr.transpose(0, 1)
        gpr = temp_gpr

        if use_all:
            attribute_ics_list = get_ics_value(self.args, Pi, gpr,
                                               torch.LongTensor(list(range(num_nodes))).to(self.device), self.labels)
            attribute_ics_list = torch.from_numpy(attribute_ics_list).float().to(self.device)
            attribute_ics_list = (attribute_ics_list - attribute_ics_list.min()) / (attribute_ics_list.max() - attribute_ics_list.min())

            return attribute_ics_list
        else:
            attribute_ics_list = get_ics_value(self.args, Pi, gpr,
                                               torch.LongTensor(list(range(num_nodes))).to(self.device), labels)
            attribute_ics_list = torch.from_numpy(attribute_ics_list).float().to(self.device)
            attribute_ics_list = (attribute_ics_list - attribute_ics_list.min()) / (attribute_ics_list.max() - attribute_ics_list.min())

            # set default value 0 to unlabeled node
            attribute_ics_all = torch.zeros(len(self.features)).float().to(self.device)
            attribute_ics_all.scatter_(0, self.idx_train, attribute_ics_list)

            return attribute_ics_all

    # Use GMM to assign confidence
    def noise_detect(self, labels, ics_list):
        conflict_value = ics_list.cpu().numpy()[:, np.newaxis]
        labels = labels.cpu().numpy()

        unlabeled_mask = np.zeros(ics_list.shape[0], dtype = np.bool_)
        unlabeled_mask[self.idx_train.cpu().numpy()] = True

        from sklearn.mixture import GaussianMixture
        confidence = np.zeros((conflict_value.shape[0],))
        if self.args.sep_gmm:
            for i in range(self.nclass):
                mask = (labels == i) & unlabeled_mask
                c = conflict_value[mask, :]
                if c.shape[0] == 1:
                    confidence[mask] = 0.99
                    continue
                gm = GaussianMixture(n_components = 2, random_state = 0).fit(c)
                pdf = gm.predict_proba(c)
                confidence[mask] = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
        else:
            gm = GaussianMixture(n_components = 2, random_state = 0).fit(conflict_value)
            pdf = gm.predict_proba(conflict_value)
            confidence = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
        confidence = torch.from_numpy(confidence).float().to(self.device)
        return confidence

    def get_train_edge(self, edge_index, features, n_p, idx_train):
        '''
        obtain the candidate edge between labeled nodes and unlabeled nodes based on cosine sim
        n_p is the top n_p labeled nodes similar with unlabeled nodes
        '''

        if n_p == 0:
            return None

        poten_edges = []
        if n_p > len(idx_train) or n_p < 0:
            for i in range(len(features)):
                indices = set(idx_train)
                indices = indices - set(edge_index[1, edge_index[0] == i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in range(len(features)):
                sim = torch.div(torch.matmul(features[i], features[idx_train].T),
                                features[i].norm() * features[idx_train].norm(dim = 1))
                _, rank = sim.topk(n_p)
                if rank.max() < len(features) and rank.min() >= 0:
                    indices = idx_train[rank.cpu().numpy()]
                    indices = set(indices)
                else:
                    indices = set()
                indices = indices - set(edge_index[1, edge_index[0] == i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges, len(features)).to(self.device)

        return poten_edges

    def get_model_edge(self, pred):

        idx_add = self.idx_unlabel[(pred.max(dim = 1)[0][self.idx_unlabel] > self.args.p_u)]

        row = self.idx_unlabel.repeat(len(idx_add))
        col = idx_add.repeat(len(self.idx_unlabel), 1).T.flatten()
        mask = (row != col)
        unlabel_edge_index = torch.stack([row[mask], col[mask]], dim = 0)

        return unlabel_edge_index, idx_add

    def KNN(self, edge_index, features, K, idx_train):
        if K == 0:
            return torch.LongTensor([])

        poten_edges = []
        if K > len(idx_train):
            for i in range(len(features)):
                sim = torch.div(torch.matmul(features[i], features[self.idx_unlabel].T),
                                features[i].norm() * features[self.idx_unlabel].norm(dim = 1))
                _, rank = sim.topk(K)
                indices = self.idx_unlabel[rank.cpu().numpy()]
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in idx_train:
                sim = torch.div(torch.matmul(features[i], features[self.idx_unlabel].T),
                                features[i].norm() * features[self.idx_unlabel].norm(dim = 1))
                _, rank = sim.topk(K)
                indices = self.idx_unlabel[rank.cpu().numpy()]
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
            for i in self.idx_unlabel:
                sim = torch.div(torch.matmul(features[i], features[idx_train].T),
                                features[i].norm() * features[idx_train].norm(dim = 1))
                _, rank = sim.topk(K)
                indices = idx_train[rank.cpu().numpy()]
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        edge_index = list(edge_index.T)
        poten_edges = set([tuple(t) for t in poten_edges]) - set([tuple(t) for t in edge_index])
        poten_edges = [list(s) for s in poten_edges]
        poten_edges = torch.as_tensor(poten_edges).T.to(self.device)

        return poten_edges


# %%
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, nfea, args, idx_train, device = 'cuda'):
        super(EstimateAdj, self).__init__()

        self.estimator = GCN(nfea, args.edge_hidden, args.edge_hidden, dropout = 0.0, device = device)
        self.device = device
        self.args = args
        self.representations = 0

    def forward(self, edge_index, features):
        representations = self.estimator(features, edge_index, \
                                         torch.ones([edge_index.shape[1]]).to(self.device).float())
        representations = F.normalize(representations, dim = -1)
        rec_loss = self.reconstruct_loss(edge_index, representations)

        return representations, rec_loss

    def get_estimated_weigths(self, edge_index, representations, origin_w = None):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0, x1), dim = 1)
        estimated_weights = F.relu(output)
        if estimated_weights.shape[0] != 0:
            estimated_weights = torch.where(estimated_weights < self.args.t_small,
                                            torch.tensor(0.0, device = output.device), estimated_weights)
            if origin_w != None:
                estimated_weights = origin_w + estimated_weights * (1 - origin_w)

        return estimated_weights, None

    def reconstruct_loss(self, edge_index, representations):
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index, num_nodes = num_nodes, num_neg_samples = self.args.n_n * num_nodes)
        randn = randn[:, randn[0] < randn[1]]

        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0, neg1), dim = 1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0, pos1), dim = 1)

        rec_loss = (F.mse_loss(neg, torch.zeros_like(neg), reduction = 'sum') \
                    + F.mse_loss(pos, torch.ones_like(pos), reduction = 'sum')) \
                   * num_nodes / (randn.shape[1] + edge_index.shape[1])

        return rec_loss
