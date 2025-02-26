import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from numpy import loadtxt
import scipy.io
from collections import Counter
from itertools import combinations

from CJEMN_utils import classification, cluster_centroids


# Optionally, if you want to use torch device (CPU/GPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device = ", device)
class Embeddings:
    def __init__(self, num_g, adj, numNodes, attributes, numAtt, d, iterr, extraiter,
                 cat_count, countCat, startIndex, weightFactors, sumWeights, clusters, train_idx, test_idx, k, gt):
        """
        Parameters (expected shapes):
          num_g         : int, number of graphs (relations)
          adj           : list of [numNodes x numNodes] numpy arrays (one per graph)
          numNodes      : int, number of nodes
          attributes    : [numNodes x numAtt] array (node attributes and class labels; -1 indicates missing)
          numAtt        : int, total number of attribute+class label columns
          d             : int, embedding dimension
          iterr         : int, maximum iterations
          extraiter     : int, early stopping patience (iterations without improvement)
          cat_count     : int, total number of categories (i.e. total rows in attributeLabelEmbedding)
          countCat      : [numAtt x max_categories] array of category counts per attribute/label
          startIndex    : [numAtt] array, starting index for each attribute in the attribute embedding matrix
          weightFactors : [num_g + numAtt] array of weights for graphs and attributes/class labels
          sumWeights    : [numNodes] array of per-node normalization factors
          clusters      : int, number of clusters (class labels)
          train_idx     : list or array of training node indices
          test_idx      : list or array of test (unlabeled) node indices
        """
        self.num_g = num_g
        self.numNodes = numNodes
        self.d = d
        self.iterr = iterr
        self.extraiter = extraiter
        self.clusters = clusters
        self.kN = k
        self.gt = gt
        
        # Convert inputs to torch tensors (and move to device if desired)
        self.attributeLabelNumber = numAtt
        self.adj = [a.clone().detach().to(device).float() for a in adj]
        self.attributes = attributes.clone().detach().to(device).long()
        self.countCat = countCat.clone().detach().to(device).float()
        self.startIndex = startIndex.clone().detach().to(device).long()
        self.weightFactors = weightFactors.clone().detach().to(device).float()
        self.sumWeights = sumWeights.clone().detach().to(device).float()
        self.train_idx = train_idx.clone().detach().to(device).long()
        self.test_idx = list(test_idx)
        
        # Initialize node embeddings randomly (shape: [numNodes, d])
        self.nodeEmbedding = (torch.rand(numNodes, d, device=device) / 10.0)
        # Allocate attribute/class label embeddings (shape: [cat_count, d])
        self.attributeLabelEmbedding = torch.zeros(cat_count, d, dtype=torch.float32, device=device)
        
        # Vectorized initialization of attribute/class label embeddings:
        self.init_loop()

    def init_loop(self):
        # For node attributes (first part of the columns)
        num_attr = self.attributeLabelNumber - self.clusters
        for k in range(num_attr):
            mask = self.attributes[:, k] > -1  # valid entries
            if mask.sum() == 0:
                continue
            # For each valid node, determine the target row index in attributeLabelEmbedding:
            idx = self.startIndex[k] + self.attributes[mask, k]
            # Compute a scaling factor per node (using countCat for that attribute and its category)
            factors = (1.0 / self.countCat[k, self.attributes[mask, k]]).unsqueeze(1)
            update = self.nodeEmbedding[mask] * factors
            self.attributeLabelEmbedding.index_add_(0, idx, update)
            
        # For class labels (last 'clusters' columns), only use training nodes:
        for k in range(num_attr, self.attributeLabelNumber):
            mask = torch.zeros(self.numNodes, dtype=torch.bool, device=device)
            mask[self.train_idx] = True
            mask = mask & (self.attributes[:, k] > -1)
            if mask.sum() == 0:
                continue
            idx = self.startIndex[k] + self.attributes[mask, k]
            factors = (1.0 / self.countCat[k, self.attributes[mask, k]]).unsqueeze(1)
            update = self.nodeEmbedding[mask] * factors
            self.attributeLabelEmbedding.index_add_(0, idx, update)
    
    def modified_gram_schmidt(self, newCoord):
        # Instead of manual loops, use the fast QR decomposition:
        Q, _ = torch.linalg.qr(newCoord)
        return Q

    def update_node_coordinates(self):
        # Compute new node coordinates as a weighted sum of neighbor contributions and attribute influences.
        newCoord = torch.zeros_like(self.nodeEmbedding)
        # --- Contributions from relation graphs ---
        # For each graph, zero out self-loops and do a matrix multiplication:
        for j in range(self.num_g):
            A = self.adj[j].clone()
            A.fill_diagonal_(0)
            # (A @ nodeEmbedding) yields a [numNodes x d] tensor.
            contrib = torch.matmul(A, self.nodeEmbedding) / self.sumWeights.unsqueeze(1)
            if j < self.num_g - 1:
                newCoord += self.weightFactors[j] * contrib
            else:
                newCoord -= self.weightFactors[j] * contrib
        
        # --- Contributions from node attributes ---
        num_attr = self.attributeLabelNumber - self.clusters
        for k in range(num_attr):
            mask = self.attributes[:, k] > -1
            if mask.sum() > 0:
                idx = self.startIndex[k] + self.attributes[mask, k]
                newCoord[mask] += self.weightFactors[self.num_g + k] * \
                                  self.attributeLabelEmbedding[idx] / self.sumWeights[mask].unsqueeze(1)
        
        # --- Contributions from class labels (only for training nodes) ---
        for k in range(num_attr, self.attributeLabelNumber):
            mask = torch.zeros(self.numNodes, dtype=torch.bool, device=device)
            mask[self.train_idx] = True
            mask = mask & (self.attributes[:, k] > -1)
            if mask.sum() > 0:
                # Use different target indices for category 0 vs. 1.
                mask1 = mask & (self.attributes[:, k] == 1)
                mask0 = mask & (self.attributes[:, k] == 0)
                if mask1.sum() > 0:
                    idx1 = self.startIndex[k] + 1
                    newCoord[mask1] += self.weightFactors[self.num_g + k] * \
                                       self.attributeLabelEmbedding[idx1] / self.sumWeights[mask1].unsqueeze(1)
                if mask0.sum() > 0:
                    idx0 = self.startIndex[k] + 0
                    newCoord[mask0] += self.weightFactors[self.num_g + k] * \
                                       self.attributeLabelEmbedding[idx0] / self.sumWeights[mask0].unsqueeze(1)
        
        # Orthogonalize (using QR) to stabilize the embeddings.
        self.nodeEmbedding = self.modified_gram_schmidt(newCoord)
        return self.nodeEmbedding

    def update_attribute_coordinates(self):
        # Update attribute (and class label) embeddings by aggregating node embeddings.
        newCoord = torch.zeros_like(self.attributeLabelEmbedding)
        num_attr = self.attributeLabelNumber - self.clusters
        for k in range(num_attr):
            mask = self.attributes[:, k] > -1
            if mask.sum() > 0:
                idx = self.startIndex[k] + self.attributes[mask, k]
                factors = (1.0 / self.countCat[k, self.attributes[mask, k]]).unsqueeze(1)
                update = self.nodeEmbedding[mask] * factors
                newCoord.index_add_(0, idx, update)
        for k in range(num_attr, self.attributeLabelNumber):
            mask = torch.zeros(self.numNodes, dtype=torch.bool, device=device)
            mask[self.train_idx] = True
            mask = mask & (self.attributes[:, k] > -1)
            if mask.sum() > 0:
                idx = self.startIndex[k] + self.attributes[mask, k]
                factors = (1.0 / self.countCat[k, self.attributes[mask, k]]).unsqueeze(1)
                update = self.nodeEmbedding[mask] * factors
                newCoord.index_add_(0, idx, update)
        self.attributeLabelEmbedding = newCoord
        return self.attributeLabelEmbedding

    def objective(self):
        # Compute the overall cost as the weighted sum of squared differences.
        cost = torch.zeros(self.num_g + self.attributeLabelNumber, dtype=torch.float32, device=device)
        X = self.nodeEmbedding  # [numNodes, d]
        # --- Graph contributions ---
        # For each graph, we sum over all pairs (j,e) weighted by A[j,e] and normalized by sumWeights[j].
        for i in range(self.num_g):
            A = self.adj[i].clone()
            A.fill_diagonal_(0)
            # diff = X.unsqueeze(1) - X.unsqueeze(0)  # [numNodes, numNodes, d]
            
            # dist_mat = (diff ** 2).sum(dim=2)         # [numNodes, numNodes]
            dist_mat = torch.cdist(X, X, p=2) ** 2

            contrib = (A * dist_mat).sum(dim=1) / self.sumWeights  # [numNodes]
            cost_i = self.weightFactors[i] * contrib.sum()
            if i == self.num_g - 1:
                cost[i] -= cost_i  # negative pairs: subtract cost
            else:
                cost[i] += cost_i
        
        # --- Attribute contributions ---
        num_attr = self.attributeLabelNumber - self.clusters
        for j in range(num_attr):
            mask = self.attributes[:, j] > -1
            if mask.sum() > 0:
                idx = self.startIndex[j] + self.attributes[mask, j]
                diff = X[mask] - self.attributeLabelEmbedding[idx]
                cost[self.num_g + j] += self.weightFactors[self.num_g + j] * (diff ** 2).sum()
        
        # --- Class label contributions ---
        for j in range(num_attr, self.attributeLabelNumber):
            mask = torch.zeros(self.numNodes, dtype=torch.bool, device=device)
            mask[self.train_idx] = True
            mask = mask & (self.attributes[:, j] > -1)
            if mask.sum() > 0:
                idx = self.startIndex[j] + self.attributes[mask, j]
                diff = X[mask] - self.attributeLabelEmbedding[idx]
                cost[self.num_g + j] += self.weightFactors[self.num_g + j] * (diff ** 2).sum()
        
        return cost.sum().item()
    

    def pseudo_labeling(self, iteration):
        # Every 5 iterations, add pseudo-labeled nodes to the training set.
        # First, extract “centroids” from the class label embeddings.
        # (Assuming that the last clusters*2 rows of attributeLabelEmbedding correspond to class label embeddings
        #  and that every second row (starting at index 1) is the “correct” one.)
        labelEmb = self.attributeLabelEmbedding[-(self.clusters * 2):, :]  # [clusters*2, d]
        
        centroids = labelEmb[1::2]  # [clusters, d]
       
        k = self.kN + int(iteration/5) 
        # Compute distances between centroids and node embeddings
        dists = torch.cdist(centroids, self.nodeEmbedding, p=2)  # [clusters, numNodes]
        # Get the k nearest nodes for each centroid
        _, indices = torch.topk(dists, k, largest=False)
        pseudo_nodes = []
        pseudo_labels = []
        
        for c in range(self.clusters):
            count = 0
            for idx in indices[c].tolist():
                if idx not in self.train_idx:
                    count += 1
                    if count <= k:
                        pseudo_nodes.append(idx)
                        pseudo_labels.append(c)
        
        # Update attribute information for pseudo-labeled nodes.
        # (Here we follow different rules based on the number of clusters.)
        base_index = -self.clusters  # This remains constant for all nodes
        for node, label in zip(pseudo_nodes, pseudo_labels):
           self.attributes[node, base_index + label] = 1
        
        # Update the training set (concatenate new pseudo-labeled nodes)
        new_train = torch.cat((self.train_idx, torch.tensor(pseudo_nodes, dtype=torch.long, device=device)))
        self.train_idx = torch.sort(new_train)[0]
        
        # Optionally, update negative-edge graph based on pseudo-label differences.
        # (Here we loop over the relatively few pseudo-labeled nodes.)
        A_neg = self.adj[-1]
        
        pseudo_nodes = np.array(pseudo_nodes)
        pseudo_labels = np.array(pseudo_labels)
        for i, j in combinations(range(len(pseudo_nodes)), 2):
            if pseudo_labels[i] != pseudo_labels[j]:
                # Check if an edge exists between the two nodes in any of the other graphs.
                if any(
                    self.adj[g][pseudo_nodes[i], pseudo_nodes[j]] == 1 or
                    self.adj[g][pseudo_nodes[j], pseudo_nodes[i]] == 1
                    for g in range(self.num_g - 1)
                ):
                    A_neg[pseudo_nodes[i], pseudo_nodes[j]] = 1
                    A_neg[pseudo_nodes[j], pseudo_nodes[i]] = 1
                
        self.adj[-1] = A_neg


    def run(self):
        minCost = self.objective()
        threshold = 0.001
        iteration = 0
        minCostID = 0
        best_nodeEmbedding = self.nodeEmbedding.clone()
        best_attributeEmbedding = self.attributeLabelEmbedding.clone()
        
        while iteration < self.iterr + self.extraiter:
            if iteration>0 and iteration % 5 == 0:
                self.pseudo_labeling(iteration)

            self.update_node_coordinates()
            self.update_attribute_coordinates()
            actualCost = self.objective()
            print("Iteration =", iteration, "actualCost =", actualCost, 
                  "minCost =", minCost, "minCostID =", minCostID)
            
            if actualCost < minCost - threshold:
                minCost = actualCost
                minCostID = iteration
                best_nodeEmbedding = self.nodeEmbedding.clone()
                best_attributeEmbedding = self.attributeLabelEmbedding.clone()
            else:
                if minCostID + self.extraiter + self.iterr <= iteration:
                    self.nodeEmbedding = best_nodeEmbedding
                    self.attributeLabelEmbedding = best_attributeEmbedding
                    break
            
            
            
            if iteration > 1000:
                self.nodeEmbedding = best_nodeEmbedding
                self.attributeLabelEmbedding = best_attributeEmbedding
                break
            iteration += 1
        
        print("Final iteration:", iteration, "; Best iteration:", minCostID)
        print("Iteration =", iteration, "actualCost =", actualCost,
              "minCost =", minCost, "minCostID =", minCostID)
        return self.nodeEmbedding, self.attributeLabelEmbedding

class ReadGraph:
    # Default parameters (will be overwritten in each reader method)
    graphs = 1            # total number of graphs/layers
    nodes = 0             # total number of nodes
    atts = 0              # total number of node attributes (or labels)
    dim = 32              # embedding dimensionality
    iterations = 10       # number of iterations
    extraiter = 2         # extra iterations for early stopping
    clusters = 3          # number of clusters
    samples = 1           # number of negative–sample repetitions
    k = 1                 # number of neighbors for kNN
    dim_cluster = 2       # clustering dimension
    name = ''             # dataset name
    
    # Data holders (all will be converted to torch tensors)
    node_attr = None      # node features (or attributes)
    adj = []              # list of adjacency matrices
    gt = None             # ground truth labels (or matrix)
    train_idx = None      # training indices (torch tensor)
    test_idx = None       # testing indices (torch tensor)
    val_idx = None        # validation indices (torch tensor)
    labels = None         # raw label data

    def __init__(self, name):
        self.name = name.lower()
        self.readData()
        self.printD()

    def load_adj_neg(self, num_nodes, samples):
        # Vectorized negative sampling:
        # For each node, sample "samples" random indices.
        row = torch.arange(num_nodes).unsqueeze(1).repeat(1, samples).flatten()
        col = torch.randint(0, num_nodes, (num_nodes * samples,))
        mask = row != col
        row = row[mask]
        col = col[mask]
        # Create symmetric pairs
        new_row = torch.cat([row, col])
        new_col = torch.cat([col, row])
        data = torch.ones(new_row.size(0), dtype=torch.float32)
        # Build a sparse tensor and convert to dense
        adj_neg = torch.sparse_coo_tensor(torch.stack([new_row, new_col]), data, (num_nodes, num_nodes))
        adj_neg = adj_neg.to_dense()
        # Zero out the diagonal and take only lower–triangular values (if desired)
        adj_neg.fill_diagonal_(0)
        adj_neg = torch.tril(adj_neg)
        return adj_neg

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features.todense()


    def readACM(self):
        # Set ACM-specific parameters
        self.graphs = 3
        self.dim = 32
        self.clusters = 3
        self.iterations = 200
        self.extraiter = 40
        self.dim_cluster = 2
        self.k = 60

        # Set file paths (adjust as needed)
        base_path = 'data/ACM/'
        
        # Load adjacency matrices from .mat files and convert to torch tensors
        adj1 = torch.tensor(scipy.io.loadmat(base_path + "PAP.mat")['PAP'], dtype=torch.float32, device=device)
        adj2 = torch.tensor(scipy.io.loadmat(base_path + "PLP.mat")['PLP'], dtype=torch.float32, device=device)
        
        # Set the number of nodes
        self.nodes = adj1.shape[0]
        
        # Load labels and features; convert to torch tensors
        self.labels = torch.tensor(scipy.io.loadmat(base_path + 'label.mat')['label'], dtype=torch.float32, device=device)
        self.node_attr = torch.tensor(scipy.io.loadmat(base_path + "feature.mat")['feature'], dtype=torch.float32, device=device)
        self.atts = self.node_attr.shape[1]
        
        # Ground truth (assumed to be stored as a text file)
        self.gt = torch.tensor(np.loadtxt(base_path + "ground_truth.txt"), dtype=torch.float32, device=device)
        
        # Load training, testing, and validation indices
        train_ids = np.sort(loadtxt(base_path + 'train_ids_1.txt')).astype(int)
        test_ids  = np.sort(loadtxt(base_path + 'test_ids_1.txt')).astype(int)
        val_ids   = np.sort(loadtxt(base_path + 'val_ids_1.txt')).astype(int)
        self.train_idx = torch.tensor(train_ids, dtype=torch.long, device=device)
        self.test_idx  = torch.tensor(test_ids, dtype=torch.long, device=device)
        self.val_idx   = torch.tensor(val_ids, dtype=torch.long, device=device)
        
        # Build a ground-truth matrix for training nodes in a fully vectorized manner
        gt_train = torch.zeros((self.nodes, self.clusters), dtype=self.labels.dtype, device=device)
        # Use advanced indexing: assume self.labels is shaped [num_nodes, clusters]
        gt_train[self.train_idx] = self.labels[self.train_idx]
        
        # Concatenate features with training ground truth (along feature dimension)
        self.node_attr = torch.cat([self.node_attr, gt_train], dim=1)
        self.atts = self.node_attr.shape[1]
        
        # Load negative samples using the vectorized function
        # neg_adj = self.load_adj_neg(self.nodes, self.samples)
        # neg_adj = neg_adj  # already a torch tensor from load_adj_neg
        neg_adj = torch.zeros((self.nodes, self.nodes), dtype=torch.float32, device=device)

        train_mask = torch.zeros(self.nodes, dtype=torch.bool, device=device)
        train_mask[self.train_idx] = True
        train_pair = train_mask.unsqueeze(0) & train_mask.unsqueeze(1)

        # diff_mask = ((self.gt.unsqueeze(0) - self.gt.unsqueeze(1)).abs().sum(dim=2)) > 0
        # Ensure self.gt is at least 2D:
        if self.gt.dim() == 1:
            gt = self.gt.unsqueeze(1)  # shape becomes [num_nodes, 1]
        else:
            gt = self.gt

        # Now compute the difference mask.
        # Using dim=-1 (or dim=1 in this case) to sum over the last dimension.
        diff_mask = ((gt.unsqueeze(0) - gt.unsqueeze(1)).abs().sum(dim=-1)) > 0
        adj_condition = ((adj1 == 1) | (adj2 == 1))
        condition = train_pair & diff_mask & adj_condition
        neg_adj[condition] = 1

        # Set the adjacency list (keeping the order: positive relations then negative)
        self.adj = [adj1, adj2, neg_adj]

    def readIMDB(self):
        self.graphs = 3
        self.dim = 32
        self.clusters = 3
        self.iterations = 200
        self.extraiter = 40
        self.dim_cluster = 2
        self.k = 36
        
        base_path = 'data/IMDB/'
        data = scipy.io.loadmat(base_path + "imdb.mat")
        # Convert loaded matrices to torch tensors
        adj1 = torch.tensor(data['MDM'], dtype=torch.float32, device=device)
        adj2 = torch.tensor(data['MAM'], dtype=torch.float32, device=device)
        self.nodes = adj1.shape[0]
        self.labels = torch.tensor(data['label'], dtype=torch.float32, device=device)
        self.node_attr = torch.tensor(data['feature'], dtype=torch.float32, device=device)
        self.atts = self.node_attr.shape[1]
        self.gt = torch.tensor(np.loadtxt(base_path + "ground_truth.txt"), dtype=torch.float32, device=device)
        
        
        train_ids = np.sort(loadtxt(base_path + 'train_ids_1.txt')).astype(int)
        test_ids  = np.sort(loadtxt(base_path + 'test_ids_1.txt')).astype(int)
        val_ids   = np.sort(loadtxt(base_path + 'val_ids_1.txt')).astype(int)
        self.train_idx = torch.tensor(train_ids, dtype=torch.long, device=device)
        self.test_idx  = torch.tensor(test_ids, dtype=torch.long, device=device)
        self.val_idx   = torch.tensor(val_ids, dtype=torch.long, device=device)
        
        gt_train = torch.zeros((self.nodes, self.clusters), dtype=self.labels.dtype, device=device)
        gt_train[self.train_idx] = self.labels[self.train_idx]
        self.node_attr = torch.cat([self.node_attr, gt_train], dim=1)
        self.atts = self.node_attr.shape[1]
        
        neg_adj = torch.zeros((self.nodes, self.nodes), dtype=torch.float32, device=device)
        
        train_mask = torch.zeros(self.nodes, dtype=torch.bool, device=device)
        train_mask[self.train_idx] = True
        train_pair = train_mask.unsqueeze(0) & train_mask.unsqueeze(1)

        if self.gt.dim() == 1:
            gt = self.gt.unsqueeze(1) 
        else:
            gt = self.gt

        diff_mask = ((gt.unsqueeze(0) - gt.unsqueeze(1)).abs().sum(dim=-1)) > 0
        adj_condition = ((adj1 == 1) | (adj2 == 1))
        condition = train_pair & diff_mask & adj_condition
        neg_adj[condition] = 1
      
        self.adj = [adj1, adj2, neg_adj]

    def readDBLP(self):
        self.graphs = 5
        self.dim = 128
        self.clusters = 3
        self.iterations = 100
        self.extraiter = 20
        self.dim_cluster = 2
        self.k = 15

        base_path = 'data/DBLP/'
        # Load .mat files and immediately convert to torch tensors.
        adj1 = torch.tensor(scipy.io.loadmat(base_path + "coauthor_mat.mat")['coauthor_mat'], dtype=torch.float32, device=device)
        adj2 = torch.tensor(scipy.io.loadmat(base_path + "apnet_mat.mat")['apnet_mat'], dtype=torch.float32, device=device)
        adj3 = torch.tensor(scipy.io.loadmat(base_path + "citation_mat.mat")['citation_mat'], dtype=torch.float32, device=device)
        adj4 = torch.tensor(scipy.io.loadmat(base_path + "cocitation_mat.mat")['cocitation_mat'], dtype=torch.float32, device=device)
        # Optionally add self-loops:
        I = torch.eye(adj1.size(0), device=device)
        adj1 = adj1 + I
        adj2 = adj2 + I
        adj3 = adj3 + I
        adj4 = adj4 + I
        self.nodes = adj1.size(0)
        self.gt = torch.tensor(np.loadtxt(base_path + "ground_truth.txt"), dtype=torch.float32, device=device)
        # If node attributes are not available, you can create a zero tensor:
        self.node_attr = torch.zeros((self.nodes, int(torch.max(self.gt).item()) + 1), device=device )
        self.atts = self.node_attr.size(1)
        
        train_ids = np.sort(loadtxt(base_path + 'train_ids.txt')).astype(int)
        test_ids  = np.sort(loadtxt(base_path + 'test_ids.txt')).astype(int)
        val_ids   = np.sort(loadtxt(base_path + 'val_ids.txt')).astype(int)
        self.train_idx = torch.tensor(train_ids, dtype=torch.long, device=device)
        self.test_idx  = torch.tensor(test_ids, dtype=torch.long, device=device)
        self.val_idx   = torch.tensor(val_ids, dtype=torch.long, device=device)
        
        # For training nodes, update node_attr with one-hot ground truth:
        onehot = F.one_hot(self.gt.to(torch.int64), num_classes=self.atts).float()
        self.node_attr[self.train_idx] = onehot[self.train_idx]
        
        neg_adj = torch.zeros((self.nodes, self.nodes), dtype=torch.float32, device=device)

        train_mask = torch.zeros(self.nodes, dtype=torch.bool, device=device)
        train_mask[self.train_idx] = True
        train_pair = train_mask.unsqueeze(0) & train_mask.unsqueeze(1)

        if self.gt.dim() == 1:
            gt = self.gt.unsqueeze(1)  
        else:
            gt = self.gt

        diff_mask = ((gt.unsqueeze(0) - gt.unsqueeze(1)).abs().sum(dim=-1)) > 0
        adj_condition = ((adj1 == 1) | (adj2 == 1) | (adj3 == 1) | (adj4 == 1))
        condition = train_pair & diff_mask & adj_condition
        neg_adj[condition] = 1

        self.adj = [adj1, adj2, adj3, adj4, neg_adj.to(device)]

    def readFREEBASE(self):
        self.graphs = 4
        self.dim = 16
        self.clusters = 3
        self.iterations = 200
        self.extraiter = 40
        self.dim_cluster = 2
        self.k = 3

        base_path = 'data/FREEBASE/'
        data = scipy.io.loadmat(base_path + 'freebase.mat')
        # Convert sparse matrices to dense and then to torch tensors:
        adj1 = torch.tensor(data['mam'].todense(), dtype=torch.float32)
        adj2 = torch.tensor(data['mdm'].todense(), dtype=torch.float32)
        adj3 = torch.tensor(data['mwm'].todense(), dtype=torch.float32)
        self.nodes = adj1.size(0)
        self.labels = torch.tensor(data['label'], dtype=torch.float32)
        self.gt = torch.tensor(np.loadtxt(base_path + "ground_truth.txt"), dtype=torch.float32)
        self.node_attr = torch.zeros((self.nodes, int(torch.max(self.gt).item()) + 1))
        self.atts = self.node_attr.size(1)
        
        train_ids = np.sort(loadtxt(base_path + 'train_ids_.txt')).astype(int)
        test_ids  = np.sort(loadtxt(base_path + 'test_ids_.txt')).astype(int)
        val_ids   = np.sort(loadtxt(base_path + 'val_ids_.txt')).astype(int)
        self.train_idx = torch.tensor(train_ids, dtype=torch.long)
        self.test_idx  = torch.tensor(test_ids, dtype=torch.long)
        self.val_idx   = torch.tensor(val_ids, dtype=torch.long)

        onehot = F.one_hot(self.gt.to(torch.int64), num_classes=self.atts).float()
        self.node_attr[self.train_idx] = onehot[self.train_idx]

        neg_adj = torch.zeros((self.nodes, self.nodes), dtype=torch.float32)
        
        train_mask = torch.zeros(self.nodes, dtype=torch.bool)
        train_mask[self.train_idx] = True
        train_pair = train_mask.unsqueeze(0) & train_mask.unsqueeze(1)

        if self.gt.dim() == 1:
            gt = self.gt.unsqueeze(1)  
        else:
            gt = self.gt

        diff_mask = ((gt.unsqueeze(0) - gt.unsqueeze(1)).abs().sum(dim=-1)) > 0
        adj_condition = ((adj1 == 1) | (adj2 == 1) | (adj3 == 1))
        condition = train_pair & diff_mask & adj_condition
        neg_adj[condition] = 1    

        self.adj = [adj1, adj2, adj3, neg_adj.to(device)]

    def readFLICKR(self):
        self.graphs = 3
        self.dim = 64
        self.clusters = 7
        self.iterations = 100
        self.extraiter = 20
        self.dim_cluster = 2
        self.k = 147

        base_path = 'data/FLICKR/'
        adj1 = torch.tensor(scipy.io.loadmat(base_path + "layer0mat.mat")['layer0mat'], dtype=torch.float32)
        adj2 = torch.tensor(scipy.io.loadmat(base_path + "layer1mat.mat")['layer1mat'], dtype=torch.float32)
        self.nodes = adj1.size(0)
        self.gt = torch.tensor(np.loadtxt(base_path + "ground_truth.txt"), dtype=torch.float32)
        
        # Since FLICKR has no node features, we use a zero matrix with one column per cluster.
        self.node_attr = torch.zeros((self.nodes, int(torch.max(self.gt).item()) + 1))
        self.atts = self.node_attr.size(1)
        
        train_ids = np.sort(loadtxt(base_path + 'train_ids_1.txt')).astype(int)
        test_ids  = np.sort(loadtxt(base_path + 'test_ids_1.txt')).astype(int)
        val_ids   = np.sort(loadtxt(base_path + 'val_ids_1.txt')).astype(int)
        self.train_idx = torch.tensor(train_ids, dtype=torch.long)
        self.test_idx  = torch.tensor(test_ids, dtype=torch.long)
        self.val_idx   = torch.tensor(val_ids, dtype=torch.long)
        
        onehot = F.one_hot(self.gt.to(torch.int64), num_classes=self.atts).float()
        self.node_attr[self.train_idx] = onehot[self.train_idx]

        neg_adj = torch.zeros((self.nodes, self.nodes), dtype=torch.float32)
        
        train_mask = torch.zeros(self.nodes, dtype=torch.bool)
        train_mask[self.train_idx] = True
        train_pair = train_mask.unsqueeze(0) & train_mask.unsqueeze(1)

        if self.gt.dim() == 1:
            gt = self.gt.unsqueeze(1)  
        else:
            gt = self.gt

        diff_mask = ((gt.unsqueeze(0) - gt.unsqueeze(1)).abs().sum(dim=-1)) > 0
        adj_condition = ((adj1 == 1) | (adj2 == 1))
        condition = train_pair & diff_mask & adj_condition
        neg_adj[condition] = 1        

        self.adj = [adj1, adj2, neg_adj]

    def readData(self):
        # Dispatch to the proper reader based on dataset name
        if self.name == 'imdb':
            self.readIMDB()
        elif self.name == 'acm':
            self.readACM()
        elif self.name == 'dblp':
            self.readDBLP()
        elif self.name == 'freebase':
            self.readFREEBASE()
        elif self.name == 'flickr':
            self.readFLICKR()
        elif self.name == 'mag':
            self.readMAG()
        else:
            raise ValueError("Unknown dataset name.")

    def printD(self):
        print("# Graphs      =", self.graphs)
        print("# Nodes       =", self.nodes)
        print("# Attributes  =", self.atts)
        print("# Dim         =", self.dim)
        print("# Negative Samples =", torch.sum(self.adj[-1]).item())
        print("# Iterations  =", self.iterations)
        print("# Extra iterations  =", self.extraiter)

class MixedSpectral:
    def __init__(self, num_g, adj, num_nodes, weighted, attributes, numAtt, clusters):
        """
        Parameters:
          num_g      : number of graphs/layers
          adj        : list of adjacency matrices (each as a NumPy array of shape [num_nodes, num_nodes])
          num_nodes  : total number of nodes
          weighted   : bool flag (unused here, but kept for compatibility)
          attributes : NumPy array of shape [num_nodes, numAtt] (categorical; use -1 for missing)
          numAtt     : number of node attributes (and class labels)
          clusters   : number of clusters
        """
        self.num_g = num_g
        self.weighted = weighted
        # Convert inputs to torch tensors
        self.attributes = attributes.clone().detach().to(device).long()  # shape: [num_nodes, numAtt]
        self.numAtt = numAtt
        # Convert each adjacency to torch.float tensor
        self.adj = [a.clone().detach().to(device).float() for a in adj]

        self.clusters = clusters
        self.num_objects = num_nodes

        # These will be computed in computeWeights:
        self.countCat = {}    # dictionary: attribute index -> Counter of categories
        self.num_cat = 0
        self.startIndex = None  # starting index for each attribute in a combined embedding
        self.weightFactors = None  # weight factors (for graphs and attributes)
        self.sumWeights = None

        # First, compute the weights (which in turn computes countCat, startIndex, and sumWeights)
        self.computeWeights()

        # (Optionally, you might want to build a countAtt tensor; here we do it in computeNumCat_CatAtt)
        self.computeNumCat_CatAtt()

        # Now compute the coefficients
        self.computeCoefficients()
        self.computeGraphCoefficients()
        self.computeAttributeCoefficients()
        self.finalUpdate()

    def computeCoefficients(self):
        # We want to compute a coefficient matrix of shape [num_objects, num_g + numAtt]
        # For the graph part, coefficient[i, j] = (# of nonzero entries in row i of adj[j]) * weightFactors[j]
        # For the attribute part, if attributes[i,k] > -1, then coefficient[i, num_g+k] = weightFactors[num_g+k].
        coeff = torch.zeros((self.num_objects, self.num_g + self.numAtt), dtype=torch.float32, device=device)

        # --- Graph coefficients: vectorized over graphs and nodes ---
        # Stack all adjacency matrices to a tensor of shape [num_g, num_objects, num_objects]
        adj_stack = torch.stack(self.adj, dim=0)
        # Count nonzero entries per node (row) for each graph: shape [num_g, num_objects]
        counts = (adj_stack > 0).sum(dim=2).float()
        # Multiply each graph’s count by its weightFactor (first num_g entries)
        wf_graph = self.weightFactors[:self.num_g].unsqueeze(1)  # shape [num_g, 1]
        coeff_graph = counts * wf_graph  # shape [num_g, num_objects]
        # Transpose so that each row corresponds to a node
        coeff[:, :self.num_g] = coeff_graph.transpose(0, 1)

        # --- Attribute coefficients ---
        # For each attribute, if attribute > -1 then coefficient = weightFactors[num_g + k]
        mask = (self.attributes > -1).float()  # shape [num_objects, numAtt]; 1 if valid, else 0
        wf_attr = self.weightFactors[self.num_g:]  # shape [numAtt]
        coeff_attr = mask * wf_attr.unsqueeze(0)   # broadcast to [num_objects, numAtt]
        coeff[:, self.num_g:] = coeff_attr

        self.coefficients = coeff

    def computeGraphCoefficients(self):
        # Compute graph coefficients for each graph and node.
        # For each graph j and node i, value = (# nonzero in row i of adj[j]) * weightFactors[j]
        adj_stack = torch.stack(self.adj, dim=0)  # shape: [num_g, num_objects, num_objects]
        counts = (adj_stack > 0).sum(dim=2).float()  # shape: [num_g, num_objects]
        wf_graph = self.weightFactors[:self.num_g].unsqueeze(1)
        self.graphCoefficients = counts * wf_graph  # shape: [num_g, num_objects]

    def computeAttributeCoefficients(self):
        # For each attribute j and node i, if attribute is valid then
        # attributeCoefficient[j, i] = weightFactors[num_g + j]
        mask = (self.attributes > -1).float()  # shape: [num_objects, numAtt]
        wf_attr = self.weightFactors[self.num_g:]  # shape: [numAtt]
        # We want the result in shape [numAtt, num_objects]
        self.attributeCoefficients = (mask * wf_attr.unsqueeze(0)).transpose(0, 1)

    def computeWeights(self):
        # Convert attributes to NumPy (for counting categories)
        attr_np = self.attributes.cpu().numpy()  # shape: [num_objects, numAtt]
        self.countCat = {}
        self.num_cat = 0
        for i in range(self.numAtt):
            c = Counter(attr_np[:, i])
            self.countCat[i] = c
            self.num_cat += len(c.keys())
        # Compute starting indices for attributes (cumulative sum of number of categories)
        startIndex = np.zeros(self.numAtt, dtype=np.float32)
        placesBefore = 0
        for i in range(self.numAtt):
            if i > 0:
                placesBefore += len(self.countCat[i-1].keys())
            startIndex[i] = placesBefore
        self.startIndex = torch.tensor(startIndex, dtype=torch.float32, device=device)

        # Compute overall weights (for graphs and attributes)
        overallWeight = np.zeros(self.num_g + self.numAtt, dtype=np.float32)
        # For each graph, overallWeight is the total number of nonzero entries
        for i in range(self.num_g):
            overallWeight[i] = (self.adj[i] > 0).sum().item()
        # For each attribute, overallWeight is the sum of counts of all categories
        for i in range(self.numAtt):
            overallWeight[self.num_g + i] = sum(self.countCat[i].values())
        # Additional adjustments (as in your original code): add count for category 1
        for i in range(self.numAtt):
            cnt = self.countCat[i].get(1, 0)
            if i < self.numAtt - self.clusters:
                overallWeight[self.num_g + i] += cnt
            else:
                overallWeight[self.num_g + i] += cnt * self.clusters
        # Find the maximum weight and use it to normalize all weights
        maxIndex = np.argmax(overallWeight)
        maxWeight = overallWeight[maxIndex]
        weightFactors = maxWeight / overallWeight  # elementwise division yields a vector
        self.weightFactors = torch.tensor(weightFactors, dtype=torch.float32, device=device)

        # Compute sumWeights for each node:
        # Graph part: for each graph, count nonzeros per node then weight.
        graph_sum = torch.zeros(self.num_objects, dtype=torch.float32, device=device)
        for j in range(self.num_g):
            graph_sum += (self.adj[j] > 0).sum(dim=1).float() * self.weightFactors[j]
        # Attribute part: add for each node, for each attribute if valid, add corresponding weightFactor.
        mask = (self.attributes > -1).float()  # shape: [num_objects, numAtt]
        attr_sum = (mask * self.weightFactors[self.num_g:].unsqueeze(0)).sum(dim=1)
        self.sumWeights = graph_sum + attr_sum

    def computeNumCat_CatAtt(self):
        # Build a countAtt tensor from countCat.
        # For each attribute i, countAtt[i, :] holds the counts for each category (in sorted order).
        max_cats = max(len(v) for v in self.countCat.values())
        countAtt = torch.zeros((self.numAtt, max_cats), dtype=torch.float32, device=device)
        for i in range(self.numAtt):
            # Sort the categories for consistency
            cats = sorted(self.countCat[i].keys())
            for j, cat in enumerate(cats):
                countAtt[i, j] = self.countCat[i][cat]
        self.countAtt = countAtt
        # For example, one can set the total number of categorical values as:
        self.num_cat = countAtt.shape[0] * countAtt.shape[1]
        # Optionally, remove any negative categories from countCat (if present)
        for i in range(self.numAtt):
            if -1 in self.countCat[i]:
                del self.countCat[i][-1]
    
    def finalUpdate(self):
        # Update the final weightFactors for the negative sampling layer, and class labels
        value = 1
        new_value = torch.tensor([value], device=device)

        self.weightFactors = torch.cat([self.weightFactors[:self.num_g], new_value, self.weightFactors[self.num_g:]])

        count = np.array(list(self.countCat.items()))
        
        w_tot = 0
        for i in range(1, self.clusters+1):
            w_tot += count[-i][1][1]

        for i in range(1, self.clusters+1):
            self.weightFactors[-i] = w_tot/ count[-i][1][1]


graphName = sys.argv[1]

dataset = ReadGraph(graphName)
num_g = dataset.graphs
graphs = dataset.graphs
node_attr = dataset.node_attr
num_atts = dataset.atts
num_nodes = dataset.nodes
adj = dataset.adj
clusters = dataset.clusters
dim = dataset.dim
iterations = dataset.iterations
extra = dataset.extraiter
train_idx = dataset.train_idx
test_idx = dataset.test_idx
dim_cluster = dataset.dim_cluster
k = dataset.k
gt = dataset.gt


m = MixedSpectral(graphs-1, adj[:graphs-1], num_nodes, False, node_attr, num_atts, clusters)
startIndex = m.startIndex
sumWeights = m.sumWeights
weightFactors = m.weightFactors
num_cat = m.num_cat
countAtt = m.countAtt

start = time.time()

embeddings = Embeddings(graphs, adj, num_nodes, node_attr, num_atts, dim, iterations,
                            extra, num_cat, countAtt, startIndex, weightFactors,
                            sumWeights, clusters, train_idx, test_idx, k, gt)

nodeEmb, attEmb = embeddings.run()
end = time.time()

print("Time : ", end-start)

test_ids = dataset.test_idx.cpu().numpy()
val_ids = dataset.val_idx.cpu().numpy()
train_ids = dataset.train_idx.cpu().numpy()
gt = dataset.gt.cpu().numpy()

print("\nNode Classification Task:")
ma_ac, mi_ac = classification(nodeEmb.cpu().numpy(), gt, test_ids, val_ids, train_ids)  

print("\nNode Clustering Task: ")
nmi1, ari1, ami1, dim1, nmi2, ari2, ami2, dim2 = cluster_centroids(nodeEmb.cpu().numpy(), attEmb.cpu().numpy(), gt, clusters,  test_ids)
print()
formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
fh = open("results/" + dataset.name.upper() + "_CJEMN.txt", "a")
fh.write('%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f' % (ma_ac, mi_ac, nmi1, ari1, ami1, dim1, nmi2, ari2, ami2, dim2, k))
fh.write('\r\n')
fh.write('---------------------------------------------------------------------------------------------------')
fh.write('\r\n')
fh.flush()
fh.close()