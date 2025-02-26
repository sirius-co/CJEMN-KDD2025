import torch 
import torch.nn.functional as F
import numpy as np
from numpy import loadtxt



from sklearn.cluster import KMeans
from sklearn.metrics import *
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, f1_score
from sklearn.linear_model import LogisticRegressionCV

def classification(X, y, test_ids, val_ids, train_ids):
    # Convert indices to integer NumPy arrays for easy indexing.
    train_ids = np.array(train_ids, dtype=int)
    val_ids = np.array(val_ids, dtype=int)
    test_ids = np.array(test_ids, dtype=int)
    
    # Build datasets using NumPy advanced indexing.
    X_train, y_train = X[train_ids], np.array(y)[train_ids]
    X_val, y_val     = X[val_ids],   np.array(y)[val_ids]
    X_test, y_test   = X[test_ids],  np.array(y)[test_ids]
    
    best_val_mi_f1 = 0.0
    best_test_ma_f1 = best_test_mi_f1 = 0.0

    # Initialize classifier with cross-validation.
    clf = LogisticRegressionCV(max_iter=3000, cv=10, class_weight='balanced', solver='sag')
    
    # Repeat training to potentially improve performance on the validation set.
    for _ in range(10):
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        
        # Compute validation scores.
        val_ma_f1 = f1_score(y_val, y_val_pred, average="macro")
        val_mi_f1 = f1_score(y_val, y_val_pred, average="micro")
        
        # If the micro F1 improves, update our best test metrics.
        if val_mi_f1 > best_val_mi_f1:
            best_val_mi_f1 = val_mi_f1
            y_test_pred = clf.predict(X_test)
            best_test_ma_f1 = f1_score(y_test, y_test_pred, average="macro")
            best_test_mi_f1 = f1_score(y_test, y_test_pred, average="micro")
    
    print(f"MA_F1: {best_test_ma_f1}, MI_F1: {best_test_mi_f1}")
    
    return best_test_ma_f1, best_test_mi_f1



def run_kmeans(X, y, k, test_ids):
    # Convert inputs from torch tensors to NumPy arrays if needed.
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    test_ids = np.array(test_ids, dtype=int)
    X_test = X[test_ids]
    y_test = y[test_ids]
    
    estimator = KMeans(n_clusters=k, n_init=10)
   
    best_nmi = 0
    best_ari = 0 
    best_ami = 0
    best_dim = 2

    # Loop over different numbers of dimensions (features) to use.
    for dim in range(2, X_test.shape[1]):
        X_subset = X_test[:, :dim]
        estimator.fit(X_subset)
        y_pred = estimator.predict(X_subset)
        
        # If using cuML, y_pred may be a GPU array; convert it to a NumPy array.
        if hasattr(y_pred, "get"):
            y_pred = y_pred.get()
            
        nmi = normalized_mutual_info_score(y_test, y_pred, average_method='arithmetic')
        ari = adjusted_rand_score(y_test, y_pred)
        ami = adjusted_mutual_info_score(y_test, y_pred)
        
        if nmi > best_nmi:
            best_nmi = nmi
            best_ari = ari
            best_ami = ami
            best_dim = dim

    # print(f"NMI: {best_nmi}, ARI: {best_ari}, AMI: {best_ami}, DIM: {best_dim}")
    return best_nmi, best_ari, best_ami, best_dim

    

def run_kmeans_test_centroid(X, y, k, centroids, test_ids):
    # Convert torch tensors to NumPy arrays if necessary.
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(centroids, torch.Tensor):
        centroids = centroids.detach().cpu().numpy()
    
    # Use advanced indexing to get test data.
    test_ids = np.array(test_ids, dtype=int)
    X_test = X[test_ids]
    y_test = np.array(y)[test_ids]
    
    KMeansEstimator = KMeans
   
    best_nmi = 0
    best_ari = 0 
    best_ami = 0
    best_dim = 2

    # Loop over dimensions from 2 to the number of columns in centroids.
    for dim in range(2, centroids.shape[1]):
        # Use only the first 'dim' features from both the test set and centroids.
        init_centroids = centroids[:, :dim]
        X_subset = X_test[:, :dim]
        
        # Initialize KMeans with the provided centroids and fit on the test subset.
        estimator = KMeansEstimator(n_clusters=k, init=init_centroids, n_init=1)
        estimator.fit(X_subset)
        y_pred = estimator.predict(X_subset)
        
        # If using cuML, convert GPU array to NumPy.
        if hasattr(y_pred, "get"):
            y_pred = y_pred.get()
        
        # Compute clustering metrics.
        nmi = normalized_mutual_info_score(y_test, y_pred, average_method='arithmetic')
        ari = adjusted_rand_score(y_test, y_pred)
        ami = adjusted_mutual_info_score(y_test, y_pred)
        
        if nmi > best_nmi:
            best_nmi = nmi
            best_ari = ari
            best_ami = ami
            best_dim = dim

    # print(f"NMI: {best_nmi}, ARI: {best_ari}, AMI: {best_ami}, DIM: {best_dim}")
    return best_nmi, best_ari, best_ami, best_dim



def cluster_centroids(nodeEmb, attEmb, gt, clusters, test_ids):
    # Convert torch tensors to NumPy arrays if needed.
    if isinstance(nodeEmb, torch.Tensor):
        nodeEmb = nodeEmb.detach().cpu().numpy()
    if isinstance(attEmb, torch.Tensor):
        attEmb = attEmb.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    
    test_ids = np.array(test_ids, dtype=int)
    
    # Extract the last (clusters*2) rows from attEmb
    labelEmb = attEmb[-(clusters * 2):, :]
    # Select every second row (starting at index 1) to form the centroids.
    centroids = labelEmb[1::2].copy()
    
    # Run clustering without using provided centroids.
    nmi1, ari1, ami1, dim1 = run_kmeans(nodeEmb, gt, clusters, test_ids)
    # Run clustering with the pre-computed centroids.
    nmi2, ari2, ami2, dim2 = run_kmeans_test_centroid(nodeEmb, gt, clusters, centroids, test_ids)
    
    print("Without Centroids")
    print(f"NMI = {nmi1}, ARI = {ari1}, AMI = {ami1}, Dim = {dim1}\n")
    print("With Centroids Clustering:")
    print(f"NMI = {nmi2}, ARI = {ari2}, AMI = {ami2}, DIM = {dim2}\n")
    
    return nmi1, ari1, ami1, dim1, nmi2, ari2, ami2, dim2


def store_result(X, A, graphName, newTrain, iteration, clusters):
	test_ids = np.sort(loadtxt('data/'+graphName.upper()+'/test_ids_1.txt'))
	val_ids = np.sort(loadtxt('data/'+graphName.upper()+'/val_ids_1.txt'))
	train_ids = np.sort(loadtxt('data/'+graphName.upper()+'/train_ids_1.txt'))
	new_train_ids = np.sort(newTrain)

	X_test = np.zeros((len(test_ids), X.shape[1]))
	X_val = np.zeros((len(val_ids), X.shape[1]))
	X_train = np.zeros((len(train_ids), X.shape[1]))
	X_new_train = np.zeros((len(new_train_ids), X.shape[1]))
	
	for t in range(len(test_ids)):
		X_test[t] = X[int(test_ids[t])]
		
	for v in range(len(val_ids)):
		X_val[v] = X[int(val_ids[v])]

	for tr in range(len(train_ids)):
		X_train[tr] = X[int(train_ids[tr])]

	for tr in range(len(new_train_ids)):
		X_new_train[tr] = X[int(new_train_ids[tr])]
	labelEmb = A[-(clusters*2):,:]
	centroids = np.zeros((clusters,X.shape[1]))
	c = 0
	for i in range(1,clusters*2,2):
		for j in range(0,X.shape[1]):
			centroids[c][j] = labelEmb[i][j]
		c+=1	
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_attribute_label_emb_'+str(iteration)+'.txt', A[:,:-clusters*2])
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_centroids_emb_'+str(iteration)+'.txt', centroids)
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_test_emb_'+str(iteration)+'.txt', X_test)
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_train_emb_'+str(iteration)+'.txt', X_train)
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_val_emb_'+str(iteration)+'.txt', X_val)
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_new_train_emb_'+str(iteration)+'.txt', X_new_train)
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_emb_'+str(iteration)+'.txt', X)
