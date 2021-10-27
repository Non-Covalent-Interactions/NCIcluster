#! /usr/bin/env python3

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

from spatial.utils import *


def get_isosurface(X, isovalue):
    """ Select gridpoints of X that have a rdg equal or lower than isovalue and saves into array. 
    
    Parameters
    ----------
    X : np.array
       Array with columns corresponding to space coordinates, sign(l2)*dens and rdg.
    isovalue : float
       Cutoff for rdg.
    """
    idx = np.where(X[:, -1] <= isovalue)
    X_iso = X[idx]
    return X_iso


def divide_nci_regions(
    X,
    n_clusters,
    method='dbscan',
    isovalue=0.3,
    only_pos=True,
    size_sample=1.0,
    min_dist=0.5,
    max_dist=3.0,
    min_cluster_portion=0.1,
    discard_tails_thr=None,
):
    """ Get elements of X smaller or equal to isovalue and assign them to clusters. 
    
    Parameters
    ----------
    X : np.array
       Array with columns corresponding to space coordinates, sign(l2)*dens and rdg.
    n_clusters : int or sequence of ints
       Number of clusters that the estimator considers. If sequence, a grid search for the optimum value is performed.
    isovalue : float, optional
       Cutoff for rdg.
    only_pos : boolean, optional
       If True, only spatial coordinates are considered in the clustering algorithm. If False, density and gradient are also taken into account.
    size_sample : float, optional
       Size of subset of data that will be considered to do the clustering. If 0.0 < size_sample <= 1.0, then it is taken as a fraction of total data.
    min_dist: float, optional
       If minimum distance between clusters is smaller, gives a warning.
    max_dist: float, optional
       If maximum distance between elements in a cluster is larger, gives a warning.
    min_cluster_portion: float, optional
       If a cluster is smaller than portion times the largest, gives a warning.
    """
    print("  Isovalue: {}".format(isovalue))

    X_iso = get_isosurface(X, isovalue)
    if discard_tails_thr is not None:
        X_iso = X_iso[np.absolute(X_iso[:,3])>discard_tails_thr]
        X_iso = X_iso[np.absolute(X_iso[:,4])>discard_tails_thr]
        print("  Discarded tails with density and s < {}".format(discard_tails_thr))    

    if size_sample <= 0:
        raise ValueError(
            "size_sample must be a positive number, your input : {}".format(size_sample)
        )
    if 0 < size_sample <= 1.0:
        size_sample = int(size_sample * len(X_iso))
    print("  Fraction of points taken: {}".format(size_sample / len(X_iso)))
    print("  Number of points taken:   {}".format(size_sample))    
    

    rand_idx = np.arange(len(X_iso))
    np.random.shuffle(rand_idx)
    X_iso = X_iso[rand_idx[:size_sample]]

    if not hasattr(n_clusters, "__iter__"):
        n_clusters = [n_clusters]
    elif method=="kmeans":
        print("  N clusters range: {}".format(list(n_clusters)))


    if only_pos:
        X_fit = np.zeros((len(X_iso), 3))
        X_fit[:, :] = X_iso[:, :3]
    else:
        X_fit = np.zeros((len(X_iso), 4))
        X_fit[:, :] = X_iso[:, :4]

    t0 = time.time()
    if method.lower()=="kmeans":
        n_space = {"n_clusters": list(n_clusters)}
        cv = [(slice(None), slice(None))]
        estimator = KMeans()
        print("  Clustering with KMeans method")
        gs = GridSearchCV(
            estimator=estimator, param_grid=n_space, scoring=cv_silhouette_scorer, cv=cv, n_jobs=-1
        )
        gs.fit(X_fit)
        n_clusters = gs.best_params_["n_clusters"]
        logging.info("Number of clusters: {}".format(n_clusters))
        model = KMeans(n_clusters=gs.best_params_["n_clusters"], random_state=0)

    elif method.lower()=="dbscan":
        print("  Clustering with DBSCAN method")
        model = DBSCAN(eps=1.0, min_samples=min(10, int(0.01*size_sample)))

    model.fit(X_fit)
    elapsed_time = time.time() - t0
    logging.info("Clustering time: {}".format(elapsed_time))
 
    #print(model.labels_)
    n_clusters = len(np.unique(model.labels_[np.where(model.labels_!=-1)[0]]))
    print("  Best N clusters: {}".format(n_clusters))
    clusters = []
    for label in range(n_clusters):
        clusters.append(X_fit[np.where(model.labels_ == label)[0]])  
    
    try:
        min_distance_clusters(clusters, warning_val=min_dist)
    except:
        print("  We are not able to provide an evaluation of the clustering through min distances inside cluster.")
    try:
        max_distance_cluster(clusters, warning_val=max_dist)
    except:
        print("  We are not able to provide an evaluation of the clustering through max distances between clusters.")
    try:
        warning_small_cluster(clusters, portion=min_cluster_portion, size=None)
    except:
        print("  We are not able to provide an evaluation of the clustering through cluster sizes.")

    return X_iso[np.where(model.labels_!=-1)[0]], model.labels_[np.where(model.labels_!=-1)[0]]
