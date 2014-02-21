__author__ = 'maucher'

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

def clustInd2ClustMember(clusterIdx):
    ClusterMembers={}
    for idx,clust in enumerate(clusterIdx):
        ClusterMembers.setdefault(clust,[])
        ClusterMembers[clust].extend([idx])
    return ClusterMembers


X=np.load("NormalizedHRVdata.npy")
#print X

for q in [0.08,0.09,0.1,0.11,0.12]:
    print "-"*10,"quantile=%f"%(q),"-"*10
    bandwidth = estimate_bandwidth(X,quantile=q)
    ms=MeanShift(bandwidth,bin_seeding=True)
    ms.fit(X)
    print ms.labels_
    print clustInd2ClustMember(ms.labels_)
    print metrics.silhouette_score(X,ms.labels_,metric="euclidean")


