__author__ = 'maucher'

import numpy as np
from sklearn.cluster import KMeans
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
print X

kSet=range(2,8)
for k in kSet:
    print "-"*10,"k=%d"%(k),"-"*10
    km =KMeans(k)
    km.fit(X)
    print "Labels:"
    reduced_data = PCA(n_components=2).fit_transform(X)
    kmRed = KMeans(k)
    kmRed.fit(reduced_data)
    imap=Isomap()
    isomap_data=imap.fit_transform(X)
    kmIso = KMeans(k)
    kmIso.fit(isomap_data)
    #print km.labels_
    print clustInd2ClustMember(km.labels_)
    #print kmRed.labels_
    print clustInd2ClustMember(kmRed.labels_)
    #print kmIso.labels_
    print clustInd2ClustMember(kmIso.labels_)
    print "Silhouette Score"
    print metrics.silhouette_score(X,km.labels_,metric="euclidean")
    print metrics.silhouette_score(X,kmRed.labels_,metric="euclidean")
    print metrics.silhouette_score(X,kmIso.labels_,metric="euclidean")