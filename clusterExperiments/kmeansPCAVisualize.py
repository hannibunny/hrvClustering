__author__ = 'maucher'

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from matplotlib import pyplot as pl

X=np.load("NormalizedHRVdata.npy")
print X

k=3
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
print km.labels_
print kmRed.labels_
print kmIso.labels_
print "Silhouette Score"
print metrics.silhouette_score(X,km.labels_,metric="euclidean")
print metrics.silhouette_score(X,kmRed.labels_,metric="euclidean")
print metrics.silhouette_score(X,kmIso.labels_,metric="euclidean")

pl.subplot(1,2,1)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min()-h , reduced_data[:, 0].max()+h
y_min, y_max = reduced_data[:, 1].min()-h , reduced_data[:, 1].max()+h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmRed.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
#pl.figure(1)
#pl.clf()
pl.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=pl.cm.Paired,
          aspect='auto', origin='lower')

pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)
tsh=0.02
for i in range(reduced_data.shape[0]):
    pl.text(reduced_data[i, 0]+tsh, reduced_data[i, 1]+tsh,str(i))
# Plot the centroids as a white X
centroids = kmRed.cluster_centers_
pl.scatter(centroids[:, 0], centroids[:, 1],
           marker='x', s=169, linewidths=3,
           color='w', zorder=10)
pl.title('K-means clustering on HRV data (PCA-reduced data)\n'
         'Centroids are marked with white cross')
pl.xlim(x_min, x_max)
pl.ylim(y_min, y_max)
pl.xticks(())
pl.yticks(())

pl.subplot(1,2,2)
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = isomap_data[:, 0].min() -h, isomap_data[:, 0].max() +h
y_min, y_max = isomap_data[:, 1].min() -h, isomap_data[:, 1].max() +h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmIso.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
#pl.figure(1)
#pl.clf()
pl.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=pl.cm.Paired,
          aspect='auto', origin='lower')

pl.plot(isomap_data[:, 0], isomap_data[:, 1], 'k.', markersize=5)

for i in range(isomap_data.shape[0]):
    pl.text(isomap_data[i, 0]+tsh, isomap_data[i, 1]+tsh,str(i))
# Plot the centroids as a white X
centroids = kmIso.cluster_centers_
pl.scatter(centroids[:, 0], centroids[:, 1],
           marker='x', s=169, linewidths=3,
           color='w', zorder=10)
pl.title('K-means clustering on HRV data (Isomap-reduced data)\n'
         'Centroids are marked with white cross')
pl.xlim(x_min, x_max)
pl.ylim(y_min, y_max)
pl.xticks(())
pl.yticks(())

pl.show()