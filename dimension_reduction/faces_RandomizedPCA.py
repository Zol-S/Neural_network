#https://shankarmsy.github.io/posts/pca-sklearn.html
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import RandomizedPCA

import matplotlib.pyplot as plt

oliv=fetch_olivetti_faces()
X, y=oliv.data, oliv.target

rpca_oliv = RandomizedPCA(64).fit(X)
print("Randomized PCA with 64 components: ", np.sum(rpca_oliv.explained_variance_ratio_))
print("PCA with 64 components: ", np.sum(rpca_oliv.explained_variance_ratio_))

# Setup a figure 8 inches by 8 inches
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the faces, each image is 64 by 64 pixels
for i in range(10):
    ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
    ax.imshow(np.reshape(rpca_oliv.components_[i,:], (64, 64)), cmap=plt.cm.bone, interpolation='nearest')

plt.show()
