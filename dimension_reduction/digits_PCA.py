from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import numpy as np

digits = load_digits()
X,y = digits.data, digits.target

#Running PCA retaining 95% of the variance
#So with 64 original features, we need 29 principal components to explain 95% of the original dataset
'''pca_digits=PCA(0.95)
X_proj = pca_digits.fit_transform(X)
print(X.shape, X_proj.shape)'''

#Projecting dataset into 2D
'''pca_digits=PCA(2)
X_proj = pca_digits.fit_transform(X)
print(np.sum(pca_digits.explained_variance_ratio_))

plt.scatter(X_proj[:,0], X_proj[:,1], c=y)
plt.colorbar()
plt.show()'''

#How much data are we throwing away?
#Lets try and plot number of components versus explained variance ratio as a cumulative sum to find out
pca_digits = PCA(64).fit(X)
plt.semilogx(np.cumsum(pca_digits.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance retained')
plt.ylim(0,1)
plt.show()
