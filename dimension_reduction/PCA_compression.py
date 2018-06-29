from sklearn.datasets import load_iris
iris = load_iris()
#checking to see what datasets are available in iris
print(iris.keys())

#checking shape of data and list of features (X matrix)
print(iris.data.shape)
print(iris.feature_names)

#checking target values
print(iris.target_names)

#importing and instantiating PCA with 2 components.
from sklearn.decomposition import PCA

pca = PCA(2)
print(pca)

X, y = iris.data, iris.target
X_proj = pca.fit_transform(X)

print(X_proj.shape)

import numpy as np
import matplotlib.pyplot as plt

#Plotting the projected principal components and try to understand the data.
#Ignoring what's in y, it looks more like 2 clusters of data points rather than 3
#c=y colors the scatter plot based on y (target)

plt.scatter(X_proj[:,0], X_proj[:,1],c=y)
plt.show()
