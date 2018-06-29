#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
#https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_MNIST_Logistic_Regression_EasierDataLoading.ipynb

import pandas as pd
import numpy as np 
# Suppress scientific notation
np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Used for Downloading MNIST
from sklearn.datasets import fetch_mldata

# Used for Splitting Training and Test Sets
from sklearn.model_selection import train_test_split

# Change data_home to wherever to where you want to download your data
mnist = fetch_mldata('MNIST original')

train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
'''plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
	plt.subplot(1, 5, index + 1)
	plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
	plt.title('Training: %i\n' % label, fontsize = 20)'''

# Reduction
pca = PCA(.9)
pca.fit(train_img)
# print(pca.n_components_)

components = pca.transform(train_img)
approximation = pca.inverse_transform(components)

plt.figure(figsize=(8,4));

# Original Image
'''plt.subplot(1, 2, 1);
plt.imshow(train_img[1].reshape(28,28),
			cmap = plt.cm.gray, interpolation='nearest',
			clim=(0, 255));
plt.xlabel('784 components', fontsize = 14)
plt.title('Original Image', fontsize = 20);

# 154 principal components
plt.subplot(1, 2, 2);
plt.imshow(approximation[1].reshape(28, 28),
			cmap = plt.cm.gray, interpolation='nearest',
			clim=(0, 255));
plt.xlabel(str(pca.n_components_) + ' components', fontsize = 14)
plt.title('95% of Explained Variance', fontsize = 20);
plt.show()'''

# Explained graph vs number of principal components
pca = PCA()
pca.fit(train_img)
tot = sum(pca.explained_variance_)
var_exp = [(i/tot)*100 for i in sorted(pca.explained_variance_, reverse=True)] 
cum_var_exp = np.cumsum(var_exp)

# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 
'''plt.figure(figsize=(10, 5))
plt.step(range(1, 785), cum_var_exp, where='mid',label='cumulative explained variance')
plt.title('Cumulative Explained Variance as a Function of the Number of Components')
plt.ylabel('Cumulative Explained variance')
plt.xlabel('Principal components')
plt.axhline(y = 95, color='k', linestyle='--', label = '95% Explained Variance')
plt.axhline(y = 90, color='c', linestyle='--', label = '90% Explained Variance')
plt.axhline(y = 85, color='r', linestyle='--', label = '85% Explained Variance')
plt.legend(loc='best')
plt.show()'''

# number of principal components
componentsVariance = [784, np.argmax(cum_var_exp > 99) + 1, np.argmax(cum_var_exp > 95) + 1, np.argmax(cum_var_exp > 90) + 1, np.argmax(cum_var_exp >= 85) + 1]
print(componentsVariance)

from sklearn.decomposition import PCA

# This is an extremely inefficient function. Will get to why in a later tutorial
def explainedVariance(percentage, images): 
	# percentage should be a decimal from 0 to 1 
	pca = PCA(percentage)
	pca.fit(images)
	components = pca.transform(images)
	approxOriginal = pca.inverse_transform(components)
	return approxOriginal

plt.figure(figsize=(20,4));

# Original Image (784 components)
plt.subplot(1, 5, 1);
plt.imshow(train_img[4].reshape(28,28),
			cmap = plt.cm.gray, interpolation='nearest',
			clim=(0, 255));
plt.xlabel('784 Components', fontsize = 12)
plt.title('Original Image', fontsize = 14);

# 331 principal components
plt.subplot(1, 5, 2);
plt.imshow(explainedVariance(.99, train_img)[4].reshape(28, 28),
			cmap = plt.cm.gray, interpolation='nearest',
			clim=(0, 255));
plt.xlabel('331 Components', fontsize = 12)
plt.title('99% of Explained Variance', fontsize = 14);

# 154 principal components
plt.subplot(1, 5, 3);
plt.imshow(explainedVariance(.95, train_img)[4].reshape(28, 28),
			cmap = plt.cm.gray, interpolation='nearest',
			clim=(0, 255));
plt.xlabel('154 Components', fontsize = 12)
plt.title('95% of Explained Variance', fontsize = 14);

# 87 principal components
plt.subplot(1, 5, 4);
plt.imshow(explainedVariance(.90, train_img)[4].reshape(28, 28),
			cmap = plt.cm.gray, interpolation='nearest',
			clim=(0, 255));
plt.xlabel('87 Components', fontsize = 12)
plt.title('90% of Explained Variance', fontsize = 14);

# 59 principal components
plt.subplot(1, 5, 5);
plt.imshow(explainedVariance(.85, train_img)[4].reshape(28, 28),
			cmap = plt.cm.gray, interpolation='nearest',
			clim=(0, 255));
plt.xlabel('59 Components', fontsize = 12)
plt.title('85% of Explained Variance', fontsize = 14);
plt.show()