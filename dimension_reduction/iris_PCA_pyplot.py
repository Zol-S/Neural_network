#https://plot.ly/ipython-notebooks/principal-component-analysis/

import pandas as pd
import numpy as np

import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True)

#print(df.tail())

X = df.ix[:,0:4].values
y = df.ix[:,4].values

# mean removal and variance scaling
X_std = StandardScaler().fit_transform(X) # [-9.00681170e-01,  1.03205722e+00, -1.34127240e+00, -1.31297673e+00]
X_std2 = (X - X.mean(axis=0)) / X.std(axis=0)
X_diff = X_std - X_std2



'''mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#cov_mat = np.cov(X_std.T)
#print('Covariance matrix \n%s' %cov_mat)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

u,s,v = np.linalg.svd(X_std.T)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)

# Projecting onto the new feature space
Y = X_std.dot(matrix_w)'''
