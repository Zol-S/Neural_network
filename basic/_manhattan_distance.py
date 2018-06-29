import numpy as np
from scipy.misc import imread, imresize

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances)
            ##Ypred[i] = self.ytr[min_index]

        return distances[min_index]

cat1 = imread('assets/cat1.jpg')
cat2 = imread('assets/cat2.jpg')

x = NearestNeighbor()
x.train(cat1, 1)
print(x.predict(cat2))


