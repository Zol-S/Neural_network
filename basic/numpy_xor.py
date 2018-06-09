import numpy as np

# Matrix multiplication test
#A = np.arange(6).reshape((2, 3))
#B = np.array([[1, 1], [2, 2], [3, 3]])

#element_multiplication = np.multiply(A, B.transpose())
#matrix_multiplication = np.matmul(A, B)

def relu(i):
    if i < 0:
        return 0
    else:
        return i

Weight = np.array([[1, 1], [1, 1]])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).transpose()
c = np.array([0, -1])

XW = np.matmul(Weight, X).transpose() + c

func_relu = np.vectorize(relu)
XW = func_relu(XW)

w = np.array([[1], [-2]])

output = np.matmul(XW, w)

