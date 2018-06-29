import numpy as np

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype = float)
y = np.array(([75], [82], [93]), dtype = float)

X = X/np.amax(X, axis=0)
y = y/100

#print(X.shape, y.shape)
class Neural_Network(object):
	def __init__(self):
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3

		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

	def forward(self, X):
		#Propagate inputs through network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self, z):
		#Derivative of Sigmoid function
		return np.exp(-z)/((1+np.exp(-z))**2)

	def costFunction(self, X, y):
		#Compute cost for given X,y, use weights already stored in class.
		self.yHat = self.forward(X)
		J = 0.5*sum((y-self.yHat)**2)
		return J

	def costFunctionPrime(self, X, y):
		#Compute derivative with respect to W1 and W2
		self.yHat = self.forward(X)

		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)

		return dJdW1, dJdW2

	def getParams(self):
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params

	def setParams(self, params):
		W1_start = 0
		W1_end = self.hiddenLayerSize*self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

	def computeGradients(self, X, y):
		dJW1, dJW2 = self.costFunctionPrime(X, y):
			return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))