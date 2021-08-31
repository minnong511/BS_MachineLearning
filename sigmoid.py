import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0 ,dtype= np.int_)

x = np.arange( -5.0 , 5.0 , 0.1 )
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))


X = np.array([1.0 , 0.5])
W1 = np.array([[0.1, 0.3 , 0.5] , [0.2 ,0.4, 0.6]])
B1 = np.array([0.1, 0.2 ,0.3])

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)
print(Z1)