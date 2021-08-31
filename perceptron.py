import numpy as np

class preceptron():

    def AND(self,x1,x2):
        x = np.array([x1,x2])
        w = np.array([0.5,0.5])
        b = -0.7
        tmp = np.sum(w*x) + b
        if tmp <= 0 :
            return 0
        else:
            return 1

    def NAND(self,x1,x2):
        x = np.array([x1,x2])
        w = np.array([-0.5,0.5])
        b = 0.7
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def OR(self,x1,x2):
        x = np.array([x1,x2])
        w = np.array([0.5, 0.5])
        b = -0.2
        tmp = np.sum(w*x) + b
        if tmp <= 0 :
            return 0
        else:
            return 1

P = preceptron()

def XOR(x1 , x2):
    s1 = P.NAND(x1,x2)
    s2 = P.OR(x1,x2)
    y  = P.AND(x1,x2)
    return y





