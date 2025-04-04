import numpy as np

"""
finding supprot vectors
need kernel 
finding margines in soft or hard in maximizing the support vectors distance
predict: distance to the hyperplane
need regulaization?
j(theta)?

"""
class SVM:
    def __init__(self, epochs, C):
        self.epochs = epochs
        self.C = C
        self.weights = None
        self.bias = None

    def _hinge_loss(self, x, y):
        #loss
        loss = (1/2)*np.dot(self.w, self.w)+ self.C * (sum(max([0, 1-(y-(x @ self.weights + self.b))])))
        #dw
        #db
        
    def fit(self,x,y):


        pass
    def predict(self,x):
        pass


import numpy as np


