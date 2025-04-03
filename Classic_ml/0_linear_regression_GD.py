import pandas as pd
import numpy as np


x = np.array([[1],[2],[3]])
y = np.array([3,7,10])


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=10, lambda_=0.1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.lambda_ = lambda_
    
    
    def _sign(self, w):
        """Subgradient of L1 penalty."""
        return np.where(w > 0, 1, np.where(w < 0, -1, 0))
    
    def fit(self,x,y):
        """
        x:input, shape (m,n)
        y:output, shape(m,)
        """
        m,n = x.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.n_iters):
            #compute preds
            y_pred = np.dot(x,self.weights)+self.bias
            #compute gradient weights and l2 penalty
            dw = (2/m) * np.dot(x.T,(y_pred-y))+ 2 * self.lambda_* self.weights
            #dw = (2/m) * np.dot(x.T,(y_pred-y))+ 2 * self.lambda_* self._sign(self.weights)
            #compute gradient bias
            db = (2/m) * np.sum(y_pred-y)
            #update weights
            self.weights -= self.lr * dw
            #update bias
            self.bias -= self.lr * db
        return self.weights, self.bias
        
    def predict(self,x):
        return np.dot(x,self.weights) + self.bias
    

model = LinearRegression(learning_rate=0.01, n_iters=100)
model.fit(x, y)

# Predict
print(model.predict([[5]])) 
