import numpy as np

x = np.array([[1],[2],[3]])
y = np.array([3,7,10])




class LinearRegression:
    def __init__(self):

        self.weights = None
        self.bias = None
    def fit(self, x, y):
        #(x.tx)^-1*xty
        #add 1s for x with bias
        x_b = np.c_[np.ones(x.shape[0]), x]
        self.weights = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
    
    def predict(self, x):
        return np.dot(x,self.weights) + self.bias
    
model = LinearRegression()
model.fit(x, y)
pred = model.predict([4])
print(pred)