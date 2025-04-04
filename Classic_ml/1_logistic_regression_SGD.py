import numpy as np
"""
randomly selects subest of data in the fit part
loss: BCE 1/n sum(ylogy')+(1-y)log(1-y'), y' = sig(xw+b)
sidw:1/n X.T(y'-y)

"""
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=5, batch_size=2, threshold=0.5):
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.weights = None
        self.bias = None 

        self.tr = threshold

    def _sig(self, x):
        return (1/1+np.exp(-x))
    def fit(self,x,y):
        #init
        n_sampels, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        #for epoch select subset 
        for _ in range(self.epochs):
            indeces = np.random.permutation(n_sampels)
            x_shuffled = x[indeces]
            y_shuffled = y[indeces]
            for i in range(0, n_sampels, self.batch_size,):
                x_batch = x_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                linear_pred = x_batch @ self.weights + self.bias
                y_pred = self._sig(linear_pred)

                dw = (1/len(x_batch)) * x_batch.T @ (y_pred - y_batch)
                db = (1/len(x_batch)) * np.sum(y_pred - y_batch)

                #update

                self.weights -= self.lr * dw
                self.bias -= self.lr * db


    
    def predict_prob(self, x):

        linear_pred = x @ self.weights + self.bias
        return self._sig(linear_pred)
    
    def predict(self, x):
        return (self.predict_prob(x) >= self.tr).astype(int)

        
x = np.array([[-1],[-2],[9],[-3],[8],[4],[5],[-3]])
y = np.array([0,0,1,0,1,1,1,0])

model = LogisticRegression()
model.fit(x,y)
print(model.predict([-8]))
