import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=30, batch_size=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.batch_size = batch_size

    def fit(self,x,y):
        #random weights and bias
        y = y.reshape(-1)
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        #select a batch
        for epoch in range(self.epochs):
            indeces = np.random.permutation(n_samples)
            x_shuffled = x[indeces]
            y_shuffled = y[indeces]
            
            #mini bach
            for i in range(0, n_samples, self.batch_size):
                x_batch = x_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                y_pred = x_batch @ self.weights + self.bias
                error = y_pred-y_batch
                dw = (1/len(x_batch)) * (x_batch.T @ error)
                db = (1/len(x_batch)) * np.sum(error)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                # epoch_loss = self._mse(y, self.predict(x))
                # self.loss_history.append(epoch_loss)
                
        
    
    def predict(self,x):
        return np.dot(x, self.weights)+self.bias
   
    def _mse(y_true, y_pred):
        return np.mean((y-y_pred)**2)

np.random.seed(42)

x = np.random.randn(100,2)
w_true = np.array([[3],[2]])
b_true = 4
y = x @ w_true + b_true + np.random.rand(100,1)*0.5
model = LinearRegression(learning_rate=0.01, epochs=10, batch_size=20)
model.fit(x,y)
print(model.predict(x[0]), y[0])

