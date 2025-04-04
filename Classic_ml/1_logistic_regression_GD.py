import numpy as np

class LogisticRegresson:

    def __init__(self, learning_rate=0.01, epochs=5):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.tr = 0.5
   
    def _sig(self,x):

        return 1/(1 + np.exp(-x))
        
    def fit(self,x,y):
        n_samples, n_features= x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
        #predict
            linear_pred = x @ self.weights + self.bias
            y_pred = self._sig(linear_pred)

            dw = (1/n_samples) * x.T @ (y_pred-y)
            db = (1/n_samples) * np.sum(y-y_pred)
        #update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def _predict_prob(self, x):
        return self._sig(x @ self.weights + self.bias)

    def predict(self, x):
        
        pred_prob = self._predict_prob(x)
        return np.where(pred_prob >= self.tr,0,1)


x = np.array([[-1],[-2],[9],[-3],[8],[4],[5]])
y = np.array([0,0,1,0,1,1,1])

model = LogisticRegresson()
model.fit(x,y)
print(model.predict([-8]))