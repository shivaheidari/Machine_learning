import numpy as np

"""
finding supprot vectors
need kernel 
finding margines in soft or hard in maximizing the support vectors distance
predict: distance to the hyperplane
need regulaization?
j(theta)

"""
class SVM:
    def __init__(self, epochs, C, learning_rate):
        self.epochs = epochs
        self.C = C
        self.lr = learning_rate
        self.weights = None
        self.bias = None


    def _hinge_loss(self, x, y):
        #loss
        distances = 1 - y * (x @ self.weights + self.bias)
        distances[distances < 0] = 0
        loss = (1/2)*np.dot(self.weights, self.weights) + self.C * (np.sum(distances))
        dw = self.weights - self.C * np.sum((y * x.T).T * (distances > 0).reshape(-1,1), axis=0)
        db = -self.C * np.sum(y*(distances > 0))
        return loss, dw, db
       
        
    def fit(self,x,y):

        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.epochs):
            loss, dw, db = self._hinge_loss(x,y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            if _ % 100 == 0:
                print(f"Epoch {_}, Loss: {loss:.4f}")
        
    def predict(self,x):
        linear_out = x @ self.weights + self.bias
        return np.sign(linear_out)
        

X = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y = np.array([-1, -1, -1, 1, 1, 1])  

# Train SVM
svm = SVM(epochs=10, C=0.01, learning_rate=0.01)
svm.fit(X, y)

# Predict
test_samples = np.array([[4, 4], [1, 1]])
print("Predictions:", svm.predict(test_samples))


