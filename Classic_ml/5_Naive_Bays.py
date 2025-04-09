import numpy as np
from collections import defaultdict

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.mean = defaultdict(list)
        self.var = defaultdict(list)

    def fit(self, X, y):

        samples, features = X.shape
        classes, counts = np.unique(y, return_counts=True)
        self.classes = classes
        
        for c in classes:
             x_c = X [y == c]
             self.mean[c] = np.mean(x_c, axis=0)
             self.var[c] = np.var(x_c, axis=0)
             self.priors[c] = x_c.shape[0] / samples
             
       
    def _calculate_likelihood(self, x, mean, var):
        """Gaussian PDF to calculate likelihood"""
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

        
    def predict(self, X):
         y_pred = []
         for x in X:
            posteriors = []
            for c in self.classes:
                # Calculate log-likelihood to avoid underflow
                likelihood = np.sum(np.log(self._calculate_likelihood(x, self.mean[c], self.var[c])))
                prior = np.log(self.priors[c])
                posterior = prior + likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
         return np.array(y_pred)
                 
            
            
             
             
             
        

# Example Usage
if __name__ == "__main__":
    # Sample data (features: sepal length, sepal width; labels: 0 or 1)
    X_train = np.array([[5.1, 3.5], [4.9, 3.0], [6.0, 2.7], [5.8, 2.8]])
    y_train = np.array([0, 0, 1, 1])

    # Train
    gnb = GaussianNaiveBayes()
    print(gnb.fit(X_train, y_train))

    # # Test
    X_test = np.array([[5.5, 3.2], [6.2, 2.9]])
    print(gnb.predict(X_test))  # Output: [0 1]