import numpy as np
"""
 simulate 1–2 rounds of AdaBoost using a stump class like:

class Stump:
    def __init__(self): ...
    def fit(self, X, y, weights): ...
    def predict(self, X): ...

"""

"""
Given several weak classifiers and weights, implement the final prediction logic.

def adaboost_predict(classifiers, alphas, X):
    # classifiers: list of weak learners with .predict
    # alphas: list of float weights
    ...

"""

def adaboost_predict(cassifiers, alphas, X):

    pass

def bagging(x, limit):
    #sampling with replacement
    return np.random.choice(x, size=limit, replace=True)
    


x = np.random.rand(30)
y = bagging(x, limit=10)
print(x,y)
