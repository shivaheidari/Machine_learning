import numpy as np
from sklearn.tree import DecisionTreeClassifier
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

def bagging(x, y, n_estimatiors, sample_size=None ):
    #sampling with replacement
    models = []
    n_samples = x.shape[0]
    for _ in range(n_estimatiors):
        indeces = np.random.choice(n_samples, sample_size, replace=True)
        x_subsample = x[indeces]
        y_subsample = y[indeces]

        model = DecisionTreeClassifier()
        model.fit(x_subsample, y_subsample)
        models.append(model)
        
    

    return models

x = np.random.randint(0,40, 20).reshape(-1,1)
y = np.random.randint(0,1,20)
models = bagging(x,y, 3, 10)
test_data = np.random.randint(0,40,3).reshape(-1,1)
test_data = [[10],[20],[30],[30]]
# for model in models:
#     estimates = []
#     for data in test_data:
#         data = np.array(data).reshape(-1,1)
#         estimate = model.predict(data)
#         estimates.append(estimate)
    
def hard_voting(x):
    

    vals, counts = np.unique(test_data, return_counts=True)
    majority = vals[np.argmax(counts)]
    return majority


input = [1,2,1,1,3]

counts = {}
for item in input:
    if item not in counts:
        counts[item] = 1
    else:
        counts[item] += 1
print(counts)
x = [20,2,3]
y = [2,1,4]
zipped = zip(x, y)
for x, y in zipped:
    print(x,y)