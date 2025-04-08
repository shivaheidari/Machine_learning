import numpy as np

class NaiveBayse:
    
    def __init__(self):
        pass
    


    def fit(self, x, y):

        #calulate probs for predicting p(spam)
        val , count = np.unique(y, return_counts=True)
        prob_spam = count / len(y)
        n_samples, n_features = x.shape
        x_c = x [y == 1]
        
        # for feature in range(n_features):
        #     sub_one = x [y [y == 1]]



        return x_c



# Each row is a document (email), columns are word counts
x = np.array([
    [3, 0, 1],  # "buy buy buy offer"
    [2, 1, 0],  # "buy click"
    [0, 0, 4],  # "hello hello hello hello"
    [0, 1, 3]   # "click hello hello hello"
])
y = np.array([1, 1, 0, 0])  # 1: spam, 0: not spam

input = np.array([4, 0, 6])
model = NaiveBayse()
print(model.fit(x,y))