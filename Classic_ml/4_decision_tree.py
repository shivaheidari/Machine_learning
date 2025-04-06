import numpy as np



class DecisionTree:
    def __init__(self, max_depth=None, min_sample_split=2):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.tree = None
        

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - sum (probs ** 2)
    
    def _entropy(self, y):
        classes , counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        entropy = np.sum(-(probs)*np.log2(probs))
        return entropy
    def info_gain(self, y, y_left, y_right):
        p_left = float(len(y_left)) / len(y)
        gain = self._entropy(y)- p_left * self._entropy(y_left) + (1-p) * self._entropy(y_right)
        return gain
    
    def show(self):
        return self.tree
    
    def _best_split(self, x, y):
        best_gini = 1
        best_feature, best_threshold = None, None
        n_features = x.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                left_branch = x[:,feature] <= threshold
                right_branch = x[:, feature] > threshold

                if len(y[left_branch]) == 0 or len(y[right_branch]) == 0:
                    continue

                gini_left = self._gini(y[left_branch])
                gini_right = self._gini(y[right_branch])
                weighted_gini = (len(y[left_branch]) * gini_left + len(y[right_branch])* gini_right)/len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def _build_tree(self, x, y, depth=0):
        n_classes = len(np.unique(y))
        n_samples, n_features = x.shape
        if n_classes == 1 or (n_samples < self.min_sample_split) or (self.max_depth is not None and depth >= self.max_depth):
            return {"class": np.bincount(y).argmax()}
        
        feature, threshold = self._best_split(x,y)
        if feature is None:
            return {'class': np.bincount(y).argmax()}
        left_branch = x[:, feature] <= threshold
        right_branch = ~left_branch

        left_subtree = self._build_tree(x[left_branch], y[left_branch], depth+1)
        right_subtree = self._build_tree(x[right_branch], y[right_branch], depth+1)
        
        return{
            "feature": feature, 
            "threshold": threshold, 
            "left": left_subtree, 
            "right": right_subtree
        }
    
    def fit(self, x, y):
        self.tree = self._build_tree(x, y)
    
    def _predict_sample(self, x, node):
        if "class" in node:
            return node["class"]
        
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(x, node["left"])
        else:
            return self._predict_sample(x, node["right"])
    
    def predict(self, x):
        return np.array([self._predict_sample(sample, self.tree) for sample in x])
    
X = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train
tree = DecisionTree(max_depth=2)
print(tree._gini(y))
# tree.fit(X, y)

# # Predict
# print(tree.predict(np.array([[4, 4], [1, 1]]))) 

# print(tree.show())