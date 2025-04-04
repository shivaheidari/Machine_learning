import numpy as np
from collections import Counter
import random
"""
input: tuple (x1,x2,y) y:(0,1)
knn:knearest neighbor
neighbor detector with k : Ecludian distance
assign the maximum of the class if equal random

"""
class KNN:
    def __init__(self,k=3,x=None,y=None, critrion="euclidean"):
        self.k = k
        self.data = x
        self.label = y
        self.critrion = critrion

    def _distance(self,a,b):
        return  np.where(self.critrion=="euclidean", np.sqrt(np.sum((a - b)**2)),np.sum(np.abs(a-b)))
        # print(res)
        # if self.critrion == "euclidean":
          
        #   return np.sqrt(np.sum((a - b)**2))
        
        # elif self.critrion == "manhatan":
        #    return np.sum(np.abs(a-b))
            
    def predict(self, x):
        k_neighbors= []
        for index, point in enumerate(self.data):
            distance = self._distance(x, point)
            if len(k_neighbors) < self.k:
                k_neighbors.append((index,distance))
            else:
                
                furthest = max(k_neighbors, key=lambda id: id[1])
                if distance < furthest[1]:
                    k_neighbors.remove(furthest)
                    k_neighbors.append((index, distance))
        #find the winner
        knn_class = []
        for item in k_neighbors:
            label = self.label[item[0]]
            knn_class.append(label)
        
        pred = self.most_frequent_random_tiebreak(knn_class)
        return pred
    
    def most_frequent_random_tiebreak(self,knn_class):
        counter = Counter(knn_class)
        most_common = counter.most_common()
        max_count = most_common[0][1] 
        candidates = [item for item, count in most_common if count == max_count]
        return random.choice(candidates)
        
            





data = np.array([[0,1],[1,1],[-8,0],[-9,1]])
label = np.array([0,0,1,1])
model = KNN(3,data, label, critrion="euclidean")
print(model.predict([-7,0]))




