import numpy as np
"""
input: tuple (x1,x2,y) y:(0,1)
knn:knearest neighbor
neighbor detector with k : Ecludian distance
assign the maximum of the class if equal random

"""
class KNN:
    def __init__(self,k=3,x=any,y=any):
        self.k = k
        self.data = x
        self.label = y

    def _distance(self,a,b):
        f_d = np.power((a[0]-b[0]),2)
        s_d = np.power((a[1]-b[1]),2)
        distance = np.sqrt(f_d+s_d)
        return distance
           
    def predict(self, x):
        k_neighbors= []
        for index, point in enumerate(self.data):
            distance = self._distance(x, point)
            
            if len(k_neighbors) < self.k:
                k_neighbors.append((index,distance))
            else:
                furthest = max(k_neighbors, key=lambda id: id[1])
                if distance < furthest[1]:
                    distance.append((index, distance))
        #find the winner
        knn_class = []
        for item in k_neighbors:
            knn_class.append(item[0])
        
        pred = max(set(knn_class), key=knn_class.count)
        return pred
        
            





data = np.array([[0,1],[1,1],[-8,0],[-9,1]])
label = np.array([0,0,1,1])

model = KNN(3,data, label)
print(model.predict([0,2]))




