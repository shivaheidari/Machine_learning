import numpy as np 


class kmeans:
    def __init__(self,num_clusters=None, max_itr = 20):
        self.num_clus = num_clusters
        self.centroids = []
        self.labels = []
        self.max_itr = max_itr
    
    def _select_centroids(self, x):
        indeces = np.random.choice(x.shape[0], self.num_clus)
        centroids = x[indeces]
        self.centroids = centroids
        return self.centroids

    def _distance_cal(self, x):
        distances = np.zeros((x.shape[0], self.num_clus))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(x - centroid, axis=1)
        return distances

    def fit(self, x):
        # select centroids randomly
        centroids = self._select_centroids(x)
       
        #calculate distance for each x 
        for _ in range(self.max_itr):
            distances = self._distance_cal(x)
            self.labels = np.argmin(distances, axis=1)
             #update centroids: mean of the xs in the same center
            for i in range(self.num_clus):
                new_centroids = np.mean(x[self.labels == i], axis=0)

            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

    def predict(self, x):
        distances = self._distance_cal(x)
        return np.argmin(distances, axis=1)















x = np.vstack([np.random.normal((0,0), 0.3, (50, 2)), np.random.normal((5,5), 0.4, (30,2)), np.random.normal((10,10), 0.6, (40, 2))])
print(x)
model = kmeans(3, max_itr=10)
model.fit(x)
model.predict(x)