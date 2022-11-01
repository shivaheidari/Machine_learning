#Distance metric 
import imp
from math import sqrt
def euclidean_distance(vector1,vector2):
    distance=0
    for i in range(0,len(vector1)):
       distance += (vector1[i]-vector2[i])**2
    distance=sqrt(distance)
    return distance

def manhatan_distance(vector1 , vector2):
    distance=0
    for i in range(0, len(vector1)):
        distance += abs(vector1[i]-vector2[i])
    distance=distance/len(vector1)
    return distance


#getting nearest neighbor
def get_neighbors(train, test_row, num_neighbors):

    distances=list()
    neighbors=list()
    for train_row in train:
        dist=euclidean_distance(train_row,test_row)
        distances.append((train_row,dist))
    distances.sort(key=lambda tup: tup[1])
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors

vec1=[1,2]
vec2=[1,3]
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

neighbors = get_neighbors(dataset, dataset[0], 3)
for neighbor in neighbors:
	print(neighbor)
# row0=dataset[0]
# for row in dataset:
#     print(euclidean_distance(row0,row))
# print(euclidean_distance(vec1,vec2))
# print(manhatan_distance(vec1,vec2))