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

vec1=[1,2]
vec2=[1,3]
print(euclidean_distance(vec1,vec2))
print(manhatan_distance(vec1,vec2))