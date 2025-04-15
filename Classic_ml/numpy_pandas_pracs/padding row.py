import numpy as np
import pandas as pd

"""
Given a ragged 2D NumPy array (rows have varying lengths), write a function to pad each row with zeros so all 
rows have equal length (max row length in the array). 
Then convert it to a Pandas DataFrame and fill missing values with the column median.

"""

def padding(arr):

    #get max lenght of arr 

    max_len = lambda a: max(map(len, a))
    ""
    max_data = max_len(arr)

    #for each item check the lenght and extend if needed 
    #fill with zero
    #convert to df adn fill with zeros with median
    return max_data

arr = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6])]
print(padding(arr))