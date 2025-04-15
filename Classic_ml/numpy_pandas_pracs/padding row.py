import numpy as np
import pandas as pd

"""
Given a ragged 2D NumPy array (rows have varying lengths), write a function to pad each row with zeros so all 
rows have equal length (max row length in the array). 
Then convert it to a Pandas DataFrame and fill missing values with the column median.

"""

def padding(arr):

    

    max_len = max(len(row) for row in arr) 
    
    """for i, item in enumerate(arr):
        if len(item) < max_d:
            subarr = np.zeros(max_d)
            for j, element in enumerate(item):
                subarr[j] = element
            arr[i] = subarr
    """

    padded_arr = [
        np.pad(row, (0, max_len - len(row)), 'constant') 
        for row in arr
    ]

    df = pd.DataFrame(padded_arr)
    df = df.replace(0, df.median())
    
    # for col in df.columns:
    #     fill = df[col].median()
    #     df[df[col] == 0] = fill
    
    return df

arr = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6])]
print(padding(arr))
