"""

Numpy: slicing/indexing
sortign with custom keys
argmax/ bincount: Most frequent label from neighbors
set operations /dict usage : Count classes, majority vote

"""

"""
for (i=0, i<n, 1++)->element wise ====> a = b + c
np.sum() all elemets
np.sum(arr, axis=1) alognt row row-wise


"""

import numpy as np
import pandas as pd


arr = np.array([1,2,3,4])
mean = np.mean(arr)
print(mean)

#replace all negative values in an array to 0

arr = np.array([-1,0,1,-1])
arr [ arr == -1] = 0
print(arr)

df = pd.DataFrame({'A': [3, 8, 2], 'B': [1, 7, 4]})
#select rows where col A >5

df = df[df["A"] > 5]
print(df)

#drop missing values 
df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
cleared_df = df.dropna() #drops rows with any NaN
print(cleared_df)

#noramlize a numpy array (sacle between 0 and 1)
arr = np.array([10, 20, 30])
normalized = (arr - np.min(arr)) / (arr.max() - arr.min())
print(normalized)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
product = A @ B
print(product)
print(np.dot(A, B))

#group by
df = pd.DataFrame({'Category': ['A', 'B', 'A'], 'Value': [10, 20, 30]})
group = df.groupby("Category")["Value"].mean()
#merge two Dataframe on a common column
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [2, 3], 'Age': [25, 30]})
df_merged = df1.merge(df2, how="inner", on="ID") #how : left,right, inner merge = join
print(df_merged)
#euclidean distance 
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
distance = np.linalg.norm(a) #from the origin sqrt(a(is) **2)
distance_manhatan = np.linalg.norm(a, ord=1)
print(distance_manhatan)

#usabale for vector normalizing
a_norm = a / np.linalg.norm(a) # unit vector
print(a_norm)

#top3 largest 
arr = np.array([[1, 5, 3], [10, 2, 8]])
top3 = np.sort(arr, axis=1)[:, -3:]

df = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02'],
    'Variable': ['A', 'B', 'A'],
    'Value': [10, 20, 30]
})

df_pivot = df.pivot(index="Date", columns="Variable", values="Value")
print(df_pivot)

#outlier handeling 
df = pd.DataFrame({'Price': [100, 200, 1000, 150]})
print("outliers")
df = pd.DataFrame({'Price': [100, 200, 1000, 150]})
cap = df['Price'].quantile(0.8)
print(cap)
df['Price'] = np.where(df['Price'] > cap, cap, df['Price'])
print(df["Price"])

df = pd.DataFrame({
    'login_time': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00', '2023-01-02 09:00'])
})
#number of logins per day
logins_per_day = df['login_time'].dt.date.value_counts()
print(logins_per_day)

#values in ‘Age’ with the median of ‘Gender’ groups
df = pd.DataFrame({'Gender': ['F', 'F', 'M'], 'Age': [25, np.nan, 30]})
# df = df["Age"].fillna(np.mean(df["Age"]))
# print(df)
df['Age'] = df.groupby('Gender')['Age'].transform("max")
print(df["Age"])


