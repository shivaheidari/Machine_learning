import numpy as np

arr = np.array([1,2,4])
z = np.zeros((3,2))
ones = np.ones((3,2))
range = np.arange(0,10,2)
li = np.linspace(0, 1, 5)
rn = np.random.rand(3, 3)
rng = np.random.randn(3,3) #guasian
np.eye(3)


#--------------
a = np.array([1,2,4])
b = np.array([4,4,2])

print(a + b)
print(a * b) #entry by entry

print(a.dot(b)) #dot and sigma
print(np.sqrt(a))

print(a.sum())
print(a.max())
print(np.max(a))

print(np.mean(a))
print(np.argmax(a)) #returns index
print(a[np.argmax(a)]) 

#linar algebra

x = np.random.rand(3, 5)
y = np.random.rand(5)

#print(x.T)
print("mul", x @ y)

print("dot", x * y)

print(np.linalg.inv(x.T @ x))


#indexing and slicing

a = np.array([[1, 2, 9], [4, 5, 7]])

print(a[0,1])
print(a[:, 1:3])
print("a", a[a > 3])

print(a.reshape(3,2))
a.flatten()
print(a.shape)

#random resampling

# np.random.seed(42)
# np.random.normal(0, 1, (2,2)) #mean, std
# idx = np.random.permutation(100) #random permutation of indexes
# x_train, y_train = x[idx[:80]], y[idx[:80]]


#tricks
x = np.random.randn(3, 5)
print("shape", x.shape)
x_wiht_bias = np.c_[np.ones(x.shape[0]), x]
print(x_wiht_bias)
y = np.random.randint(0, 2, size=10)
print(y)

y = np.random.choice([0,1], size=10, p=[0.3, 0.7])
print(y)
# one_hot_encoding = np.eye(3)[y.astype(int)]
# print(one_hot_encoding)

