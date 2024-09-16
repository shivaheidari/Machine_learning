import torch

x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])

#addition
z = torch.empty(3)
torch.add(x, y, out=z)
z = torch.add(x, y)

#subtraction
z = x - y

#division
z = torch.true_divide(x, y) # element wise division if the shapes are the same

#inplace operations
t = torch.zeros(3)
t.add_(x) # t = t + x
t += x  # t = t + x (not inplace)

#exponentiation
z = x.pow(2) # element wise x^2
z = x ** 2 # element wise x^2

#comparision
z = x > 0
z = x < 0

#matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2x3
x3 = x1.mm(x2) # 2x3

#matrix exponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3) # 3rd power of matrix

#element wise multiplication
z = x * y

#dot product
z = torch.dot(x, y) # 1*4 + 2*5 + 3*6

#batch matrix multiplication
Batch = 32
n=10
m=20
p=30

tensor1 = torch.rand((Batch, n, m))
tensor2 = torch.rand((Batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # (Batch, n, p)

#example of broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2 # x2 is broadcasted to 5x5

#other useful tensor operations
sum_x = torch.sum(x, dim=0) # sum of all elements
values, indices = torch.max(x, dim=0) # max value and index
values, indices = torch.min(x, dim=0) # min value and index
abs_x = torch.abs(x) # absolute value
z = torch.argmax(x, dim=0) # index of maximum value
z = torch.argmin(x, dim=0) # index of minimum value
mean_x = torch.mean(x.float(), dim=0) # mean x
z = torch.eq(x, y) # element wise comparison
torch.sort(y, dim=0, descending=False) # sort y

z = torch.clamp(x, min=0, max=10) # all elements < 0 set to 0 and > 10 set to 10

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x) # True
z = torch.all(x) # False
