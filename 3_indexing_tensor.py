import torch


batch_size = 10
features = 25
x = torch.randn(batch_size, features)

print(x[0].shape) # 1st batch
print(x[:, 0].shape) # 1st feature

# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8] # list of indices
print(x[indices]) # elements at 2, 5, 8

x = torch.rand((3, 5))
print(x)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols]) # elements at (1,4) and (0,0)

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)]) # elements <2 or >8
print(x[x.remainder(2) == 0]) # even elements

torch.where(x > 5, x, x*2) # if x>5 x else x*2
torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique() # unique elements
x.ndimension() # number of dimensions
x.numel() # number of elements