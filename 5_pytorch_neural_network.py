#imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

#create a neural network class
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    



#set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
print(train_loader.dataset)
# initialize network

model = NN(input_size=input_size, num_classes=num_classes).to(device)


#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#train network

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
       data = data.to(device=device)
       target = target.to(device=device)

       #reshape the data

       data = data.reshape(data.shape[0], -1)

       scores = model(data)
       loss = criterion(scores, target)
       #backward
       optimizer.zero_grad()
       loss.backward()
       

       #gradient descent or adam step
       optimizer.step()




#check accuracy on training & test to see how good our model


def check_accuracy(loader, model):
    if loader.dataset.train:
       print('Checking accuracy on training data')
    else:
       print('Checking accuracy on test data')
    if loader.dataset.train:
       num_correct = 0 
       num_sapmles = 0
       model.eval()

       with torch.no_grad():
           for x, y in loader:
               x = x.to(device=device)
               y = y.to(device=device)
               x = x.reshape(x.shape[0], -1)

               scores = model(x)
               _, predictions = scores.max(1)
               num_correct += (predictions == y).sum()
               num_sapmles += predictions.size(0)
           print(f'Got {num_correct} / {num_sapmles} with accuracy {float(num_correct)/float(num_sapmles)*100:.2f}')
    model.train()

check_accuracy(train_loader, model)
check_accuracy (test_loader, model)