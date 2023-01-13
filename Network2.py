from torch import nn
import torch
from torchvision import datasets, transforms


transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# model defining
model = nn.Sequential(
                    nn.Linear(784,128), 
                    nn.ReLU(),
                    nn.Linear(128,64),
                    nn.ReLU(), 
                    nn.Linear(64,10),
                    nn.LogSoftmax(dim=1)
                    
)

# defining loss function
criteria = nn.NLLLoss()
 
#data 
images, labels = next(iter(trainloader))

# Flattening
images = images.view(images.shape[0],-1)       

# forward and get logits
logits = model(images)  

loss = criteria(logits,labels) 

print(loss)