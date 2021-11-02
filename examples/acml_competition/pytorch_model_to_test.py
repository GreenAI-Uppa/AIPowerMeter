from torchvision import models
from torch import nn
import torch


##### load your own model trained on cifar10
# here we just take pretrained imagenet weights
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def predict(X): #, model):
    """run inference on a batch"""
    #X = X.to(device)
    return model(X)
