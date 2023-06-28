import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_learning_power_measure.power_measure import model_complexity 

# defining a small network with one convolution layer
class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(3,1, (3,3)) 
    def forward(self, x):
        return self.conv(x)

# setting the size of the input
# has to be a minibatch: nSamples x nChannels x Height x Width
input_size = 1, 3, 128, 128

net = Conv()
# generating some fake data
data = torch.randn(input_size) 

# unsing gpu if available
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#data = data.to(device)
#net.to(device)

model_card = model_complexity.get_summary(net, input_size)
print(model_card)
