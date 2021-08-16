from deep_learning_power_measure.power_measure import experiment, parsers
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Lin(nn.Module):
    def __init__(self, input_dim):
        super(Lin, self).__init__()
        self.fc = nn.Linear(input_dim, 10)
    def forward(self, x):
        x = torch.flatten(x)
        return self.fc(x)

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(3,1, (3,3)) # output torch.Size([1, 6, 30, 30])
    def forward(self, x):
        return self.conv(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): # the backward function will be computed automatically with autograd
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



input_size = 1, 3, 128, 128
name = "conv_1_3_3"
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

net = Conv()
net.to(device)
# The learnable parameters of a model are returned by net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
outdir = 'results'

# generating some fake data
data = torch.randn(input_size) # has to be a minibatch: nSamples x nChannels x Height x Width
data = data.to(device)

driver = parsers.JsonParser('/home/paul/data/power_measure')
exp = experiment.Experiment(driver, model=net, input_size=input_size)
p, q = exp.measure_yourself(period=2)
## starting the experiment
n_iter = 200000
print('starting to burn the planet')
start_exp = time.time()
for i in range(n_iter):
    out = net(data)
    if i%10000 == 0:
        print(i,'over',n_iter,'iterations')
end_exp = time.time()
q.put(experiment.STOP_MESSAGE)

from deep_learning_power_measure.power_measure import experiment, parsers

driver = parsers.JsonParser('/home/paul/data/power_measure')
exp_result = experiment.ExpResults(driver)
exp_result.print()
