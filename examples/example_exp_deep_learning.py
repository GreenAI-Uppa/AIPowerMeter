from deep_learning_power_measure.power_measure import experiment, parsers
import time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Run a convolution layer on random input and record the energy consumption')
parser.add_argument('--output_folder',
                    help='directory to save the energy consumption records',
                    default='measure_power', type=str)
args = parser.parse_args()


# defining a small network with one convolution layer
class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(3,1, (3,3))
    def forward(self, x):
        return self.conv(x)
net = Conv()


# setting the size of the input
# minibatch_size x nChannels x Height x Width
input_size = 1, 3, 128, 128
# generating some fake data
data = torch.randn(input_size)

# this parser will be in charge to write the results to a json file
driver = parsers.JsonParser(args.output_folder)
# instantiating the experiment.
# Note that it takes the model and the input as a parameter in order to give a model summary
exp = experiment.Experiment(driver) 

# starting the record, and wait two seconds between each energy consumption measurement
p, q = exp.measure_yourself(period=2, model=net, input_size=input_size)
# using gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
net.to(device)
data = data.to(device)

## starting the experiment
n_iter = 200000
print('starting to burn the planet')
for i in range(n_iter):
    out = net(data)
    if i%10000 == 0:
        print(i,'over',n_iter,'iterations')
q.put(experiment.STOP_MESSAGE)
## end of the experiment

### displaying the result of the experiment.
from deep_learning_power_measure.power_measure import experiment, parsers
# reinstantiating a parser to reload the result.
# a reload function should be used, but this way, it shows how to read results from a past experiment
driver = parsers.JsonParser(args.output_folder)
exp_result = experiment.ExpResults(driver)
exp_result.print()
