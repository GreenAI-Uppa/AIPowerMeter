"""
Recording of a simple experiment where a random image is fed multiple times to a convolutional layer
"""
import argparse
import torch
import torch.nn as nn
from deep_learning_power_measure.power_measure import experiment, parsers

parser = argparse.ArgumentParser(
    description='Run convolution layer on random input and record the energy consumption'
    )
parser.add_argument('--output_folder',
                    help='directory to save the energy consumption records',
                    default='measure_power', type=str)
args = parser.parse_args()


# defining a small network with one convolution layer
class Conv(nn.Module):
    """one conv layer model"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,3, (3,3))
    def forward(self, input_data):
        """forward pass"""
        return self.conv(input_data)
net = Conv()


# setting the size of the input
# minibatch_size x nChannels x Height x Width
input_size = 32, 3, 128, 128
# generating some fake data
data = torch.randn(input_size)
# using gpu if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
net.to(DEVICE)
data = data.to(DEVICE)

# this parser will be in charge to write the results to a json file
driver = parsers.JsonParser(args.output_folder, meta_data="Simple experiment with a convnet and fake input data")
# instantiating the experiment.
exp = experiment.Experiment(driver)

# starting the record, and wait two seconds between each energy consumption measurement
# Note that it takes the model and the input as a parameter in order to give a model summary
p, q = exp.measure_yourself(period=2, measurement_period=2)

## starting the experiment
N_ITER = 100000 
print('starting to burn the planet')
for i in range(N_ITER):
    out = net(data)
    if i%10000 == 0:
        print(i,'over',N_ITER,'iterations')
q.put(experiment.STOP_MESSAGE)
## end of the experiment

### displaying the result of the experiment.
# reinstantiating a parser to reload the result.
# a reload function should be used, but this way,
# it shows how to read results from a past experiment
driver = parsers.JsonParser(args.output_folder)
exp_result = experiment.ExpResults(driver)
exp_result.print()
