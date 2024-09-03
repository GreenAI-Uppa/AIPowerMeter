"""
Recording of a simple experiment where a random image is fed multiple times to a convolutional layer
"""
from codecarbon import OfflineEmissionsTracker
import argparse
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(
    description='Run convolution layer on random input and record the energy consumption'
    )
parser.add_argument('--output_folder',
                    help='directory to save the energy consumption records',
                    default='codecarbon', type=str)
args = parser.parse_args()

#defining a small network with one convolution layer
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



outputdir = args.output_folder 
if os.path.isdir(outputdir):
  os.makedirs(outputdir)
tracker = OfflineEmissionsTracker(country_iso_code="CAN",save_to_file=True, output_dir=outputdir)
tracker.start()
N_ITER = 5000
print('starting to burn the planet')
for i in range(N_ITER):
    out = net(data)
    if i%10000 == 0:
        print(i,'over',N_ITER,'iterations')
tracker.stop()

