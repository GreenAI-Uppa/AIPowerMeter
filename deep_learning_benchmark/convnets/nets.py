import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self,channels,nb_filters,nb_conv_layers):
        super(ConvNet, self).__init__()
        self.channels = channels
        self.nb_filters = nb_filters
        self.nb_conv_layers = nb_conv_layers
        self.conv_layers = nn.ModuleList()
        for k in range(nb_conv_layers):
            if k==0:
                print(k)
                self.conv_layers.append(nn.Conv2d(in_channels=channels, out_channels=nb_filters, kernel_size=3,stride=1, padding=1))
            else:
                print(k)
                self.conv_layers.append(nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=3,stride=1, padding=1))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(nb_filters * 224 * 224, 256)  # 5*5 from image dimension
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        for u,layer in enumerate(self.conv_layers):
            print('inf layer',u)
            x = F.relu(layer(x))
            print('output size',x.shape)
        print('input size required',self.nb_filters * 224 * 224)
        x = x.view(x.size(0), -1) 
        output = F.relu(self.fc1(x))
        print('output size',output.shape)
        print('input size required',256)
        out = F.softmax(self.fc2(output), dim=1)
        return out


