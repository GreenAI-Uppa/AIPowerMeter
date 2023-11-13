"""
Recording of a simple experiment where a random image is fed multiple times to a convolutional layer
"""
import argparse, os, time

from deep_learning_power_measure.power_measure import experiment, parsers
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


parser = argparse.ArgumentParser(
    description='Simulate an experiment where you record the train validation and test phases separately. WARNING, this should not be used when phases have short duration, like less than one minute, because of inaccurate results'
    )
parser.add_argument('--output_folder',
                    help='directory to save the energy consumption records',
                    default='measure_power', type=str)
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)



# this parser will be in charge to write the results to a json file
train_driver = parsers.JsonParser(os.path.join(args.output_folder,'train'))
train_exp = experiment.Experiment(train_driver)
val_driver = parsers.JsonParser(os.path.join(args.output_folder,'val'))
val_exp = experiment.Experiment(val_driver)
test_driver = parsers.JsonParser(os.path.join(args.output_folder,'test'))
test_exp = experiment.Experiment(test_driver)

n_epochs = 2
for epoch in range(n_epochs):
    #### training phase
    p, q = train_exp.measure_yourself(period=0.1)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    q.put(experiment.STOP_MESSAGE)

    #### test phase
    p, q = test_exp.measure_yourself(period=0.1)

    total, correct = 0, 0
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    q.put(experiment.STOP_MESSAGE)
    #q.get(block=True)


print('TRAINING PHASE CONSUMPTION')
experiment.ExpResults(train_driver).print()

print('TESTING PHASE CONSUMPTION')
experiment.ExpResults(test_driver).print()
