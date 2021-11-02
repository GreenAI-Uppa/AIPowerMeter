"""running the power consumption on a pytorch"""
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
from deep_learning_power_measure.power_measure import experiment, parsers
import pytorch_model_to_test
import os


torch.backends.cudnn.benchmark = True


print("SET DATAFOLDER")
data_folder="/data/pytorch_cifar10/"

test_data = datasets.CIFAR10(root=data_folder,
    train=False,
    download=True,
    transform=ToTensor())

batch_size = 10000 
# getting the data
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=16)

def evaluate(X, y):
    with torch.no_grad():
        X = X.to(pytorch_model_to_test.device)
        pred = pytorch_model_to_test.predict(X)
        pred = pred.to('cpu')
        y = y.to('cpu')
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= y.shape[0] 
    return correct


# this parser will be in charge to write the results to a json file
output_folder = "measure_power"
driver = parsers.JsonParser(output_folder)
# instantiating the experiment.
exp = experiment.Experiment(driver)


# starting the record, and wait two seconds between each energy consumption measurement
# Note that it takes the model and the input as a parameter in order to give a model summary
X, y = next(iter(test_dataloader))
X = X.to(pytorch_model_to_test.device)

accuracy = evaluate(X, y)

p, q = exp.measure_yourself(period=2)

numiter = 40
for i in range(numiter):
    print('iteration ',i, 'over', numiter)
    pytorch_model_to_test.predict(X) 

q.put(experiment.STOP_MESSAGE)

# read the power recordings and print the results
driver = parsers.JsonParser(output_folder)
exp_result = experiment.ExpResults(driver)

exp_result.print()

import json
total_power_draw = exp_result.total_power_draw()
results = {'accuracy':float(accuracy), 'power_draw':total_power_draw}
json.dump(results, open('result.json','w'))

