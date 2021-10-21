"""
Recording of a simple experiment where a random image is fed multiple times to a convolutional layer
"""
import argparse
import numpy as np
from deep_learning_power_measure.power_measure import experiment, parsers

parser = argparse.ArgumentParser(
    description='Run convolution layer on random input and record the energy consumption'
    )
parser.add_argument('--output_folder',
                    help='directory to save the energy consumption records',
                    default='measure_power', type=str)
args = parser.parse_args()



# setting the size of the input
# minibatch_size x nChannels x Height x Width
input_size = (256, 256)
# generating some fake data
mat1 = np.random.rand(input_size[0], input_size[1])
mat2 = np.random.rand(input_size[0], input_size[1])

# this parser will be in charge to write the results to a json file
driver = parsers.JsonParser(args.output_folder)
# instantiating the experiment.
exp = experiment.Experiment(driver)

# starting the record, and wait two seconds between each energy consumption measurement
p, q = exp.measure_yourself(period=2)

## starting the experiment
N_ITER = 100000
print('starting to burn the planet')
for i in range(N_ITER):
    np.matmul(mat1, mat2)
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
