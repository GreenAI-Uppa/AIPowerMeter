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


### displaying the result of the experiment.
# reinstantiating a parser to reload the result.
# a reload function should be used, but this way,
# it shows how to read results from a past experiment
driver = parsers.JsonParser(args.output_folder)
exp_result = experiment.ExpResults(driver)
exp_result.print()
