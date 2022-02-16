"""
Recording of a simple experiment where a random image is fed multiple times to a convolutional layer
"""
import argparse
import numpy as np
import os
from deep_learning_power_measure.power_measure import experiment, parsers

parser = argparse.ArgumentParser(
    description='To illustrate how the consumption can be recorded for non python program, here is the measure the consumption of a small R script. R must be previously installed.'
    )
parser.add_argument('--output_folder',
                    help='directory to save the energy consumption records',
                    default='measure_power', type=str)
args = parser.parse_args()




# this parser will be in charge to write the results to a json file
driver = parsers.JsonParser(args.output_folder)
# instantiating the experiment.
exp = experiment.Experiment(driver)

# starting the record, and wait two seconds between each energy consumption measurement
p, q = exp.measure_yourself(period=2)
os.system('Rscript examples/matmul_r.m')
q.put(experiment.STOP_MESSAGE)
## end of the experiment

### displaying the result of the experiment.
# reinstantiating a parser to reload the result.
# a reload function should be used, but this way,
# it shows how to read results from a past experiment
driver = parsers.JsonParser(args.output_folder)
exp_result = experiment.ExpResults(driver)
exp_result.print()
