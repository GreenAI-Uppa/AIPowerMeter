import sys, os
from deep_learning_power_measure.power_measure import experiment, parsers
import subprocess

def main():
    """TODO parse arguments and make a system call"""
    # parse sys argv to obtain the output folder, and the period
    output_folder = "power_measure"
    period = 2
    driver = parsers.JsonParser(output_folder)
    exp = experiment.Experiment(driver)
    p, q = exp.measure_yourself(period=period)
    ## calling the experiment script
    cmd = " ".join(sys.argv[1:])
    subprocess.run(cmd, shell=True, check=True)
    q.put(experiment.STOP_MESSAGE)
    driver = parsers.JsonParser(output_folder)
    exp_result = experiment.ExpResults(driver)
    exp_result.print()
