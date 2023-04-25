import sys, os
from deep_learning_power_measure.power_measure import experiment, parsers
import subprocess
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="the running command you want to monitor",type=str)
    parser.add_argument("--output_folder", help="directory to save the energy consumption records",type=str, default="power_measure")
    args = parser.parse_args()
    """TODO parse arguments and make a system call"""
    # parse sys argv to obtain the output folder, and the period
    output_folder = args.output_folder 
    cmd = args.cmd
    period = 2
    driver = parsers.JsonParser(output_folder)
    exp = experiment.Experiment(driver)
    p, q = exp.measure_yourself(period=period)
    ## calling the experiment script
    subprocess.run(cmd, shell=True, check=True)
    q.put(experiment.STOP_MESSAGE)
    driver = parsers.JsonParser(output_folder)
    exp_result = experiment.ExpResults(driver)
    exp_result.print()
