import sys, os
from deep_learning_power_measure.power_measure import experiment, parsers
import subprocess
import argparse



def main():
    """small function mainly used to record power consumption of a command by calling directly the module deep_learning_power_measure
    for an example, please see the end section in the quickstart documentation : "TIPS and use cases"
    https://greenai-uppa.github.io/AIPowerMeter/usage/quick_start.html#tips-and-use-cases
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="the running command you want to monitor",type=str)
    parser.add_argument("--output_folder", help="directory to save the energy consumption records",type=str, default="power_measure")
    parser.add_argument("--period", help="time interval used to compute the cpu usage", type=float, default=0.1)
    parser.add_argument("--measurement_period", help="time interval between two recordings for all the metrics (energy, cpu usage, etc). Increase this parameter to reduce the size of the generated jsons", type=float, default=2)
    args = parser.parse_args()
    """TODO parse arguments and make a system call"""
    # parse sys argv to obtain the output folder, and the period
    output_folder = args.output_folder 
    cmd = args.cmd
    driver = parsers.JsonParser(output_folder)
    exp = experiment.Experiment(driver)
    p, q = exp.measure_yourself(period=args.period, measurement_period=args.measurement_period)
    ## calling the experiment script
    try:
      subprocess.run(cmd, shell=True, check=True)
    except:
      print("ERROR : in the measured program, see log before this message" )
    q.put(experiment.STOP_MESSAGE)
    driver = parsers.JsonParser(output_folder)
    exp_result = experiment.ExpResults(driver)
    exp_result.print()
