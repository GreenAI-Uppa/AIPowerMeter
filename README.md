# deep_learning_power_measure

This repo contains python modules in the backstage linux powercap interface of RAPL and nvidia-smi to measure cpu and gpu energy consumption
It is largely inspired from this [experiment Tracker](https://github.com/Breakend/experiment-impact-tracker) 

## requirements

RAPL is introduced in the Intel processors since SkyLake. 

To check if your linux os is supporting RAPL, you can check that the following folder is not empty.
```
/sys/class/powercap/intel-rapl/
```

## installation

this repo has been tested with torch 1.9.0
```
pip install -r requirements.txt
python setup.py install
```


## Usage

See `examples/example_exp_deep_learning.py` to run and measure the energy consumption of one experiment. 

Essentially, you instantiate an experiment and place the code you want to measure between a start and stop signal.

```
from deep_learning_power_measure.power_measure import experiment, parsers

model = ... define your pytorch model
input_size = ... this information enables count the number of mac operations
driver = parsers.JsonParser("output_folder_for_power_recordings")
exp = experiment.Experiment(driver)

p, q = exp.measure_yourself(period=2, model=net, input_size=input_size)
###################
#  place here the code that you want to profile
################
q.put(experiment.STOP_MESSAGE)

``` 
