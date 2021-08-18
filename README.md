# Measure the efficiency of your deep learning

Record the energy consumption of your cpu and gpu. 

This repo is largely inspired from this [experiment Tracker](https://github.com/Breakend/experiment-impact-tracker) 

## Requirements

Running Average Power Limit (RAPL) and its linux interface : powercap 

RAPL is introduced in the Intel processors starting with the SkyLake version. 

Your linux os supports RAPL if the following folder is not empty:
```
/sys/class/powercap/intel-rapl/
```

Empty folder? If your cpu is very recent, it is worth to check the most recent linux kernels.

## Installation

Install pytorch, then,
```
pip install -r requirements.txt
python setup.py install
```

You need to authorize the reading of the rapl related files: 
```
sudo chmod -R 755 /sys/class/powercap/intel-rapl/
```

## Usage

See `examples/example_exp_deep_learning.py` to run and measure the energy consumption of one experiment. 

Essentially, you instantiate an experiment and place the code you want to measure between a start and stop signal.

```
from deep_learning_power_measure.power_measure import experiment, parsers

driver = parsers.JsonParser("output_folder")
exp = experiment.Experiment(driver)

p, q = exp.measure_yourself(period=2)
###################
#  place here the code that you want to profile
################
q.put(experiment.STOP_MESSAGE)

``` 

This will save the recordings as json file in the `output_folder`. You can display them with: 

```
from deep_learning_power_measure.power_measure import experiment, parsers
driver = parsers.JsonParser(output_folder)
exp_result = experiment.ExpResults(driver)
exp_result.print()
``` 
### model card
We use a wrapper to [torchinfo](https://pypi.org/project/torchinfo/) to provide statistics about your model. 
To obtain them, add additional parameters:
```
net = ... the model you are using for your experiment 
input_size = ... (batch_size, *data_point_shape)
p, q = exp.measure_yourself(period=2, model=net, input_size=input_size)

```
