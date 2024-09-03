# Measure the efficiency of your deep learning

Record the energy consumption of your cpu and gpu. Check [our documentation](https://greenai-uppa.github.io/AIPowerMeter/) for usage.

This repo is largely inspired from this [experiment Tracker](https://github.com/Breakend/experiment-impact-tracker) 

## Requirements

Running Average Power Limit (RAPL) and its linux interface : powercap 

RAPL is introduced in the Intel processors starting with the Sandy bridge architecture in 2011. 

Your linux os supports RAPL if the following folder is not empty:
```
/sys/class/powercap/intel-rapl/
```

Empty folder? If your cpu is very recent, it is worth to check the most recent linux kernels.

## Installation

```
pip install -r requirements.txt
pip install --force-reinstall --no-cache-dir .
```

You need to authorize the reading of the rapl related files: 
```
sudo chmod -R 755 /sys/class/powercap/intel-rapl/
```


> some examples requires pytorch or tensorflow.
## Usage

See `examples/example_exp_deep_learning.py`.

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

