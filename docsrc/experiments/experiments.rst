Insights into energy consumption
====================================

Experimental protocol
---------------------

We've started to run experiments to measure the energy consumption of classical deep learning pretrained model at inference. Our protocol acts as follows:

- we load a pretrained architecture,

- we select an input size (resolution for Computer Vision, number of tokens for NLP),

- we run x inferences and measure power draws with AIPowerMeter,

- we repeat the experiment 10 times to have more robustness.

For each set of experiments, power measurements are written into severals power_metrics.json files (one by tuple (input_size,experiment). We then compile  `here <https://github.com/GreenAI-Uppa/AIPowerMeter/blob/main/power_metrics_management/concat_power_measure.py>`_ an estimate of different power draws of one inference and compile the median of the over the 10 runs. For each pretrained model, results are generated into a csv file where each row corresponds to one input size and each column represents the median of one power draw.  


Alexnet study
--------------



Resnet study
------------




Bert Transformers
-----------------
