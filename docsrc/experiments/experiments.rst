Insights into energy consumption
====================================

Experimental protocol
---------------------

We've started to run experiments to measure the energy consumption of classical deep learning pretrained model at inference. Our protocol acts as follows:

- we load a pretrained architecture,

- we select an input size (resolution for Computer Vision, number of tokens for NLP),

- we run x inferences and measure power draws with AIPowerMeter,

- we repeat the experiment 10 times to have more robustness.

For each set of experiments, power measurements are written into severals power_metrics.json files (one by tuple (input_size,experiment). We then compile the median over the 10 runs and generate a csv file where each row corresponds to one input size and each column represents the median of one measure.  


Alexnet study
--------------



Resnet study
------------




Bert Transformers
-----------------
