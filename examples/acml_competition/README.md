## ACML 2021 Evaluation framework

As indicated in the [competition webpage](https://greenai-uppa.github.io/acml_competition/#evaluation), the evaluation will perform multiple runs of inference on the full cifar dataset and compute the power draw during these runs.

More concretely, it relies on two scripts.
- `model_to_test.py` : competitors are expected to provide this module. It will load the model and provide a `predict` function which will take as input a tensor containing the cifar dataset and return the predictions.  
In practice, you can launch the script `tf_measure_consumption.py` to perform an evaluation. 
- `tf_measure_consumption.py` : this script will use the predict function, perform the accuracy computation, and record the power draw. The results are stored in a json file `result.json`. You can use the template provided on this repo or provide your own if you are using another framework such as pytorch


To test the evaluation, just run 
```
python tf_measure_consumption.py
```

Note: The IAPowerMeter library should be [installed](https://greenai-uppa.github.io/AIPowerMeter/usage/quick_start.html)


