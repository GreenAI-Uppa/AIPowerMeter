## ACML 2021 Evaluation framework


As indicated in the [competition webpage](https://greenai-uppa.github.io/acml_competition/#evaluation), the evaluation will perform multiple runs of inference on the full cifar dataset and compute the power draw during these runs.

More concretely, it relies on two scripts.
- `*_model_to_test.py` : competitors are expected to provide this module. It will load the model and provide a `predict` function which will take as input a tensor containing the cifar dataset and return the predictions.  
- `*_measure_consumption.py` : this is the main script. It will use the predict function, perform the accuracy computation, and record the power draw. The results are stored in a json file `result.json`. 

To test the evaluation, just run 
```
python [pytorch|tf]_measure_consumption.py
```

This will create a `measure_power` folder with the log of your energy consumption. You can modify the folder name inside the script.


Note: The IAPowerMeter library should be [installed](https://greenai-uppa.github.io/AIPowerMeter/usage/quick_start.html)

In the evaluation, we will use the following versions: 

- pytorch : 1.9.0+cu111
- tensorflow : 2.5.0


## for non tensorflow/pytorch users. 

For other frameworks, we ask you to provide a docker image so that it can run smoothly on our servers.
