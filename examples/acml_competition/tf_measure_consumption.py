"""running the power consumption on a pytorch"""
import tensorflow as tf 
from tensorflow.keras import datasets
import tensorflow.keras.backend as K
import tf_model_to_test
from deep_learning_power_measure.power_measure import experiment, parsers



# getting the data
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()


def evaluate(test_images, test_labels):
    pred = tf_model_to_test.predict(test_images)
    pred = tf.cast(tf.argmax(pred, axis=1), dtype = 'uint8')
    accuracy = K.mean(tf.equal(pred, tf.transpose(test_labels)[0]))
    return accuracy

accuracy = evaluate(test_images, test_labels)

# this parser will be in charge to write the results to a json file
output_folder = "measure_power"
driver = parsers.JsonParser(output_folder)
# instantiating the experiment.
exp = experiment.Experiment(driver)


# starting the record, and wait two seconds between each energy consumption measurement
# Note that it takes the model and the input as a parameter in order to give a model summary
p, q = exp.measure_yourself(period=2)
for i in range(20):
    print('iteration ',i)
    tf_model_to_test.predict(test_images)

q.put(experiment.STOP_MESSAGE)
## end of the experiment

### displaying the result of the experiment.
# reinstantiating a parser to reload the result.
# a reload function should be used, but this way,
# it shows how to read results from a past experiment
driver = parsers.JsonParser(output_folder)
exp_result = experiment.ExpResults(driver)
exp_result.print()


import json
total_power_draw = exp_result.total_power_draw()
results = {'accuracy':float(accuracy), 'power_draw':total_power_draw}
json.dump(results, open('result.json','w'))
