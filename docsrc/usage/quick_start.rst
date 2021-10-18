Quick start
===========

Hardware requirements
---------------------
**CPU** power measure is done with RAPL. 
Support is ensured on intel processor since Sandy Bridge architecture.


To see if your processor is compatible, first check that the CPU is a GenuineIntel:

.. code-block:: console

 $ cat /proc/cpuinfo | grep vendor | uniq | awk '{print $3}'
 GenuineIntel


**GPU** will be measured by nvidia-smi. 
Again, not all gpu cards (for ex: Jetson Nano board) include the required sensors.

A quick check is to run 
.. code-block:: console

 $ nvidia-smi -q -x 

and search if the xml output contains values at the "power_readings" field.

Installation
------------


.. code-block:: console

   $ git clone https://github.com/GreenAI-Uppa/AIPowerMeter.git
   $ pip install -r requirements.txt
   $ python setup.py install


Measuring my first program
--------------------------

See `examples/example_exp_deep_learning.py <https://github.com/GreenAI-Uppa/AIPowerMeter/blob/main/examples/example_exp_deep_learning.py>`_.

Essentially, you instantiate an experiment and place the code you want to measure between a start and stop signal.

.. code-block:: python

  from deep_learning_power_measure.power_measure import experiment, parsers

  driver = parsers.JsonParser("output_folder")
  exp = experiment.Experiment(driver)

  p, q = exp.measure_yourself(period=2)
  ###################
  #  place here the code that you want to profile
  ################
  q.put(experiment.STOP_MESSAGE)

