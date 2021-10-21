Quick start
===========

Hardware requirements
---------------------
**CPU** power measure is done with RAPL

Support is ensured on intel processor since Sandy Bridge architecture.


To see if your processor is compatible, first check that the CPU is a GenuineIntel:

.. code-block:: console

 $ cat /proc/cpuinfo | grep vendor | uniq | awk '{print $3}'
 GenuineIntel


In linux, RAPL will log the energy consumption in  `/sys/class/powercap/intel-rapl`

Change the permissions so that our program can read these logs:

.. code-block:: console

  $ sudo chmod -R 755 /sys/class/powercap/intel-rapl


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

We provide examples for pytorch and tensorflow, but the model creation is independant from the power recording part.


If you wish to get the number of parameters and mac operations of your pytorch model, also install: 

.. code-block:: console

   $ pip install torchinfo

Measuring my first program
--------------------------

You have an example `there <https://github.com/GreenAI-Uppa/AIPowerMeter/blob/main/examples/example_exp_deep_learning.py>`_, in a nutshell,

 you instantiate an experiment and place the code you want to measure between a start and stop signal.

.. code-block:: python

  from deep_learning_power_measure.power_measure import experiment, parsers

  driver = parsers.JsonParser("output_folder")
  exp = experiment.Experiment(driver)

  p, q = exp.measure_yourself(period=2) # measure every 2 seconds
  ###################
  #  place here the code that you want to profile
  ################
  q.put(experiment.STOP_MESSAGE)

This will create a directory `output_folder` in which a `power_metrics.json` will contain the power measurements. See the section :ref:`json` for details on this file. If it already exists, the content folder will be replaced. So you should have one folder per experiment.
You can then get a summary of the recordings

.. code-block:: python

  from deep_learning_power_measure.power_measure import experiment, parsers
  driver = parsers.JsonParser(output_folder)
  exp_result = experiment.ExpResults(driver)
  exp_result.print()

and the console output should looks like: 

.. code-block:: console

  ================= EXPERIMENT SUMMARY ===============
  MODEL SUMMARY:  28 parameters and  444528 mac operations during the forward pass

  ENERGY CONSUMPTION:
  on the cpu

  RAM consumption not available. Your usage was  4.6GiB with an overhead of 4.5GiB
  Total CPU consumption: 107.200 joules, your experiment consumption:  106.938 joules
  total intel power:  146.303 joules
  total psys power:  -4.156 joules


  on the gpu
  nvidia total consumption: 543.126 joules, your consumption:  543.126, average memory used: 1.6GiB
