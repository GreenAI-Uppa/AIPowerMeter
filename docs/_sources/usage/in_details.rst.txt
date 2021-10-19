In details
==========




Experimental results
---------------------


The json logs power and use of the CPU and GPU for the pids related to your experiment.


Most of the recordings are done for each pid related to your experiments: `metric_name : {... pid_i: v_i, ....}`


**CPU use**

`cpu_uses`: percentage of cpu clock used by this pid during the recording. 

`mem_use_abs`: Number of bytes. The recording uses psutil in the background, check: :py:func:`deep_learning_power_measure.power_measure.rapl_power.get_mem_uses` for more details.

`mem_use_abs`: Relative number of bytes.

**CPU power**
check for more details on RAPL domains

`Intel_power` :






Threading 
------------

.. autofunction:: deep_learning_power_measure.power_measure.experiment.Experiment



Linking energy and computation usage
------------------------------------

Drivers
--------
