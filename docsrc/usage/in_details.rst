Advanced use
============

See also the `example scripts <https://github.com/GreenAI-Uppa/AIPowerMeter/tree/main/examples>`_ to test the different metrics provided by the library.

.. _json:

Recorded fields
---------------------

Recording are logged in a json file and include the power draw and the use of the CPU and GPU for the pids related to your experiment. Some of the recordings are done for each pid related to your experiments: `per_process_metric_name : {... pid_i: v_i, ....}`

Unless specified otherwise, the power is logged in watts.

**CPU use**

`cpu_uses`: percentage of cpu clock used by this pid during the recording. 

`mem_use_abs`: Number of bytes used in the CPU RAM. The recording uses psutil in the background, check: :py:func:`deep_learning_power_measure.power_measure.rapl_power.get_mem_uses` for more details.

`mem_use_percent`: Relative number of bytes used in the CPU RAM.

**CPU power**


`Intel_power`: total consumption measured by RAPL

`psys_power`: System on chip consumption

`uncore_power`: other hardware present on the cpu board, for instance, an integrated graphic card. This is NOT the nvidia gpu which is on another board.

`total_cpu_power`: core power consumption.

In other words, you have the following relation: 

.. math::

  Intel\_power = psys + uncore + total\_cpu

For the ram and the core power, we multiply by the cpu and memory use of each pid to get the per process value in the fields `per_process_cpu_power` and `per_process_dram_power`.

Check the :ref:`rapl` section for more details on RAPL domains, and :py:func:`deep_learning_power_measure.power_measure.rapl_power.get_power` for implementation details. The support for different power domains varies according to the processor model, our library will ignore not available domains.

**GPU use**

`sm`: Streaming multiprocessor usage. Analog to the `cpu_uses` for the gpu.

`per_gpu_attributable_mem_use`: Number of bytes used in the GPU

**GPU power**

This is done by the nvidia-smi tool supported by the NVML library of nvidia. 

`nvidia_draw_absolute`: the amount of power used by the whole nvidia card.


model complexity
----------------

We use a wrapper for `torchinfo <https://pypi.org/project/torchinfo/>`_ to extract statistics about your model, essentially number of parameters and mac operation counts.
To obtain them, add additional parameters:
```
net = ... the model you are using for your experiment
input_size = ... (batch_size, *data_point_shape)
exp = experiment.Experiment(driver, model=net, input_size=input_size)
```

You can log the number of parameters and the number of multiply and add (mac) operations of your model. 
Currently, only pytorch is supported, and tensorflow should be supported shortly
