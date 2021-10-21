Advanced use
============

.. _json:

Recorded fields
---------------------

Recording are logged in a json file and include the power draw and the use of the CPU and GPU for the pids related to your experiment. Some of the recordings are done for each pid related to your experiments: `metric_name : {... pid_i: v_i, ....}`
Unless specified otherwise, the power is logged in watts.

**CPU use**

`cpu_uses`: percentage of cpu clock used by this pid during the recording. 

`mem_use_abs`: Number of bytes used in the CPU RAM. The recording uses psutil in the background, check: :py:func:`deep_learning_power_measure.power_measure.rapl_power.get_mem_uses` for more details.

`mem_use_abs`: Relative number of bytes.

**CPU power**


`Intel_power`: total consumption measured by RAPL

`psys_power`: System on chip consumption

`uncore_power`: some graphic cards, but not nvidia gpu

`total_cpu_power`: core power consumption.

In other words, you have the following relations: 

.. math::

  Intel\_power = psys + uncore + total\_cpu

For the ram and the core power, we multiply by the cpu and memory use of each pid to get the per process value in the fields `per_process_cpu_power` and `per_process_dram_power`.

Check the :ref:`rapl` section for more details on RAPL domains, and :py:func:`deep_learning_power_measure.power_measure.rapl_power.get_power` for implementation details. The support for different power domains varies according to the processor model, our library will ignore not available domains.

**GPU use**

`sm`: Streaming multiprocessor usage. Analog to the `cpu_uses` for the gpu.

`memory`: Number of bytes used in the GPU

**GPU power**

This is done by the nvidia-smi tool supported by the NVML library of nvidia. The record are done per pid and per gpu.

`nvidia_draw_absolute`: the amount of power used by the whole nvidia card.

model complexity
----------------
You can log the number of parameters and the number of multiply and add (mac) operations of your model. 
Currently, only pytorch is supported, and tensorflow should be supported shortly




