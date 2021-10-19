Advanced use
============

Experimental results
---------------------


The json logs power and use of the CPU and GPU for the pids related to your experiment.


Most of the recordings are done for each pid related to your experiments: `metric_name : {... pid_i: v_i, ....}`



**CPU use**

`cpu_uses`: percentage of cpu clock used by this pid during the recording. 

`mem_use_abs`: Number of bytes. The recording uses psutil in the background, check: :py:func:`deep_learning_power_measure.power_measure.rapl_power.get_mem_uses` for more details.

`mem_use_abs`: Relative number of bytes.

**CPU power**

The following metrics are in Watts:

`Intel_power`: total consumption measured by RAPL
`psys_power`: System on chip consumption
`uncore_power`: some graphic cards, but not nvidia gpu
`total_cpu_power`: core power consumption.

In other words, you have the following relations: 

.. math::

  Intel\_power = psys + uncore + total\_cpu



For the ram and the core power, we multiply by the cpu and memory use of each pid to get the per process value in the fields `per_process_cpu_power` and `per_process_dram_power`.

Check the :ref:`rapl` section for more details on RAPL domains, and :py:func:`deep_learning_power_measure.power_measure.rapl_power.get_power` for implementation details.

**GPU use**

sm

memory

**GPU power**

This is done by the nvidia-smi tool supported by the NVML library of nvidia. The record are done per pid and per gpu.

`nvidia_draw_absolute`: the amount of power used by the whole nvidia card.
`streaming multiprocessors`: streaming multiprocessors used. Analog to `cpu_uses` for the gpu.
`per_gpu_average_estimated_utilization_absolute`: contains the memory and the Streaming multiprocessor (SM) used. The latter is the analog to the `cpu_uses` for the gpu.


Processing the resulting files
------------------------------

Linking energy and computation usage
------------------------------------

