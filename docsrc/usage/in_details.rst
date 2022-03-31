Advanced use
============

See also the `example scripts <https://github.com/GreenAI-Uppa/AIPowerMeter/tree/main/examples>`_ to test the different metrics provided by the library.

.. _json:

Recorded fields
---------------------

Recordings are logged in a json file and include the power draw and the use of the CPU and GPU for the pids related to your experiment. Some of the recordings are done for each pid related to your experiments: `per_process_metric_name : {... pid_i: v_i, ....}`. However, the monitoring of multiple programs on the same device should be done with care (see :ref:`multiple`).

Unless specified otherwise, the power is logged in watts.

**CPU use**

- `cpu_uses`: percentage of cpu clock used by this pid during the recording. 

- `mem_use_abs`: Number of bytes used in the CPU RAM. The recording uses psutil in the background, check: :py:func:`deep_learning_power_measure.power_measure.rapl_power.get_mem_uses` for more details.

- `mem_use_percent`: Relative number of bytes used in the CPU RAM.

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

- `per_gpu_attributable_mem_use` : memory usage for each gpu
- `per_gpu_per_pid_utilization_absolute` : absolute % of Streaming Multiprocessor (SM) used per gpu per pid
- `per_gpu_absolute_percent_usage` : absolute % of SM used per gpu for the given pid list
- `per_gpu_estimated_attributable_utilization` : relative use of SM used per gpu by the experiment

**GPU power**

This is done by the nvidia-smi tool supported by the NVML library of nvidia. 

`nvidia_draw_absolute` : total nvidia power draw for all the gpus per_gpu_power_draw : nvidia power draw per gpu
`nvidia_draw_absolute`: the amount of power used by the whole nvidia card.


model complexity
----------------

We use a wrapper for `torchinfo <https://pypi.org/project/torchinfo/>`_ to extract statistics about your model, essentially number of parameters and mac operation counts.
To obtain them, add additional parameters:

.. code-block:: python

  net = ... the model you are using for your experiment
  input_size = ... (batch_size, *data_point_shape)
  exp = experiment.Experiment(driver, model=net, input_size=input_size)


You can log the number of parameters and the number of multiply and add (mac) operations of your model. 
Currently, only pytorch is supported, and tensorflow should be supported shortly

.. _docker:

Docker integration
---------------------

For the implementation of AIPowerMeter in a docker container, we need to use a special branch of the code because of the behaviour of the command :

.. code-block:: console

  $ nvidia-smi pmon

An hot fix has been implemented, it forces the tracking of all the GPU processes. It's then impossible to isolate a process running at the same time than others.

See the github repo `docker_AIPM <https://github.com/GreenAI-Uppa/docker_AIPM>`_ for more details. You will also find slides explaining the motivations for the use of Docker images and container.
