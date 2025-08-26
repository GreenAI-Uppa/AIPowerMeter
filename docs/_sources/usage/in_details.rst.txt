Advanced use
============

See also the `example scripts <https://github.com/GreenAI-Uppa/AIPowerMeter/tree/main/examples>`_ to test the different metrics provided by the library.

.. _json:

Recorded fields
---------------------

Recordings are logged in a json file and include the power draw and the use of the CPU and GPU for the pids related to your experiment. Some of the recordings are done for each pid related to your experiments: `per_process_metric_name : {... pid_i: v_i, ....}`. However, the monitoring of multiple programs on the same device should be done with care (see :ref:`multiple`). In the following, we details the different metrics recorded. Unless specified otherwise, the power is logged in Watt.

First you can load the data of an experiment contained in "your_output_folder"
.. code-block:: python
  
  from deep_learning_power_measure.power_measure import experiment, parsers
  driver = parsers.JsonParser('your_output_folder')
  exp_result = experiment.ExpResults(driver)

From then, you can compute some statistics 

.. code-block:: python
  
  # power consummed by the CPU measured by RAPL of your experiment
  print(exp_result.total_("rel_intel_power"))
  # duration of your experiments
  d, start, end = print(exp_result.get_duration())
  

To check the list of available metrics (might depend on your setup):

.. code-block:: python 

  print(exp_result)


.. code-block:: console

  Available metrics :
  CPU
    per_process_mem_use_abs,per_process_cpu_uses,per_process_mem_use_percent,intel_power,psys_power,uncore_power,per_process_cpu_power,total_cpu_power,per_process_mem_use_uss
  GPU
    nvidia_draw_absolute,nvidia_attributable_power,nvidia_mem_use,nvidia_sm_use,per_gpu_power_draw,per_gpu_attributable_power,per_gpu_estimated_attributable_utilization
  Experiments
    nvidia_draw_absolute,nvidia_attributable_power,nvidia_mem_use,nvidia_sm_use,per_gpu_power_draw,per_gpu_attributable_power,per_gpu_estimated_attributable_utilization

  
Below are the definitions of these metrics:

**CPU use**

- `per_process_mem_use_abs` : RAM PSS Memory usage for each recorded process in bytes*

- `per_process_mem_use_percent` : RAM PSS Memory usage for each recorded process in percentage of the overall memory usage*

- `per_process_mem_use_uss` : RAM  USS Memory usage for each recorded process*

- `per_process_cpu_uses` : Percentage of CPU usage for each process, relatively to the general CPU usage.

- `cpu_uses`: percentage of cpu clock used by this pid during the recording. 

- `mem_use_abs`: Number of bytes used in the CPU RAM. The recording uses psutil in the background, check: :py:func:`deep_learning_power_measure.power_measure.rapl_power.get_mem_uses` for more details.

- `mem_use_percent`: Relative number of bytes used in the CPU RAM PSS.

* For details on the USS and PSS memory, check :py:func:`deep_learning_power_measure.power_measure.rapl_power.get_mem_uses`

**Non GPU Energy consumption**

- `intel_power` : total consumptino measured by RAPL

- `total_cpu_power`: total consumption measured by RAPL for the CPU

- `psys_power`: System on chip consumption

- `uncore_power`: other hardware present on the cpu board, for instance, an integrated graphic card. This is NOT the nvidia gpu which is on another board.

- `total_cpu_power`: core power consumption.

- `per_process_cpu_power` : Essentially :  * intel_power. Should be used with caution (see :ref:`multiple`)

- `per_process_mem_use_uss` : USS memory per CPU in RAM.  

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

- `nvidia_draw_absolute`: the amount of power used by the whole nvidia card and all GPUs.

- `per_gpu_power_draw`: the amount of power used by the whole nvidia card for each GPUs

- `nvidia_attributable_power` : Total nvidia power consumption attributatble to the processes you recorded. It corresponds to  

- `per_gpu_attributable_power` : same as `nvidia_attributable_power` but for each gpu

Monitoring whole machine with Prometheus
----------------------------------------

The following code will launch the monitoring and a flask app on the port 5001

.. code-block:: python

  from deep_learning_power_measure.power_measure import experiment, prometheus_client

  driver = prometheus_client.PrometheusClient()
  exp = experiment.Experiment(driver)
  exp.monitor_machine(period=5)


Then, you can launch a prometheus instance

.. code-block:: console

   ./prometheus --config.file=prometheus.yml


with a config file which look like the following

.. code-block:: console

  global:
  scrape_interval: 3s

  external_labels:
    monitor: "example-app"

  rule_files:

  scrape_configs:
    - job_name: "flask_test"
      static_configs:
        - targets: ["localhost:5001"]

Then visit the following url : `http://localhost:9090/graph`

Currently, the following metrics are supported 

.. code-block:: console

   ['power_draw_cpu', 'intel_power', 
   'mem_used_cpu', 'mem_used_gpu', 
   'power_draw_gpu']


About Generative AI
-------------------

For ollama, or any external llm server :  the python script call an external call to a server which actually run the llm. Currently, AIPowerMeter does not track this call, and thus the cpu related metrics specifics to your experiment will be innacurate. So in this case, and actually with any program which rely on external tools on your machine, you follow this protocol:
 
- turn off every program you can
- trust the global/total metrics : total_intel_power, total_ram_consumption, total ram usage, and so on.

Anyway, isolating the consumption of a program should is always inacurrate (see :ref:`multiple`).

The GPU is already measured overall, assuming your experiment is the only one running on the GPU, so this part is fine.

When you can't (or don't want to) run the model on your own machine: check `EcoLogits <https://github.com/genai-impact>`_ for a tool which enables you to measure the impact of cloud instances (chatgpt, azur, and so on). They rely on approximations to estimate closed models, extrapolating from known models, using public leaderboard, or comparing prices.

.. _docker:

Docker integration
---------------------

For the implementation of AIPowerMeter in a docker container, we need to use a special branch of the code because of the behaviour of the command :

.. code-block:: console

  $ nvidia-smi pmon

An hot fix has been implemented, it forces the tracking of all the GPU processes. It's then impossible to isolate a process running at the same time than others.

See the github repo `docker_AIPM <https://github.com/GreenAI-Uppa/docker_AIPM>`_ for more details. You will also find slides explaining the motivations for the use of Docker images and container.
