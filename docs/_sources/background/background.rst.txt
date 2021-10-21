Background on power measure
===========================

General considerations
----------------------
Main sources of energy consumption are the gpu, the cpu and memory.


However, some consumption sources will be missed by your setup, such as optical drives, motherboards, and hard drives.

Preliminaries
-------------

The unit to measure energy is the Joule, it is the energy transferred to an object when a force of one newton acts on that object in the direction of the force’s motion through a distance of one metre (1 newton-metre or Nm). 

The watt is the unit to measure power. 

1 watt = 1 Joule per second = The energy required to lift a medium-sized tomato up 1 metre (wikipedia)

1kWh = 3600000 Joules ~= 3 hours of GPU


**And for a computer:**

A computer consumes energy from different parts. This library allows you to measure the one highlighted in red.

Thousands of cores to enable parallelism
Lower amount of RAM memory available
Higher latency : GPU clock speed < CPU clock speed
Higher memory throughput : GPU operates on larger chunks of data
GPU can fetch data from its RAM more quickly
CPU bandwidth < GPU bandwidth
Smaller set of instructions dedicated to graphics and matrix calculus
More power hungry and requires a CPU
Energy efficient since the computations is faster.

.. _rapl:

CPU and RAPL
-----------------------------------------------------

The Running Average Power Limit (RAPL) reports the accumulated energy consumption of the cpu, the ram mechanism, and a few other devices (but NOT the cpu). 
It is present since and from Haswell is supported by integrated voltage regulators in addition to power models ( [Hackenberg2015]_ ), and there has been considerable study to validate its use for software monitoring ( [Khan2018]_ ).

It is divided into different physically meaningfull domains:

.. figure:: rapl_domain.png
   
   RAPL power domains (from [Khan2018]_ )

- Power Plane 0 : the cores alias the CPU
- Power Plane 1 : uncore : memory controller and cache, and an integrated on chip gpu is present (this is not the external nvidia GPU). 
- DRAM : energy consumption of the RAM
- Psys : System on Chip energy consumption


The recording are done for the entire cpu sockets. Thus, to take into account the energy consumed from each program, we adopt the approach implemtented in the `experiment impact tracker <https://github.com/Breakend/experiment-impact-tracker>`_ and multiply the RAPL value by the percentage of cpu and memory used.


The rapl interface writes these values in module specific registers located in `/dev/cpu/*/msr`. These values are updated every 1ms. Although reading from these files is possible, our code relies on the powercap linux tool which updates the energy consumption for the different domains in `/sys/class/powercap/intel-rapl`.


**More readings on RAPL**:

The official documentation is the Intel® 64 and IA-32 Architectures Software Developer Manual, Volume 3: System Programming Guide. But it is not trivial for most data scientists.


GPU and nvidia-smi 
---------------------------
description of nvidia-smi
Things are more simple, and unfortunately because we have much less information.
from the man page of `nvidia-smi <https://man.archlinux.org/man/nvidia-utils/nvidia-smi.1.en>`_ : *The last measured power draw for the entire board, in watts. Only available if power management is supported. Please note that for boards without INA sensors, this refers to the power draw for the GPU and not for the entire board.*

Related work
------------

There are several tools developed to monitor energy consumption of softwares, all based on RAPL and nvidia-smi. `Performance Application Programming Interface <https://icl.utk.edu/papi/>`_ has a long history and is a very complete library to measure numerous aspects of program run. In the specific field of AI and deep learning, serveral repos such as `CarbonTracker <https://github.com/lfwa/carbontracker/>`_ and `Experiment Impact Tracker <https://github.com/Breakend/experiment-impact-tracker>`_ propose to compute a carbon footprint of your experiment. The development of our own library has started as a fork of this latter repository. It's aim is to focus on fine grained energy consumption of deep learning models. Stay tuned with the `Coca4AI <https://greenai-uppa.github.io/Coca4AI/>`_ for an measurement campaign at the scale of a data center. 

https://www.tensorflow.org/guide/profiler

Bibliography
------------
.. [Hackenberg2015] An Energy Efficiency Feature Survey of the Intel Haswell Processor.  IEEE International Parallel and Distributed Processing Symposium Workshop. 2015
.. [Khan2018] Khan et al. RAPL in Action: Experiences in Using RAPL for Power Measurements. ACM Transactions on Modeling and Performance Evaluation of Computing Systems. 2018
