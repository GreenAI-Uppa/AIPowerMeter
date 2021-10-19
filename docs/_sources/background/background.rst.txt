Background on power measure
===========================

General considerations
----------------------
Main sources of energy consumption are the gpu, the cpu and memory.


However, some consumption sources will be missed by your setup, such as optical drives, motherboards, and hard drives


.. _rapl:

CPU and RAPL
-----------------------------------------------------

The Running Average Power Limit (RAPL) reports the accumulated energy consumption of the cpu, the ram mechanism, and a few other devices (but NOT the cpu).  


The official documentation is the Intel® 64 and IA-32 Architectures Software Developer Manual, Volume 3: System Programming Guide. But it is not trivial for most data scientists.

Frequency of 1000Hz

Add ref to the finnish guys [Khan2018]_



GPU and nvidia-smi 
---------------------------
description of nvidia-smi
Things are more simple, and unfortunately because we have much less information.




.. [Khan2018] Khan et al. RAPL in Action: Experiences in Using RAPL for Power Measurements. ACM Transactions on Modeling and Performance Evaluation of Computing Systems. 2018
