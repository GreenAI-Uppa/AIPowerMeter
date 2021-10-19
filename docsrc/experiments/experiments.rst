Insights into energy consumption
====================================

Experimental protocol
---------------------

We've started to run experiments to measure the energy consumption of classical deep learning pretrained model at inference. Our protocol acts as follows:

- we load a pretrained architecture,

- we select an input size (resolution for Computer Vision, number of tokens for NLP),

- we run x inferences and measure power draws with AIPowerMeter,

- we repeat the experiment 10 times to have more robustness.

For each set of experiments, power measurements are written into severals power_metrics.json files (one by tuple (input_size,experiment). We then compile  `here <https://github.com/GreenAI-Uppa/AIPowerMeter/blob/main/power_metrics_management/concat_power_measure.py>`_ an estimate of different power draws of one inference and compile the median of the over the 10 runs. For each pretrained model, results are generated into a csv file where each row corresponds to one input size and each column represents the median of one power draw.  


Alexnet study
--------------
As a gentle start, we measure the consumption at inference of a vanilla `Alexnet <https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>`_ on a GeForce RTX 3090 GPU, and 16 i9 Intel cores CPU.

We first import necessary modules for power draws and torch models downloads.

.. code-block:: python

  import time, os
  import numpy as np
  from deep_learning_power_measure.power_measure import experiment, parsers
  import torchvision.models as models
  import torch

We then load Alexnet model and push it into our GeForce RTX 3090 GPU.

.. code-block:: python

  #load your favorite model
  alexnet = models.alexnet(pretrained=True)
  
  #choose your favorite device
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Using {} device'.format(device))
  
  #load the model to the device
  alexnet = alexnet.to(device)

We then prepare the experiment protocol described above for a random 224X244 image. We choose to run 4000 inferences to let AIPowerMeter reports the power draws during 40 seconds.

.. code-block:: python

  #experiments protocol
  iters = 4000#number of inferences
  xps = 10#number of experiments to reach robustness
  
  #choose a resolution size
  input_size = 224
  
  #create a random image
  image_test = torch.rand(1,3,input_size,input_size)

We then start the inferences and measurements.

.. code-block:: python

  #start of the experiments
  for k in range(xps):
  	print('Experience',k,'/',xps,'is running')
  	latencies = []
  	#AIPM
  	input_image_size = (1,3,input_size,input_size)
  	driver = parsers.JsonParser(os.path.join(os.getcwd(),"input_"+str(input_size)+"/run_"+str(k)))
	exp = experiment.Experiment(driver,model=alexnet,input_size=input_image_size)
	p, q = exp.measure_yourself(period=2)
	start_xp = time.time()
	for t in range(iters):
		start_iter = time.time()
		y = alexnet(image_test)
		res = time.time()-start_iter
		#print(t,'latency',res)
		latencies.append(res)
	q.put(experiment.STOP_MESSAGE)
	end_xp = time.time()
	print("power measuring stopped after",end_xp-start_xp,"seconds for experience",k,"/",xps)
	driver = parsers.JsonParser("input_"+str(input_size)+"/run_"+str(k))
	#write latency.csv next to power_metrics.json file
	np.savetxt("input_"+str(input_size)+"/run_"+str(k)+"/latency.csv",np.array(latencies))



Resnet study
------------




Bert Transformers
-----------------
