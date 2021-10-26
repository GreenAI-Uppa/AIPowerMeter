import time, os
import numpy as np
from deep_learning_power_measure.power_measure import experiment, parsers
import torchvision.models as models
import torch

#load your favorite model
alexnet = models.alexnet(pretrained=True)


#choose your favorite device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

#load the model to the device
alexnet = alexnet.to(device)

#resolution size list
input_sizes = [112*(k+1) for k in range(20)]


for u,input_size in enumerate(input_sizes):
    #experiments protocol
    iters = int(40000/(u+1))#number of inferences
    #print(iters)
    xps = 10#number of experiments to reach robustness
    #create a random image
    image_test = torch.rand(1,3,input_size,input_size)
    image_test = image_test.to(device)
    #start of the experiments
    for k in range(xps):
        print('Experience',k+1,'/',xps,'is running')
        latencies = []
        #AIPM
        input_image_size = (1,3,input_size,input_size)
        driver = parsers.JsonParser(os.path.join(os.getcwd(),"input_"+str(input_size)+"/run_"+str(k)))
        exp = experiment.Experiment(driver)
        p, q = exp.measure_yourself(period=2)
        start_xp = time.time()
        for t in range(iters):
            #print(t)
            start_iter = time.time()
            y = alexnet(image_test)
            res = time.time()-start_iter
            #print(t,'latency',res)
            latencies.append(res)
        q.put(experiment.STOP_MESSAGE)
        end_xp = time.time()
        print("power measuring stopped after",end_xp-start_xp,"seconds for experience",k+1,"/",xps)
        driver = parsers.JsonParser("input_"+str(input_size)+"/run_"+str(k))
        #write latency.csv next to power_metrics.json file
        np.savetxt("input_"+str(input_size)+"/run_"+str(k)+"/latency.csv",np.array(latencies))

