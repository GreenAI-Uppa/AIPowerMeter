import time, os
import numpy as np
from deep_learning_power_measure.power_measure import experiment, parsers

import nets
import torch

#Create your favorites CNN
channels = 3
nb_filters = 64
nb_conv_layers_list = [2*k for k in range(1,21)]

#choose your favorite device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

#generate a random image
image_size = 224
image_test = torch.rand(1,channels,image_size,image_size)
image_test = image_test.to(device)


for u,input_size in enumerate(nb_conv_layers_list):
    #load the model to the device
    print('Start Number of layers',input_size)
    nb_conv_layers = input_size
    model = nets.ConvNet(channels,nb_filters,nb_conv_layers)
    model = model.to(device)
    iters = int(40000/input_size)#number of inferences
    print(iters,'inferences')
    xps = 10#number of experiments to reach robustness
    for k in range(xps):
        print('Experience',k+1,'/',xps,'is running')
        latencies = []
        #AIPM
        driver = parsers.JsonParser(os.path.join(os.getcwd(),"/data/sloustau/measure/convnets/input_"+str(input_size)+"/run_"+str(k)))
        exp = experiment.Experiment(driver)
        p, q = exp.measure_yourself(period=2)
        start_xp = time.time()
        for t in range(iters):
            #print(t)
            start_iter = time.time()
            y = model(image_test)
            res = time.time()-start_iter
            #print(t,'latency',res)
            latencies.append(res)
        q.put(experiment.STOP_MESSAGE)
        end_xp = time.time()
        print("power measuring stopped after",end_xp-start_xp,"seconds for experience",k+1,"/",xps)
        driver = parsers.JsonParser("/data/sloustau/measure/convnets/input_"+str(input_size)+"/run_"+str(k))
        #write latency.csv next to power_metrics.json file
        np.savetxt("/data/sloustau/measure/convnets/input_"+str(input_size)+"/run_"+str(k)+"/latency.csv",np.array(latencies))
