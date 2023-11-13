from deep_learning_power_measure.power_measure import rapl_power, gpu_power, rapl 

rapl_available, msg = rapl_power.is_rapl_compatible()
print(msg)

nvidia_available, msg = gpu_power.is_nvidia_compatible()
if nvidia_available:
    print(msg)
else:
    print(msg)
