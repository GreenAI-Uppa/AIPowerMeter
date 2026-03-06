import pynvml



def get_gpu_sample():
    pynvml.nvmlInit()
    per_gpu = {}
    deviceCount = pynvml.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle =pynvml.nvmlDeviceGetHandleByIndex(i)
        tot_power = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000
        tot_use_rate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        tot_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        per_gpu[i] = {"tot_power":tot_power,
                      "tot_use_rate":tot_use_rate,
                      "tot_memory":tot_memory}
    return per_gpu
