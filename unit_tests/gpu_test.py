from deep_learning_power_measure.power_measure import gpu_power
import time

gpu1 = gpu_power.get_gpu_sample()

time.sleep(2)

gpu2 = gpu_power.get_gpu_sample()

diff = gpu_power.get_diff_gpu_sample(gpu1,gpu2)
print('start',gpu1)
print('end',gpu2)
print('diff',diff)
