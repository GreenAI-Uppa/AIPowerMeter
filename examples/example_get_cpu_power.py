from deep_learning_power_measure.power_measure import rapl, rapl_power
import time

print('The intel processor has one or more "packages". The package contains multiple PP0 PP1 power plane or domains. \n PP0 refers to the processor cores. \n PP1 domain refers to the power plane of a specific device in the uncore (eg the graphic card). \n DRAM corresponds to the power used by the memory.')
pause = 1
while True:
  sample = rapl.RAPLSample()
  s1 = sample.take_sample()
  time.sleep(pause)
  s2 = sample.take_sample()
  intel_power, cpu_power, dram_power, uncore_power, psys_power = rapl_power.get_power(s2 - s1)

  print('total intel_power : ',intel_power)
  print('cpu power (pp0) : ', cpu_power)
  print('uncore power, pp1 : ', uncore_power)
  print('dram power:',dram_power)
  print('system on chip power :',psys_power)
  print()
