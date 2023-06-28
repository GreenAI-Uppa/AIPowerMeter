from deep_learning_power_measure.power_measure import rapl
import time

try:
  # reading rapl files a first time
  first = {}
  for dirpath, dirnames, filenames in rapl._walk_rapl_dir(rapl.rapl_dir):
      if len(dirpath.split(":")) == 1:
          continue # base of rapl tree, no name associated
      name, energy_uj, max_energy_range_uj = rapl._get_domain_info(dirpath)
      first[dirpath] = (name, energy_uj)

  time.sleep(1)
  # reading rapl a file a second time
  for dirpath, dirnames, filenames in rapl._walk_rapl_dir(rapl.rapl_dir):
      if len(dirpath.split(":")) == 1:
          continue # base of rapl tree, no name associated
      name, energy_uj, max_energy_range_uj = rapl._get_domain_info(dirpath)
      _, energy_uj1 = first[dirpath]
      print(dirpath)
      # and take the substraction between the two moments
      print(name, energy_uj - energy_uj1)

except PermissionError as e:
    print("rapl files are not readeable. Change the permissions  : \n sudo chmod -R 755 /sys/class/powercap/intel-rapl/")
