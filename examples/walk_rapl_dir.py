from deep_learning_power_measure.power_measure import rapl

try:
  for dirpath, dirnames, filenames in rapl._walk_rapl_dir(rapl.rapl_dir):
      if len(dirpath.split(":")) == 1:
          continue # base of rapl tree, no name associated
      print(dirpath)
      name, energy_uj, max_energy_range_uj = rapl._get_domain_info(dirpath)
      print(name)
except PermissionError as e:
    print("rapl files are not readeable. Change the permissions  : \n sudo chmod -R 755 /sys/class/powercap/intel-rapl/")
