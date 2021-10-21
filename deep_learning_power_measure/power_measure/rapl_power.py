"""Handling of the CPU use and CPU consumption with RAPL"""
import os
import time
import warnings
import psutil

from . import rapl
rapl_dir = "/sys/class/powercap/intel-rapl/"

def is_rapl_compatible():
    """
    Check if rapl logs are available on this machine.
    """
    if not os.path.isdir(rapl_dir):
        return (False, "cannot find rapl directory in "+rapl_dir)
    if not (os.path.isfile('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj') and os.access('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', os.R_OK)):
        return (False, "the energy_uj files in "+rapl_dir+" are not readeable. Change the permissions of these files : \n sudo chmod -R 755 /sys/class/powercap/intel-rapl/")

    sample = rapl.RAPLSample()
    domain_names = set()
    s1 = sample.take_sample()
    for d in s1.domains:
        domain = s1.domains[d]
        domain_names.add(domain.name)
        for sd in domain.subdomains:
            subdomain = domain.subdomains[sd]
            domain_names.add(subdomain.name)
    msg = "RAPL available:\n"
    if 'dram' not in domain_names and 'ram' not in domain_names:
        msg += 'RAM related energy consumption NOT available\n'
    else:
        msg += 'RAM related energy consumption available\n'
    if 'core' not in domain_names and 'cpu' not in domain_names:
        msg += 'CPU core related energy consumption NOT available\n'
    else:
        msg += 'CPU core related energy consumption available\n'
    if 'uncore' not in domain_names:
        msg += 'uncore related energy consumption NOT available\n'
    else:
        msg += 'uncore related energy consumption available\n'
    if 'psys' not in domain_names:
        msg += 'System on Chip related energy consumption NOT available\n'
    else:
        msg += 'System on Chip related energy consumption available\n'
    return (True, msg)

_timer = getattr(time, "monotonic", time.time)

def get_info_time(process_list, zombies=None):
    """
    input
    process_list : a list of process objects
    zombies : a list of zombies processes that will be enlarged

    output:
    for each process in process_list: (st11, st12, system_wide_pt1, pt1)
    st11 : starting time
    st12 :
    system_wide_pt1 : system wide process time
    pt1 : process time for this process
    """
    infos = {}
    if zombies is None:
        zombies = []
    for p in process_list:
        st11 = _timer()
        # units in terms of cpu-time, so we need the cpu in the last time period that are for the process only
        system_wide_pt1 = psutil.cpu_times()
        st12 = _timer()
        try:
            pt1 = p.cpu_times()
            infos[p.pid] = (st11, st12, system_wide_pt1, pt1)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            zombies.append(p.pid)
    return infos, zombies

def get_processes(pid_list):
    """Obtain the process object given the list of pid integers"""
    process_list = []
    # gather processes as process objects
    for process in pid_list:
        try:
            p = psutil.Process(process)
            process_list.append(p)
        except psutil.NoSuchProcess:
            warnings.warn(
                    "Process with pid {} used to be part of this process chain, but was shut down. Skipping."
                )
    return process_list

def get_power(diff):
    """return the power accumulation of the provided pair of rapl samples for
    the different RAPL domains

    Args:
        diff : difference between two RAPL samples

    Returns:
        Dictionnary where each key correspond to an RAPL domain and the value
        is the accumulated energy consumption in Joules
    """
    total_intel_power = 0
    total_dram_power = 0
    total_cpu_power = 0
    total_uncore_power = 0
    psys_power = -1
    domains_found = set()
    for d in diff.domains:
        domain = diff.domains[d]
        power = diff.average_power(package=domain.name)
        if domain.name == "psys":  # skip SoC aggregate reporting
            psys_power = power
            continue
        if "package" not in domain.name:
            raise NotImplementedError(
                "Unexpected top level domain for RAPL package. Not yet supported."
            )

        total_intel_power += power
        for sd in domain.subdomains:
            subdomain = domain.subdomains[sd]
            power = diff.average_power(package=domain.name, domain=subdomain.name)
            subdomain = subdomain.name.lower()
            domains_found.add(subdomain)
            #print(subdomain, power)
            if subdomain in ("ram", "dram"):
                total_dram_power += power
            elif subdomain in ("core", "cpu"):
                total_cpu_power += power
            elif subdomain == "uncore":
                total_uncore_power += power
            # other domains get don't have relevant readouts to give power attribution, therefore
            # will get assigned the same amount of credit as the CPU
    ## this block should be put much higher the stack, in another function
    if total_intel_power == 0:
        raise ValueError(
            "It seems that power estimates from Intel RAPL are coming back 0, this indicates a problem."
        )
    power_metrics = {
            'intel_power': total_intel_power,
            'psys_power':psys_power
            }
    if 'ram' in domains_found or 'dram' in domains_found:
        power_metrics['dram_power'] = total_dram_power
    if 'core' in domains_found or 'cpu' in domains_found:
        power_metrics['cpu_power'] = total_cpu_power
    if 'uncore' in domains_found:
        power_metrics['uncore_power'] = total_uncore_power
    return power_metrics

def get_percent_uses(infos1, infos2, zombies, process_list):
    """
    infos1 and infos2 : cpu times gathered at two different times and both system and process wised.
    """
    cpu_percent = {}
    for p in process_list:
        if p.pid in zombies:
            continue
        st1, st12, system_wide_pt1, pt1 = infos1[p.pid]
        st2, st22, system_wide_pt2, pt2 = infos2[p.pid]

        #  time used by this process
        delta_proc = (pt2.user - pt1.user) + (pt2.system - pt1.system)
        cpu_util_process = delta_proc / float(st2 - st1)
        # time used system wide
        delta_proc2 = (system_wide_pt2.user - system_wide_pt1.user) + (system_wide_pt2.system - system_wide_pt1.system)
        cpu_util_system = delta_proc2 / float(st22 - st12)
        # percent of cpu-hours in time frame attributable to this process (e.g., attributable compute)
        if cpu_util_system == 0:
            print("WARNING cpu_util_system is 0", p.pid, delta_proc2, cpu_util_system, cpu_util_process)
            print("I cannot attribute cpu time to the different pids.")
            print("consider to set a larger period value when calling measurement function")
            attributable_compute = 0
        else:
            attributable_compute = cpu_util_process / cpu_util_system

        cpu_percent[p.pid] = attributable_compute
    return cpu_percent # should be for multiple softwares

def get_cpu_uses(process_list, period=2.0):
    """Extracts the relative number of cpu clock attributed to each process

    Args:
        process_list : list of process [pid1, pid2,...] for which the cpu use
        will be measured
        pause : sleeping time during which the cpu use will be recorded.

    Returns:
        cpu_uses = {pid1 : cpu_use, }  where cpu_use is the percentage of use
        of this cpu with the respect to the total use of the cpu on this period
    """

    # get the cpu time used
    infos1, zombies = get_info_time(process_list)
    # wait a bit
    time.sleep(period)
    # get the cpu time used
    infos2, zombies = get_info_time(process_list, zombies)
    cpu_uses = get_percent_uses(infos1, infos2, zombies, process_list)
    return cpu_uses

def get_rel_power(rel_uses, power):
    """
    input:
        dictionnary : pid : relative use in percentages
        power : total power used in watts
    return
        dictionnary pid : power use for this process

    We use this function to multiply the entire socket consumption and gpu board
    values by the relative usage of each process to obtain the per process energy
    consumption.
    """
    power_per_process = {}
    for pid, rel_use in rel_uses.items():
        power_per_process[pid] = rel_use * power
    return power_per_process

def get_relative_mem_use(mem_info_per_process):
    """
    Get the percentage of system memory which is used, with respect to the total of the memory available
    """
    total_physical_memory = psutil.virtual_memory()

    mem_percent = {}
    for pid, mem in mem_info_per_process.items():
        mem_percent[pid] = mem / float(total_physical_memory.total - total_physical_memory.available)
    return mem_percent

def get_mem_uses(process_list):
    """Get memory usage.
    psutil will be used to collect pss and uss values. rss is collected if pss
    is not available
    some info from psutil documentation:

    USS : (Linux, macOS, Windows): aka “Unique Set Size”, this is the memory
    was terminated right now which is unique to a process and which would be
    freed if the process

    PSS :  (Linux): aka “Proportional Set Size”, is the amount of memory
    shared with other processes, accounted in a way that the amount is
    divided evenly between the processes that share it. I.e. if a process
    has 10 MBs all to itself and 10 MBs shared with another process its
    PSS will be 15 MBs.

    RSS : On the other hand RSS is resident set size : the
    non-swapped physical memory that a task has used in bytes. so with the
    previous example, the result would be 20Mbs instead of 15Mbs

    Args :
        process_list : list of psutil.Process objects
    Returns:
        mem_info_per_process : memory consumption for each process
    """
    mem_info_per_process, mem_pss_per_process, mem_uss_per_process = {}, {}, {}
    for p in process_list:
        try:
            try:
                mem_info = p.memory_full_info()
            except psutil.AccessDenied:
                mem_info = p.memory_info()
            mem_info_per_process[p.pid]= mem_info._asdict()
        except (psutil.ZombieProcess, psutil.NoSuchProcess):
            pass
    for k, info in mem_info_per_process.items():
        if "pss" in info:
            mem_pss_per_process[k] = info["pss"]
        else:
            # Sometimes we don't have access to PSS so just need to make due with rss
            mem_pss_per_process[k] = info["rss"]
        if 'uss' in info:
            mem_uss_per_process[k] = info['uss']
    return mem_pss_per_process, mem_uss_per_process

def get_metrics(pid_list, period = 2.0):
    """
    main function which will return power uses given a list of process ids
    pause : indicates how many seconds to wait to compute the delta of power draw and cpu uses
    """
    sample = rapl.RAPLSample()
    s1 = sample.take_sample()
    process_list = get_processes(pid_list)
    cpu_uses = get_cpu_uses(process_list, period = period)
    mem_pss_per_process, mem_uss_per_process = get_mem_uses(process_list)
    mem_uses = get_relative_mem_use(mem_pss_per_process)
    s2 = sample.take_sample()
    power_metrics = get_power(s2 - s1)
    metrics = {
        'per_process_mem_use_abs':mem_pss_per_process,
        'per_process_cpu_uses': cpu_uses,
        'per_process_mem_use_percent': mem_uses,
        'intel_power' :power_metrics['intel_power'],
        'psys_power':power_metrics['psys_power']
    }
    if 'uncore_power' in power_metrics:
        metrics['uncore_power'] = power_metrics['uncore_power'],
    if 'cpu_power' in power_metrics:
        cpu_power = power_metrics['cpu_power']
        cpu_power_use = get_rel_power(cpu_uses, cpu_power)
        metrics['per_process_cpu_power'] = cpu_power_use
        metrics['total_cpu_power'] = cpu_power
    if 'dram_power' in power_metrics:
        dram_power = power_metrics['dram_power']
        dram_power_use = get_rel_power(mem_uses, dram_power)
        metrics['per_process_dram_power'] = dram_power_use
        metrics['total_dram_power'] = dram_power
    if len(mem_uss_per_process) > 0:
        metrics['per_process_mem_use_uss'] = mem_uss_per_process
    return metrics
