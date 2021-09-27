# sudo chmod -R 555 /sys/class/powercap/intel-rapl/
import os, time
import psutil
import warnings
from multiprocessing import Process, Queue

from . import rapl
rapl_dir = "/sys/class/powercap/intel-rapl/"

def is_rapl_compatible():
    if not os.path.isdir(rapl_dir):
        return (False, "cannot find directory "+rapl_dir + " maybe modify the value in rapl_power.rapl_dir")
    if not (os.path.isfile('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj') and os.access('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', os.R_OK)):
        return (False, "the energy_uj files in "+rapl_dir+" are not readeable. Can you change the permissions of these files : \n sudo chmod -R 755 /sys/class/powercap/intel-rapl/ ")
    return (True, "rapl ok")

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
    for i, p in enumerate(process_list):
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
    """
    diff : difference between two RAPL samples
    """
    total_intel_power = 0
    total_dram_power = 0
    total_cpu_power = 0
    total_uncore_power = 0
    psys_power = -1
    total = 0
    for d in diff.domains:
        domain = diff.domains[d]
        power = diff.average_power(package=domain.name)
        # this should get the power per package (e.g., total rapl power)
        # see images/power-planes.png for example
        # Downloaded from: https://blog.chih.me/images/power-planes.jpg
        #  Recent (Sandy Bridge and later) Intel processors that implement the RAPL (Running Average Power Limit)
        # interface that provides MSRs containing energy consumption estimates for up to four power planes or
        # domains of a machine, as seen in the diagram above.
        # PKG: The entire package.
        # PP0: The cores.
        # PP1: An uncore device, usually the GPU (not available on all processor models.)
        # DRAM: main memory (not available on all processor models.)
        # PSys: Skylake mobile SoC total energy
        # The following relationship holds: PP0 + PP1 <= PKG. DRAM is independent of the other three domains.
        # Most processors come in two packages so top level domains shold be package-1 and package-0
        total += power
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
            #pint(subdomain, power)
            if subdomain == "ram" or subdomain == "dram":
                total_dram_power += power
            elif subdomain == "core" or subdomain == "cpu":
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
    return total_intel_power, total_dram_power, total_cpu_power, total_uncore_power, psys_power

def get_percent_uses(infos1, infos2, zombies, process_list):
    """
    infos1 and infos2 : cpu times gathered at two different times and both system and process wised.
    """
    cpu_percent = {}
    for i, p in enumerate(process_list):
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
            print("cpu_util_system is 0", p.pid, delta_proc2, cpu_util_system, cpu_util_process)
            attributable_compute = 0
        else:
            attributable_compute = cpu_util_process / cpu_util_system

        cpu_percent[p.pid] = attributable_compute
    return cpu_percent # should be for multiple softwares

def get_cpu_uses(process_list, pause=2.0):
    """
    input:
        process_list : list of process for which the cpu use will be measured
        pause : sleeping time during which the cpu use will be recorded.
    return a dictionnary
        cpu_uses = {soft1 : cpu_use, }
    where cpu_use is the percentage of use of this cpu with the respect to the total use of the cpu on this period (mouthfull!!)
    """

    # get the cpu time used
    infos1, zombies = get_info_time(process_list)
    # wait a bit
    time.sleep(pause)
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
    """
    input :
        process_list : list of psutil.Process objects
    output:
        mem_info_per_process : memory consumption for each process

        some info from psutil documentation:
            uss (Linux, macOS, Windows): aka “Unique Set Size”, this is the memory which is unique to a process and which would be freed if the process was terminated right now
            pss (Linux): aka “Proportional Set Size”, is the amount of memory shared with other processes,
            accounted in a way that the amount is divided evenly between the processes that share it.
            I.e. if a process has 10 MBs all to itself and 10 MBs shared with another process its PSS will be 15 MBs.

            # on the other hand RSS is resident set size : the non-swapped physical memory that a task has used in bytes.
            # so with the previous example, the result would be 20Mbs instead of 15Mbs
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

def get_metrics(pid_list, pause = 2.0):
    """
    main function which will return power uses given a list of process ids
    pause : indicates how many seconds to wait to compute the delta of power draw and cpu uses
    """
    sample = rapl.RAPLSample()
    s1 = sample.take_sample()
    process_list = get_processes(pid_list)
    cpu_uses = get_cpu_uses(process_list, pause = pause)
    mem_pss_per_process, mem_uss_per_process = get_mem_uses(process_list)
    mem_uses = get_relative_mem_use(mem_pss_per_process)
    s2 = sample.take_sample()
    intel_power, dram_power, cpu_power, uncore_power, psys_power = get_power(s2 - s1)
    cpu_power_use = get_rel_power(cpu_uses, cpu_power)
    dram_power_use = get_rel_power(mem_uses, dram_power)
    metrics = {
        'mem_use_abs':mem_pss_per_process,
        'cpu_uses': cpu_uses, 'mem_use_percent': mem_uses,
        'intel_power' :intel_power,
        'total_cpu_power':cpu_power,
        'total_dram_power':dram_power,
        'uncore_power':uncore_power,
        'per_process_cpu_power':cpu_power_use,
        'per_process_dram_power':dram_power_use,
        'psys_power':psys_power
    }
    if len(mem_uss_per_process) > 0:
        metrics['mem_use_uss'] = mem_uss_per_process
    return metrics
