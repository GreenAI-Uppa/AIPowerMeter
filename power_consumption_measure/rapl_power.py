# sudo chmod -R 555 /sys/class/powercap/intel-rapl/
import os, time, numpy as np
import json
import psutil
import warnings
import networkx as nx
import sys
from functools import wraps
import traceback
from multiprocessing import Process, Queue

from . import rapl
rapl_dir = "/sys/class/powercap/intel-rapl/"

def is_rapl_compatible():
    if not os.path.isdir(rapl_dir):
        return (False, "cannot find directory "+rapl_dir + " maybe modify the value in rapl_power.rapl_dir")
    if not (os.path.isfile('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj') and os.access('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', os.R_OK)):
        return (False, "the energy_uj files in "+rapl_dir+" are not readeable. Can you change the permissions of these files : \n sudo chmod -R 755 /sys/class/powercap/intel-rapl/ ")
    return (True, "rapl ok")

def get_pids():
    """
    return a tree where each node corresponds to a running process 
    the parent-child in the tree follows the parent child process relations, ie a child process had been launched by its father
    """
    process_tree = nx.DiGraph()
    for proc in psutil.process_iter():
        try:
            process_tree.add_node(proc.pid)
            process_tree.nodes[proc.pid]['name'] = proc.name()
            process_tree.nodes[proc.pid]['user'] = proc.username()
            for child in proc.children(recursive=False):
                process_tree.add_edge(proc.pid, child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return process_tree

_timer = getattr(time, "monotonic", time.time)

def get_process_tree():
    """
    process_tree : return a networkx tree of the processes currently running on this machine. A child process corresponds to a process launched by its father. 

    check the following code to visualise the tree with matplotlib :

    labels = nx.get_node_attributes(process_tree, 'user')
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(process_tree, prog="twopi", args="")
    plt.figure(figsize=(8, 8))
    nx.draw(process_tree, pos, labels=labels, node_size=20, alpha=0.5, node_color="blue", with_labels=True)
    plt.axis("equal")
    plt.show()

    """
    process_tree = nx.DiGraph()
    for proc in psutil.process_iter():
        try:
            process_tree.add_node(proc.pid)
            process_tree.nodes[proc.pid]['name'] = proc.name()
            process_tree.nodes[proc.pid]['user'] = proc.username()
            for child in proc.children(recursive=False):
                process_tree.add_edge(proc.pid, child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return process_tree


def get_info_time(process_list, zombies=None):
    """
    input
    process_list : a list of process objects
    zombies : a list of zombies processes that will be enlarged

    output:
    for each process in process_list: system_wide and process cpu time, with the corresponding time when the measure is taken
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
        #pint(domain.name, power)
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

        # change in cpu-hours process
        delta_proc = (pt2.user - pt1.user) + (pt2.system - pt1.system)
        cpu_util_process = delta_proc / float(st2 - st1)
        # change in cpu-hours system
        delta_proc2 = (system_wide_pt2.user - system_wide_pt1.user) + (
            system_wide_pt2.system - system_wide_pt1.system
        )
        cpu_util_system = delta_proc2 / float(st22 - st12)
        #print('cpu time', p.pid, 'cpu-hours process', delta_proc, 'cpu-hours system', delta_proc2, cpu_util_process, cpu_util_system)

        # percent of cpu-hours in time frame attributable to this process (e.g., attributable compute)
        attributable_compute = cpu_util_process / cpu_util_system

        cpu_percent[p.pid] = attributable_compute
    return cpu_percent # should be for multiple softwares 

def get_cpu_uses(process_list, pause=2.0):
    """
    input: 
      process_list : list of 
    return a dictionnary
    cpu_uses = {soft1 : cpu_use, }
    where cpu_use is the percentage of use of this cpu with the respect to the total use of the cpu on this period (mouthfull!!)
    """
    cpu_percent = 0
    infos2 = []
    cpu_times_per_process = {}
    infos1, zombies = get_info_time(process_list)
    time.sleep(pause)
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

def get_relative_mem_use(process_list):
    """
    Get the percentage of system memory which is used, with respect to the total of the memory available
    """
    mem_info_per_process = get_mem_uses(process_list)
    total_physical_memory = psutil.virtual_memory()
    # what percentage of used memory can be attributed to this process
    # pss (Linux): aka “Proportional Set Size”, is the amount of memory shared with other processes, accounted in a way
    # that the amount is divided evenly between the processes that share it. I.e. if a process has 10 MBs all to itself
    # and 10 MBs shared with another process its PSS will be 15 MBs.
    # summing these two gets us a nice fair metric for the actual memory used in the RAM hardware.
    # The unique bits are directly attributable to the process
    # and the shared bits we give credit based on how many processes share those bits
    mem_percent = {}
    for pid, mem_info in mem_info_per_process.items():
        #pss_avail = all(["pss" in x for x in mem_info_per_process.values()])
        if "pss" in mem_info:
            mem_percent[pid] = mem_info["pss"] / float(total_physical_memory.total - total_physical_memory.available)
        #if pss_avail:
            """
            mem_percent = np.sum(
                [
                    float(x["pss"])
                    / float(total_physical_memory.total - total_physical_memory.available)
                    for x in mem_info_per_process.values()
                ]
            )
            """
        else:
            # Sometimes we don't have access to PSS so just need to make due with rss
            mem_percent[pid] = mem_info["rss"] / float(total_physical_memory.total - total_physical_memory.available)
            """
            mem_percent = np.sum(
                [
                    float(x["rss"])
                    / float(total_physical_memory.total - total_physical_memory.available)
                    for x in mem_info_per_process.values()
                ]
            )
            """
    return mem_percent

def get_mem_uses(process_list):
    """
    input :
        process_list : list of psutil.Process objects
    output:
        mem_info_per_process : memory consumption for each process
    """
    mem_info_per_process = {}
    for p in process_list:
        try:
             try:
                 mem_info = p.memory_full_info()
             except psutil.AccessDenied:
                 mem_info = p.memory_info()
             mem_info_per_process[p.pid]= mem_info._asdict()
        except (psutil.ZombieProcess, psutil.NoSuchProcess):
            pass
    return mem_info_per_process

def get_metrics(pid_list, pause = 2.0):
    """
    main function which will return power uses given a list of process ids
    pause : indicates how many seconds to wait to compute the delta of power draw and cpu uses
    """
    sample = rapl.RAPLSample()
    s1 = sample.take_sample()
    process_list = get_processes(pid_list)
    cpu_uses = get_cpu_uses(process_list, pause = pause)
    mem_uses = get_relative_mem_use(process_list)
    s2 = sample.take_sample()
    intel_power, cpu_power, dram_power, uncore_power, psys_power = get_power(s2 - s1)
    cpu_power_use = get_rel_power(cpu_uses, cpu_power)
    dram_power_use = get_rel_power(mem_uses, dram_power)
    metrics = {'cpu_uses': cpu_uses, 'mem_uses': mem_uses, 'intel_power' :intel_power, 'total_cpu_power':cpu_power, 'total_dram_power':dram_power, 'uncore_power':uncore_power, 'per_process_cpu_power':cpu_power_use, 'per_process_dram_power':dram_power_use, 'psys_power':psys_power}
    return metrics

def write(data, outfile):
    """
    append the json formated data as the last line in the file outfile
    """
    string = json.dumps(data)



if __name__ ==  '__main__':
    """
    take the power consumption ponderated by the cpu usage average over a duration of two seconds.

        - Take the energy consumption ie Get a sample of RAPL
        - get the cpu usage for each processors
        - sleep for 2 seconds
        - get the cpu usage for each processors
        - Take the energy consumption ie Get a sample of RAPL
        - Take the difference of
    """
    sample = rapl.RAPLSample()
    s1 = sample.take_sample()
    pid_list = get_pids()
    process_list = get_processes(pid_list)
    cpu_uses = get_cpu_uses(process_list)
    mem_uses = get_relative_mem_use(process_list)
    s2 = sample.take_sample()
    intel_power, cpu_power, dram_power, uncore_power = get_system_wide_power_use(s2 - s1)
    cpu_power_use = get_rel_power(cpu_uses, cpu_power)
    dram_power_use = get_rel_power(mem_uses, dram_power)
    # assign the rest of the power, ie the uncore, the system agent, and others...
    print('used power by the system')
    print('total: ',intel_power)
    print('cpu:', cpu_power)
    print('dram:', dram_power)
    print('uncore (graphics):', uncore_power)
    print()
    print('used power by this program')
    print('cpu:', cpu_power_use)
    print('dram:', dram_power_use)
