"""This module parses the xml provided by nvidia-smi to obtain the consumption, memory and SM used for each gpu and each pid."""

from subprocess import PIPE, Popen
import re
import subprocess
from collections import OrderedDict
from io import StringIO
from xml.etree.ElementTree import fromstring
from shutil import which
import numpy as np
import pandas as pd

def is_nvidia_compatible():
    """
    return boolean corresponding to checking Check if this system supports nvidia tools required
    """

    msg = "nvidia NOT available for energy consumption\n"
    if which("nvidia-smi") is None:
        return (False, msg+"nvidia-smi program is not in the path")
    # make sure that nvidia-smi doesn't just return no devices
    p = Popen(["nvidia-smi"], stdout=PIPE)
    stdout, _ = p.communicate()
    output = stdout.decode("UTF-8")
    if "no devices" in output.lower():
        return (False, msg+"nvidia-smi did not found GPU device on this machine")
    if "NVIDIA-SMI has failed".lower() in output.lower():
        return (False, msg+output)
    xml = get_nvidia_xml()
    for _, gpu in enumerate(xml.findall("gpu")):
        try:
            get_gpu_data(gpu)
        except RuntimeError as e:
            return (False, msg+e.__str__())
        break
    msg = "GPU power will be measured with nvidia"
    return True, msg

def get_gpu_use_pmon(nsample=1):
    """
    Find per process per gpu usage info
     -c corresponds to the number of samples
     according to the docs, this command is limited to 4 gpus
     information includes the pid, command name and       average utilization values for SM (streaming multiprocessor), Memory, Encoder  and  Decoder  since       the  last  monitoring  cycle
     result is a panda frame with the following columns
     gpu   pid    sm   mem  enc  dec
    """
    sp = subprocess.Popen(
        ["nvidia-smi", "pmon", "-c", str(nsample)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out_str = sp.communicate()

    # now doing some processing to extract the info from the command output string
    out_str_split = out_str[0].decode("utf-8").split("\n")
    # sometimes with too many processess on the machine or too many gpus, this command will reprint the headers
    # to avoid that we just remove duplicate lines
    out_str_split = list(OrderedDict.fromkeys(out_str_split))
    # remove the idx line
    out_str_pruned = [
        x for x in out_str_split if "Idx" not in x
    ]  # [out_str_split[0], ] + out_str_split[2:]

    # For some weird reason the header position sometimes gets jumbled so we need to re-order it to the front
    position = -1

    for i, x in enumerate(out_str_pruned):
        if re.match(r"#.* gpu ", x):
            position = i
    if position == -1:
        raise ValueError("Problem with output in nvidia-smi pmon -c 10")
    out_str_pruned.insert(0, out_str_pruned.pop(position).strip())
    out_str_final = "\n".join(out_str_pruned)
    out_str_final = out_str_final.replace("-", "0")
    out_str_final = out_str_final.replace("# ", "")
    out_str_final = re.sub(
        "  +", "\t", out_str_final
    )  # commands may have single spaces
    out_str_final = re.sub("\n\t", "\n", out_str_final)  # remove preceding space
    out_str_final = re.sub(r"\s+\n", "\n", out_str_final)  # else pd will mis-align
    out_str_final = out_str_final.strip()
    df = pd.read_csv(StringIO(out_str_final), engine="python", delimiter="\t")
    process_percentage_used_gpu = df.groupby(["gpu", "pid"]).mean().reset_index()
    per_gpu_per_pid_utilization_absolute = {}
    for _, row in process_percentage_used_gpu.iterrows():
        gpu_id = int(row['gpu'])
        if gpu_id not in per_gpu_per_pid_utilization_absolute:
            per_gpu_per_pid_utilization_absolute[gpu_id] = {}
        pid = int(row['pid'])
        per_gpu_per_pid_utilization_absolute[gpu_id][pid] = row['sm']/100
    return per_gpu_per_pid_utilization_absolute 

def get_gpu_mem(gpu):
    """Get the gpu memory usage from one gpu"""
    memory_usage = gpu.findall("fb_memory_usage")[0]
    total_memory = memory_usage.findall("total")[0].text
    used_memory = memory_usage.findall("used")[0].text
    free_memory = memory_usage.findall("free")[0].text
    per_pid_mem_use = {}
    processes = gpu.findall("processes")[0]
    for info in processes.findall("process_info"):
        pid = info.findall("pid")[0].text
        # memory used for this pid for this gpu
        used_memory = info.findall("used_memory")[0].text
        per_pid_mem_use[int(pid)] = int(used_memory.replace('MiB',''))*1048576 # convert from mbytes to bytes
    return {
        "total": total_memory,
        "used_memory": used_memory,
        "free_memory": free_memory,
        "per_pid_mem_use": per_pid_mem_use
    }

def get_gpu_use(gpu):
    """Get gpu utilization and memory usage"""
    utilization = gpu.findall("utilization")[0]
    gpu_util = utilization.findall("gpu_util")[0].text
    memory_util = utilization.findall("memory_util")[0].text
    return {"gpu_util": gpu_util, "memory_util": memory_util}

def get_gpu_power(gpu):
    """get the power draw for this gpu"""
    power_readings = gpu.findall("power_readings")[0]
    power_draw = power_readings.findall("power_draw")[0].text
    if power_draw  == 'N/A':
        raise RuntimeError("nvidia-smi could not retrieve power draw from the nvidia card. Check that it is supported on your hardware ?")
    power_draw = float(power_draw.replace("W", ""))
    return {"power_draw": power_draw}

def get_gpu_data(gpu):
    """get consumption, SM and memory use for one gpu

    Args:
        gpu: xml part regarding one specific gpu
    """
    gpu_data = {}
    name = gpu.findall("product_name")[0].text
    gpu_data["name"] = name
    gpu_data["memory"] = get_gpu_mem(gpu)
    gpu_data["utilization"] = get_gpu_use(gpu)
    gpu_data["power_readings"] = get_gpu_power(gpu)
    return gpu_data

def get_nvidia_xml():
    """Call nvidia-smi program to obtain the details about the GPUs"""
    p = subprocess.Popen(["nvidia-smi", "-q", "-x"], stdout=subprocess.PIPE)
    outs, _ = p.communicate()
    xml = fromstring(outs)
    return xml

def get_min_power():
    min_powers = {}
    xml = get_nvidia_xml()
    for gpu_id, gpu in enumerate(xml.findall("gpu")):
        power_readings = gpu.findall("power_readings")[0]
        power_min = power_readings.findall("min_power_limit")[0].text
        min_powers[gpu_id] = float(power_min.replace('W',''))
    return min_powers

def get_nvidia_gpu_power(pid_list=None, nsample = 1):
    """Get the power and use of each GPU.
    first, get gpu usage per process
    second get the power use of nvidia for each GPU
    then for each gpu and each process in pid_list compute its attributatble
    power

    Args:
        pid_list : list of processes to be measured

        nsample : number of queries to nvidia

    """
    # collect per gpu per pid sm usage
    per_gpu_per_pid_utilization_absolute = get_gpu_use_pmon(nsample=nsample)

    # this commmand provides the full xml output
    xml = get_nvidia_xml()
    power = 0 # power attributed to the pids involved in the experiment over all the gpus
    per_gpu_power_draw = {}
    per_gpu_per_pid_mem_use = {}

    # for each gpu
    #    collect memory, power draw and state for each pid
    for gpu_id, gpu in enumerate(xml.findall("gpu")):
        gpu_data = get_gpu_data(gpu)
        per_gpu_power_draw[gpu_id] = gpu_data["power_readings"]["power_draw"] # power_this_gpu
        per_gpu_per_pid_mem_use[gpu_id] = dict(
                [ (pid, use) for (pid,use) in gpu_data['memory']['per_pid_mem_use'].items() if pid_list is None or pid in pid_list])

    # power attributed over all the gpus involved in the experiment
    absolute_power = sum(per_gpu_power_draw.values())
    
    # for each gpu, get percentage of sm used for all the pid involved the experiment
    per_gpu_absolute_percent_usage = {} 
    for gpu_id, pids in per_gpu_per_pid_utilization_absolute.items():
        per_gpu_absolute_percent_usage[gpu_id] = 0 
        for pid, sm_use in pids.items():
            if pid_list is None or pid in pid_list: 
                per_gpu_absolute_percent_usage[gpu_id] += sm_use

    # relative value compared to per_gpu_absolute_percent_usage 
    per_gpu_relative_percent_usage = {} 
    for gpu_id, pids in per_gpu_per_pid_utilization_absolute.items():
        all_sm = sum([ v for (pid,v) in pids.items()])
        this_exp_sm = per_gpu_absolute_percent_usage[gpu_id]
        if this_exp_sm == 0:
            per_gpu_relative_percent_usage[gpu_id] = 0
        else:
            per_gpu_relative_percent_usage[gpu_id] = all_sm / this_exp_sm

    data_return_values_with_headers = {
        "nvidia_draw_absolute": absolute_power, # total nvidia power draw
        "per_gpu_power_draw": per_gpu_power_draw,
        "per_gpu_attributable_mem_use": per_gpu_per_pid_mem_use,
        "per_gpu_per_pid_utilization_absolute": per_gpu_per_pid_utilization_absolute, # absolute % of sm used per gpu per pid
        "per_gpu_absolute_percent_usage": per_gpu_absolute_percent_usage, # absolute % of sm used per gpu by the experiment
        "per_gpu_estimated_attributable_utilization": per_gpu_relative_percent_usage, # relative use of sm used per gpu by the experiment
    }
    return data_return_values_with_headers
