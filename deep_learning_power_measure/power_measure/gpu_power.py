from subprocess import PIPE, Popen
import subprocess
import numpy as np
import pandas as pd
import re
from collections import OrderedDict
from io import StringIO
import csv
from xml.etree.ElementTree import fromstring


def is_nvidia_compatible():
    """
    return boolean corresponding to checking Check if this system supports nvidia tools required
    """
    from shutil import which

    if which("nvidia-smi") is None:
        return False

    # make sure that nvidia-smi doesn't just return no devices
    p = Popen(["nvidia-smi"], stdout=PIPE)
    stdout, stderror = p.communicate()
    output = stdout.decode("UTF-8")
    if "no devices" in output.lower():
        return False
    if "NVIDIA-SMI has failed".lower() in output.lower():
        return False
    return True


def get_process_nvidia_use(nsample=1):
    sp = subprocess.Popen(
        ["nvidia-smi", "pmon", "-c", str(nsample)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out_str = sp.communicate()
    return out_str

def get_gpu_power():
    p = subprocess.Popen(["nvidia-smi", "-q", "-x"], stdout=subprocess.PIPE)
    outs, errors = p.communicate()
    #xml = fromstring(outs)
    return outs

def get_gpu_use(nsample=1):
    #Find per process per gpu usage info
    # -c corresponds to the number of samples
    # according to the docs, this command is limited to 4 gpus
    # information includes the pid, command name and       average utilization values for SM (streaming multiprocessor), Memory, Encoder  and  Decoder  since       the  last  monitoring  cycle
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
    out_str_final = re.sub("\s+\n", "\n", out_str_final)  # else pd will mis-align
    out_str_final = out_str_final.strip()
    df = pd.read_csv(StringIO(out_str_final), engine="python", delimiter="\t")
    return df

def get_nvidia_gpu_power(pid_list, nsample = 1, logger=None, **kwargs):
    """
    first, get gpu usage per process
       nsample indicates the number of queries to nvidia
    second get the power use of nvidia
    """
    df = get_gpu_use(nsample=nsample)
    # result is a panda frame with the following columns
    # gpu   pid    sm   mem  enc  dec
    process_percentage_used_gpu = df.groupby(["gpu", "pid"]).mean().reset_index()

    # this commmand provides the full xml output
    p = subprocess.Popen(["nvidia-smi", "-q", "-x"], stdout=subprocess.PIPE)
    outs, errors = p.communicate()
    xml = fromstring(outs)
    num_gpus = int(xml.findall("attached_gpus")[0].text)
    results = []
    power = 0
    per_gpu_absolute_percent_usage = {}
    per_gpu_relative_percent_usage = {}
    absolute_power = 0
    per_gpu_performance_states = {}

    # now double loop
    # for each gpu
    #    for each pid
    #        collect the amount of power the process's pid is consuming
    for gpu_id, gpu in enumerate(xml.findall("gpu")):
        gpu_data = {}

        name = gpu.findall("product_name")[0].text
        gpu_data["name"] = name

        # get memory
        memory_usage = gpu.findall("fb_memory_usage")[0]
        total_memory = memory_usage.findall("total")[0].text
        used_memory = memory_usage.findall("used")[0].text
        free_memory = memory_usage.findall("free")[0].text
        gpu_data["memory"] = {
            "total": total_memory,
            "used_memory": used_memory,
            "free_memory": free_memory,
        }

        # get utilization
        utilization = gpu.findall("utilization")[0]
        gpu_util = utilization.findall("gpu_util")[0].text
        memory_util = utilization.findall("memory_util")[0].text
        gpu_data["utilization"] = {"gpu_util": gpu_util, "memory_util": memory_util}

        # get power
        power_readings = gpu.findall("power_readings")[0]
        power_draw = power_readings.findall("power_draw")[0].text

        gpu_data["power_readings"] = {"power_draw": power_draw}
        absolute_power += float(power_draw.replace("W", ""))

        # processes
        processes = gpu.findall("processes")[0]

        infos = []
        # all the info for processes on this particular gpu that we're on
        gpu_based_processes = process_percentage_used_gpu[
            process_percentage_used_gpu["gpu"] == gpu_id
        ]
        # what's the total absolute SM for this gpu across all accessible processes
        percentage_of_gpu_used_by_all_processes = float(gpu_based_processes["sm"].sum())
        per_gpu_power_draw = {}
        for info in processes.findall("process_info"):
            pid = info.findall("pid")[0].text
            process_name = info.findall("process_name")[0].text
            used_memory = info.findall("used_memory")[0].text
            sm_absolute_percent = gpu_based_processes[
                gpu_based_processes["pid"] == int(pid)
            ]["sm"].sum()
            if percentage_of_gpu_used_by_all_processes == 0:
                # avoid divide by zero, sometimes nothing is used so 0/0 should = 0 in this case
                sm_relative_percent = 0
            else:
                sm_relative_percent = (
                    sm_absolute_percent / percentage_of_gpu_used_by_all_processes
                )
            infos.append(
                {
                    "pid": pid,
                    "process_name": process_name,
                    "used_memory": used_memory,
                    "sm_relative_percent": sm_relative_percent,
                    "sm_absolute_percent": sm_absolute_percent,
                }
            )

            if int(pid) in pid_list:
                # only add a gpu to the list if it's being used by one of the processes. sometimes nvidia-smi seems to list all gpus available
                # even if they're not being used by our application, this is a problem in a slurm setting
                if gpu_id not in per_gpu_absolute_percent_usage:
                    # percentage_of_gpu_used_by_all_processes
                    per_gpu_absolute_percent_usage[gpu_id] = 0
                if gpu_id not in per_gpu_relative_percent_usage:
                    # percentage_of_gpu_used_by_all_processes
                    per_gpu_relative_percent_usage[gpu_id] = 0

                if gpu_id not in per_gpu_performance_states:
                    # we only log information for gpus that we're using, we've noticed that nvidia-smi will sometimes return information
                    # about all gpu's on a slurm cluster even if they're not assigned to a worker
                    performance_state = gpu.findall("performance_state")[0].text
                    per_gpu_performance_states[gpu_id] = performance_state

                power += sm_relative_percent * float(power_draw.replace("W", ""))
                per_gpu_power_draw[gpu_id] = float(power_draw.replace("W", "")) # this could above this loop
                # want a proportion value rather than percentage
                per_gpu_absolute_percent_usage[gpu_id] += sm_absolute_percent / 100.0
                per_gpu_relative_percent_usage[gpu_id] += sm_relative_percent

        gpu_data["processes"] = infos

        results.append(gpu_data)

    if len(per_gpu_absolute_percent_usage.values()) == 0:
        average_gpu_utilization = 0
        average_gpu_relative_utilization = 0
    else:
        average_gpu_utilization = np.mean(list(per_gpu_absolute_percent_usage.values()))
        average_gpu_relative_utilization = np.mean(
            list(per_gpu_relative_percent_usage.values())
        )
    per_gpu_average_estimated_utilization_absolute = []
    for i, row in process_percentage_used_gpu.iterrows():
        d = dict([(k,float(row[k])) for k in process_percentage_used_gpu.columns] )
        per_gpu_average_estimated_utilization_absolute.append(d)
    data_return_values_with_headers = {
        "nvidia_draw_absolute": absolute_power,
        "nvidia_estimated_attributable_power_draw": power,
        "average_gpu_estimated_utilization_absolute": average_gpu_utilization,
        "per_gpu_average_estimated_utilization_absolute": per_gpu_average_estimated_utilization_absolute,
        "average_gpu_estimated_utilization_relative": average_gpu_relative_utilization,
        "per_gpu_performance_state": per_gpu_performance_states,
        "per_gpu_power_draw": per_gpu_power_draw,
    }

    return data_return_values_with_headers

"""
{'nvidia_draw_absolute': 25.27,
 'nvidia_estimated_attributable_power_draw': 0,
 'average_gpu_estimated_utilization_absolute': 0,
 'per_gpu_average_estimated_utilization_absolute': [{'gpu': 0,
   'pid': 902,
   'sm': 0,
   'mem': 0,
   'enc': 0,
   'dec': 0},
  {'gpu': 0, 'pid': 981, 'sm': 0, 'mem': 0, 'enc': 0, 'dec': 0}],
 'average_gpu_estimated_utilization_relative': 0,
 'per_gpu_performance_state': {},
 'per_gpu_power_draw': {}}
 """
