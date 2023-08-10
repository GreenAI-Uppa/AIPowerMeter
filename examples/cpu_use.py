import os
from multiprocessing import Process, Queue
from queue import Empty as EmptyQueueException
import psutil
import numpy as np
from deep_learning_power_measure.power_measure import rapl_power

def f(q, process_list):
    """get the cpu uses and place them in the queue"""
    cpu_use, absolute_cpu_time_per_pid = rapl_power.get_cpu_uses(process_list, period=5.0)
    q.put((cpu_use,absolute_cpu_time_per_pid))

current_process = psutil.Process(os.getppid())
pid_list = [current_process.pid] + [
    child.pid for child in current_process.children(recursive=True)
]
print('I will measure the cpu use of the following processes',pid_list)
process_list = rapl_power.get_processes(pid_list)

n = 1000
a = np.random.rand(n,n)
b = np.random.rand(n,n)

q = Queue()
p = Process(target=f, args=(q,process_list))
p.start()
while True:
    np.matmul(a,b)
    try:
        cpu_use,absolute_cpu_time_per_pid = q.get(block=False)
        total_percent = sum(cpu_use.values())
        total_time = sum(absolute_cpu_time_per_pid.values())
        print("our programm used ", total_percent, "of the cpu times when we were measuring")
        print("our programm used ", total_time, "of cpu time")
        break
    except EmptyQueueException:
        pass
