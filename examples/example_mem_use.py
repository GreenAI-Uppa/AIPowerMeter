import psutil, os
from deep_learning_power_measure.power_measure import rapl_power
from multiprocessing import Process, Queue
from queue import Empty as EmptyQueueException
import time
import numpy as np

def f(q, process_list):
    time.sleep(10)
    mem_use = rapl_power.get_mem_uses(process_list)
    q.put(mem_use)

current_process = psutil.Process(os.getppid())
pid_list = [current_process.pid] + [
    child.pid for child in current_process.children(recursive=True)
]
print('I will measure the cpu use of the following processes',pid_list)
process_list = rapl_power.get_processes(pid_list)

n = 10000
a = np.random.rand(n,n)
b = np.random.rand(n,n)

q = Queue()
p = Process(target=f, args=(q,process_list))
p.start()
while True:
    np.matmul(a,b)
    try:
        mem_pss_per_process, mem_uss_per_process = q.get(block=False)
        uss = sum(mem_uss_per_process.values())
        pss = sum(mem_pss_per_process.values())
        print("USS value : ", uss,  " : 'Unique Set Size', this is the memory which is unique to a process and which would be freed if the process was terminated right now")
        print("PSS (or rss if not available ):", pss, "  : 'Proportional Set Size', is the amount of memory shared with other processes. Linux only")
        print()
        print("Our programm used ", uss + pss, "bytes from the memory")
        break
    except EmptyQueueException:
        pass
