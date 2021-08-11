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
        msg = q.get(block=False)
        print(msg)
        total_percent = sum(msg.values())
        print("our programm used ", total_percent, "of the cpu times when we were measuring")
        break
    except EmptyQueueException:
        pass
