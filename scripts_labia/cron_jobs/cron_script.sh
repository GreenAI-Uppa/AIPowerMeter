#!/bin/bash

today=`date +"%m_%d_%Y"`
ps -aux | grep -i wattmetre-read | awk '{print $2}' | xargs kill

mv /mnt/beegfs/home/gay/coca4ai/recordings/log_omegawatt_USB1 /mnt/beegfs/home/gay/coca4ai/recordings/log_omegawatt_of_the_day
zip -j /mnt/beegfs/home/gay/coca4ai/recordings/omegawatt/${today}_USB1.zip /mnt/beegfs/home/gay/coca4ai/recordings/log_omegawatt_of_the_day
awk 'NR == 1 || NR == 2 || NR % 100 == 0' /mnt/beegfs/home/gay/coca4ai/recordings/log_omegawatt_of_the_day > /mnt/beegfs/home/gay/coca4ai/recordings/tmp/log_omegawatt_of_the_day
zip -j /mnt/beegfs/home/gay/coca4ai/recordings/omegawatt_subsample/${today}_USB1.zip /mnt/beegfs/home/gay/coca4ai/recordings/tmp/log_omegawatt_of_the_day

mv /mnt/beegfs/home/gay/coca4ai/recordings/log_omegawatt_USB0 /mnt/beegfs/home/gay/coca4ai/recordings/log_omegawatt_of_the_day
zip -j /mnt/beegfs/home/gay/coca4ai/recordings/omegawatt/${today}_USB0.zip /mnt/beegfs/home/gay/coca4ai/recordings/log_omegawatt_of_the_day
awk 'NR == 1 || NR == 2 || NR % 100 == 0' /mnt/beegfs/home/gay/coca4ai/recordings/log_omegawatt_of_the_day > /mnt/beegfs/home/gay/coca4ai/recordings/tmp/log_omegawatt_of_the_day
zip -j /mnt/beegfs/home/gay/coca4ai/recordings/omegawatt_subsample/${today}_USB0.zip /mnt/beegfs/home/gay/coca4ai/recordings/tmp/log_omegawatt_of_the_day

nohup /mnt/beegfs/home/gay/libs/omegawatt/wattmetre-read --tty=/dev/ttyUSB0 --nb=0 > /mnt/beegfs/home/gay/coca4ai/recordings/log_omegawatt_USB0 2>&1 &
nohup /mnt/beegfs/home/gay/libs/omegawatt/wattmetre-read --tty=/dev/ttyUSB1 --nb=0 > /mnt/beegfs/home/gay/coca4ai/recordings/log_omegawatt_USB1 2>&1 &

/mnt/beegfs/home/gay/miniconda3/bin/python /mnt/beegfs/home/gay/libs/read_results/append_summary_last_jobs.py
