
# node name to csv column in the zip files
node_name_to_csv_column = {'n51': [30, 42],
 'n52': [32, 46],
 'n53': [34, 44],
 'n54': [36, 48],
 'n55': [38, 50],
 'n101': [40, 52],
 'MD1400': [6, 18],
 'backup': [8, 20],
 'service': [12, 22],
 'SWB1': [14, 16],
 'slurm': [24, 10],
 'n102': [26, 28],
 'n3': [6, 24],
 'n4': [8, 12],
 'n1': [10, 28],
 'switch 10Gpbs': [14, 18],
 'n5': [16, 20],
 'n2': [22, 26],
 'bee1': [30, 34],
 'bee2': [32, 36]}

# use to print a definition for the different fields
# keys of this dictionnary are keys used in the labia modules
# for instance, to write a summary of a job
mapping = {'number_of_jobs':'#jobs', 
    'date':'start date',
    'job_id':'job_id',
    'gpu_consumption':'GPU Consumption  (kWH)', 
    'cpu_consumption':'CPU Consumption (kWH)',
    'total_duration':'Total Duration (sec)',
    'cpu_ram_abs':'Avg. CPU RAM USAGE (Gb)',
    'gpu_ram': 'Avg. GPU RAM USAGE (Gb)',
    'omegawatt_power_draw':'Total Consumption (Ext. PowerMeter, kWH)',
    'relative_cpu_usage':'CPU usage (%)',
    'gpu_sm_usage':'GPU SM Usage (%)',
    'node':'node name'
    }

#baie gauche
node_to_column_usb1 = {'n3': {'slaves': ['slave1', 'slave2'],
  'columns': [1, 10],
  'alims': ['G', 'D']},
 'n4': {'slaves': ['slave1', 'slave1'],
  'columns': [2, 4],
  'alims': ['D', 'G']},
 'n1': {'slaves': ['slave1', 'slave2'],
  'columns': [3, 12],
  'alims': ['G', 'D']},
 'switch 10Gpbs': {'slaves': ['slave1', 'slave2'],
  'columns': [5, 7],
  'alims': ['D', 'G']},
 'n5': {'slaves': ['slave1', 'slave2'],
  'columns': [6, 8],
  'alims': ['D', 'G']},
 'n2': {'slaves': ['slave2', 'slave2'],
  'columns': [9, 11],
  'alims': ['G', 'D']},
 'bee1': {'slaves': ['slave3', 'slave3'],
  'columns': [13, 15],
  'alims': ['G', 'D']},
 'bee2': {'slaves': ['slave3', 'slave3'],
  'columns': [14, 16],
  'alims': ['D', 'G']}}

#baie droite
node_to_column_usb0 = {'n51': {'slaves': ['slave1', 'slave2'],
  'columns': [13, 19],
  'alims': ['G', 'D']},
 'n52': {'slaves': ['slave1', 'slave2'],
  'columns': [14, 21],
  'alims': ['G', 'D']},
 'n53': {'slaves': ['slave1', 'slave2'],
  'columns': [15, 20],
  'alims': ['G', 'D']},
 'n54': {'slaves': ['slave1', 'slave2'],
  'columns': [16, 22],
  'alims': ['G', 'D']},
 'n55': {'slaves': ['slave1', 'slave2'],
  'columns': [17, 23],
  'alims': ['G', 'D']},
 'n101': {'slaves': ['slave1', 'slave2'],
  'columns': [18, 24],
  'alims': ['G', 'D']},
 'MD1400': {'slaves': ['slave3', 'slave4'],
  'columns': [1, 7],
  'alims': ['D', 'G']},
 'backup': {'slaves': ['slave3', 'slave4'],
  'columns': [2, 8],
  'alims': ['D', 'D']},
 'service': {'slaves': ['slave3', 'slave4'],
  'columns': [4, 9],
  'alims': ['D', 'D']},
 'SWB1': {'slaves': ['slave3', 'slave3'],
  'columns': [5, 6],
  'alims': ['D', 'G']},
 'slurm': {'slaves': ['slave4','slave3'], 'columns': [10,3], 'alims': ['G','D']},
 'n102': {'slaves': ['slave4', 'slave4'],
  'columns': [11, 12],
  'alims': ['D', 'G']}}

RECORDING_RAPL_NVIDIA_DIR="/mnt/beegfs/power_monitor/prolog_log" 
RECORDING_OMEGA_WATT_DIR="/mnt/beegfs/home/gay/coca4ai/recordings/omegawatt_subsample/"

# voieNumber are the id used by omegawatt for example with number 3 (#current3,#activepow3)
# these numbers also correspond (almost always) to the tags written in the cables. 
voieNumber_to_csv_column = {1: 6,
 2: 8,
 3: 10,
 4: 12,
 5: 14,
 6: 16,
 7: 18,
 8: 20,
 9: 22,
 10: 24,
 11: 26,
 12: 28,
 13: 30,
 14: 32,
 15: 34,
 16: 36,
 17: 38,
 18: 40,
 19: 42,
 20: 44,
 21: 46,
 22: 48,
 23: 50,
 24: 52,
 25: 54,
 26: 56,
 27: 58,
 28: 60,
 29: 62,
 30: 64,
 31: 66,
 32: 68,
 33: 70,
 34: 72,
 35: 74,
 36: 76,
 37: 78,
 38: 80,
 39: 82,
 40: 84,
 41: 86,
 42: 88,
 43: 90,
 44: 92,
 45: 94,
 46: 96,
 47: 98,
 48: 100,
 49: 102,
 50: 104,
 51: 106,
 52: 108,
 53: 110,
 54: 112,
 55: 114,
 56: 116,
 57: 118,
 58: 120,
 59: 122}


# node name to csv column in the zip files
node_name_to_gpu_memory = {'n51': 32,
 'n52': 32,
 'n53': 32,
 'n54': 32,
 'n55': 32,
 'n101': 32,
 'MD1400': None,
 'backup': None,
 'service': None,
 'SWB1': None,
 'slurm': None,
 'n102': 32,
 'n3': 48,
 'n4': 48,
 'n1': 48,
 'switch 10Gpbs': None,
 'n5': 48,
 'n2': 48,
 'bee1': None,
 'bee2': None}