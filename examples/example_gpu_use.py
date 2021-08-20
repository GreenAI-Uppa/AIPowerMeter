from deep_learning_power_measure.power_measure import gpu_power 
import subprocess
import pandas as pd
import re
from collections import OrderedDict
from io import StringIO

print(gpu_power.get_gpu_use(nsample=1))
