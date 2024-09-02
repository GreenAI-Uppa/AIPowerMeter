#!/bin/bash

NODE_TOTAL=""
NODE_RESPONDING=""
NODE_IDLE=""
NODE_IDLE_LIST=""

GPU_TOTAL=""
GPU_USED=""
GPU_USER=""

CPU_TOTAL=""
CPU_USED=""
CPU_USER=""

# Function definitions from the first script
function Get_Total_Node_Count () {
  NODE_TOTAL=$(sinfo --noheader -p all -o %D);
  echo "${NODE_TOTAL}";
}

function Get_Responding_Node_Count () {
  NODE_RESPONDING=$(sinfo --noheader -p all --responding -o %D);
  echo "${NODE_RESPONDING}";
}

function Get_Available_Node_Count () {
  NODE_IDLE=$(sinfo --noheader -p all -t IDLE,MIX -o %D)
  echo "${NODE_IDLE}";
}

function Get_Available_Node_List () {
  NODE_IDLE_LIST=$(sinfo --noheader -p all -t IDLE,MIX -O Nodelist)
  echo "${NODE_IDLE_LIST}";
}

function Get_Gpu_Total_Count () {
  GPU_TOTAL=$(sinfo --noheader -p all -N -O gres \
            | grep 'gpu:' \
            | sed -e 's/gpu://' \
            | awk '{s+=$1} END {print s}')
  if [[ -z "${GPU_TOTAL}" ]]; then
    echo "0";
  else
    echo "${GPU_TOTAL}";
  fi;
}

function Get_Gpu_Used_Count () {
  GPU_USED=$(sacct --noheader -s RUNNING -a -X -o AllocTres -P \
           | grep 'gpu=' \
           | sed -e 's/.*gpu=//' \
           | awk '{s+=$1} END {print s}')
  if [[ -z "${GPU_USED}" ]]; then
    echo "0";
  else
    echo "${GPU_USED}";
  fi;
}

function Get_Gpu_User_Count () {
  GPU_USER=$(sacct -u $USER --noheader -s RUNNING -X -o AllocTres -P \
           | grep 'gpu=' \
           | sed -e 's/gpu=//' \
           | awk '{s+=$1} END {print s}')
  if [[ -z "${GPU_USER}" ]]; then
    echo "0";
  else
    echo "${GPU_USER}";
  fi;
}

function Get_Cpu_Total_Count () {
  CPU_TOTAL=$(sinfo --noheader -p all -N -O CPUS \
            | awk '{s+=$1} END {print s}')
  if [[ -z "${CPU_TOTAL}" ]]; then
    echo "0";
  else
    echo "${CPU_TOTAL}";
  fi;
}

function Get_Cpu_Used_Count () {
  CPU_USED=$(sacct --noheader -s RUNNING -a -X -o AllocCpu%-4 \
           | awk '{s+=$1} END {print s}')
  if [[ -z "${CPU_USED}" ]]; then
    echo "0";
  else
    echo "${CPU_USED}";
  fi;
}

function Get_Cpu_User_Count () {
  CPU_USER=$(sacct -u $USER --noheader -s RUNNING -X -o AllocCpu%-4 \
           | awk '{s+=$1} END {print s}')
  echo "${CPU_USER}";
}

function Get_User_Job_Status() {
  sacct -u $USER -s RUNNING,PENDING --format=JobID%8,JobName%20,State%8,Elapsed%10,AllocCpu%9,AllocTres%9
}

function Get_User_Storage_Status() {
  beegfs-ctl --getquota --gid --list $(groups | sed -e "s/ /,/g") \
  | tail -n +4;
}

function Last_Jobs_Recordings () {
# Print specific columns of the .csv for the current user.

# Get username
USER_NAME=$USER

# Path of csv files
FILE="/mnt/beegfs/projects/coca4ai/metrics/summary_per_user/${USER_NAME}.csv"

# Colors in ANSI
COLORS=(
    '\033[0;31m'  # Red
    '\033[0;32m'  # Green
    '\033[0;33m'  # Yellow
    '\033[0;34m'  # Blue
    '\033[0;35m'  # Magenta
    '\033[0;36m'  # Cyan
    '\033[0;37m'  # White
    '\033[1;31m'  # Light Red
    '\033[1;32m'  # Light Green
    '\033[1;33m'  # Light Yellow
    '\033[1;34m'  # Light Blue
    '\033[1;35m'  # Light Magenta
    '\033[1;36m'  # Light Cyan
)
NC='\033[0m' # No Color

LINE_LIMIT=5 # Default number of lines to display, without the header

#to change when we have lab-ia pue
PUE=1.36
# Function to calculate total consumption and total emissions and display default columns
display_default_columns() {
    awk -F, -v line_limit="$LINE_LIMIT" -v colors="${COLORS[*]}" -v nc="$NC" -v pue="$PUE" -v intensite_carbone_ademe=52 '
    BEGIN {
        split(colors, color_arr, " ")
        print color_arr[1] "node name" nc "," color_arr[2] "job_id" nc "," color_arr[3] "start date" nc "," color_arr[4] "Total duration" nc "," color_arr[5] "Ademe Emissions (gCO2eq)" nc "," color_arr[6] "Consumption (WH)" nc "," color_arr[7] "Total facility power (WH)" nc
    }
    NR>1 {
        total_consumption = $5 + $8;
        total_ademe_emission = (($5 + $8) / 1000) * intensite_carbone_ademe; # Calcul des émissions Ademe
        total_facility_power = total_consumption * pue;
        total_facility_power = sprintf("%.2f", total_facility_power);
        data[NR-1] = $1 "," $2 "," $3 "," $4 "," sprintf("%.2f", total_ademe_emission) "," total_consumption "," total_facility_power
    }
    END {
        start = (NR-1 > line_limit) ? NR-line_limit : 1
        for (i = start; i <= NR-1; i++) {
            split(data[i], fields, ",")
            for (j = 1; j <= 7; j++) {
                printf "%s%s%s", color_arr[j], fields[j], nc
                if (j < 7) {
                    printf ","
                } else {
                    printf "\n"
                }
            }
        }
        
        print ""
        print "Showed " ((NR-1 > line_limit) ? line_limit : NR-1) " jobs over " (NR-1) " recorded jobs"
    }' "$FILE"
}


# Execute default behavior
if [ -f "$FILE" ]; then
    display_default_columns
else
    echo "No logs recorded"
fi
}

function main () {
  echo "-- LAB-IA STATUS ----------------------------------------------------";
  echo;
  echo "NODE STATUS:     $(Get_Responding_Node_Count)/$(Get_Total_Node_Count)";
  echo "AVAILABLE NODES: $(Get_Available_Node_Count) - $(Get_Available_Node_List);"
  echo -n "AVAILABLE CPU:   $(($(Get_Cpu_Total_Count) - $(Get_Cpu_Used_Count)))/";
  echo "$(Get_Cpu_Total_Count)";
  echo -n "AVAILABLE GPU:   $(($(Get_Gpu_Total_Count)-$(Get_Gpu_Used_Count)))/";
  echo -n "$(Get_Gpu_Total_Count) - You are using $(Get_Gpu_User_Count) GPU";
  echo;
  echo "-- STORAGE ----------------------------------------------------------";
  Get_User_Storage_Status;
  echo;
  echo "-- JOB STATUS -------------------------------------------------------";
  Get_User_Job_Status;
  echo;
  echo "-- LAST JOBS RECORDINGS ---------------------------------------------";
  Last_Jobs_Recordings;
  echo -e "\nIf you wish to have more informations about your jobs, you can use the following command : TODO -h";
}

main;
