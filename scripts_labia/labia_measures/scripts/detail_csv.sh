#!/bin/bash

# Print the header of the .csv and the 5 last jobs of the current user.

# Get username
#USER_NAME='cuevas_villarmin'
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


PUE=1.36

# Values in gC02eq
TREE_MONTH=917
CAR_PERKM=175
FLIGHT_PAR_LON=50000

# Function to colorize a line
colorize_line() {
    local line=$1
    local IFS=',' # , as delimiter
    local i=0
    local output=""
    for col in $line; do
        output+="${COLORS[i % ${#COLORS[@]}]}$col${NC},"
        ((i++))
    done
    # Remove the trailing comma and print the line
    echo -e "${output%,}"
}

calculate_averages() {
    total_jobs=$(awk 'END {print NR-1}' "$FILE")
    awk -F, -v pue=1.36 -v intensite_carbone_ademe=52 '
    function parse_duration(duration) {
        match(duration, /([0-9]+)h/, a); h = (a[1] != "") ? a[1] : 0;
        match(duration, /([0-9]+)m/, a); m = (a[1] != "") ? a[1] : 0;
        match(duration, /([0-9]+)s/, a); s = (a[1] != "") ? a[1] : 0;
        return h * 3600 + m * 60 + s;
    }
    NR>1 {
        count++;
        duration = parse_duration($4);
        total_duration += duration;
        
        gpu_consumption += $5;
        gpu_sm_usage += $7;
        cpu_consumption += $8;
        cpu_usage += $10;
        
        gpu_emission_rte += $11;
        cpu_emission_rte += $12;

        # Calcul des émissions ADEME
        gpu_emission_ademe += ($5 / 1000) * intensite_carbone_ademe;
        cpu_emission_ademe += ($8 / 1000) * intensite_carbone_ademe;
        
        
        }
    END {
        if (count > 0) {
            printf "Total duration:                                         %dh%dm%ds\n", total_duration/3600, (total_duration%3600)/60, total_duration%60;
            printf "Average duration per job:                               %dh%dm%ds\n", total_duration/count/3600, (total_duration/count%3600)/60, total_duration/count%60;
            printf "GPU Metrics:\n";
            printf "  Average GPU consumption :                             %.2f WH\n", gpu_consumption/count;
            printf "  Average GPU SM usage :                                %.2f %%\n", gpu_sm_usage/count;
            printf "  Average GPU emission (ADEME) :                        %.2f gC02eq\n", gpu_emission_ademe/count;
            printf "  Average GPU emission (RTE) :                          %.2f gC02eq\n", gpu_emission_rte/count;
            printf "CPU Metrics:\n";
            printf "  Average CPU consumption:                              %.2f WH\n", cpu_consumption/count;
            printf "  Average CPU usage :                                   %.2f %%\n", cpu_usage/count;
            printf "  Average CPU emission (ADEME) :                        %.2f gCO2eq\n", cpu_emission_ademe/count;
            printf "  Average CPU emission (RTE) :                          %.2f gC02eq\n", cpu_emission_rte/count;
            printf "Average total facility power consumption:               %.2f WH\n", pue*(gpu_consumption + cpu_consumption)/count;
        } else {
            print "No data to calculate averages";
        }
    printf "Displayed average of %d recorded jobs\n", count
    }' "$FILE"
}



display_co2_emissions() {
    awk -F, -v line_limit="$LINE_LIMIT" -v colors="${COLORS[*]}" -v nc="$NC" -v tree_month="$TREE_MONTH" -v car_perkm="$CAR_PERKM" -v flight_par_lon="$FLIGHT_PAR_LON" -v intensite_carbone_ademe=52 '    
    function formatText_treemonths(tm_float) {
        tm = int(tm_float)
        ty = int(tm / 12)
        if (tm_float < 1) {
            text_trees = sprintf("%.3f tree-months", tm_float)
        } else if (tm == 1) {
            text_trees = sprintf("%.1f tree-month", tm_float)
        } else if (tm < 6) {
            text_trees = sprintf("%.1f tree-months", tm_float)
        } else if (tm <= 24) {
            text_trees = sprintf("%d tree-months", tm)
        } else if (tm < 120) {
            text_trees = sprintf("%d tree-years and %d tree-months", ty, tm - ty * 12)
        } else {
            text_trees = sprintf("%d tree-years", ty)
        }
        return text_trees
    }
    BEGIN {
        if (tree_month == 0) {
            print "Error: tree_month cannot be zero."
            exit 1
        }
        split(colors, color_arr, " ")
        print color_arr[1] "node_name" nc "," color_arr[2] "job_id" nc "," color_arr[3] "start date" nc "," color_arr[4] "total duration" nc "," color_arr[5] "Total emission ADEME (gCO2eq)" nc "," color_arr[6] "Total emission RTE (gCO2eq)" nc "," color_arr[7] "GPU Emission ADEME (gCO2eq)" nc "," color_arr[8] "GPU Emission RTE (gCO2eq)" nc "," color_arr[9] "CPU Emission ADEME (gCO2eq)" nc "," color_arr[10] "CPU Emission RTE (gCO2eq)" nc
    }
    NR>1 {
        total_emission_ademe = $12 + $14; # Calcul des émissions totales ADEME
        gpu_emission_ademe = $12;         # Calcul des émissions GPU ADEME
        cpu_emission_ademe = $14;         # Calcul des émissions CPU ADEME
        total_emission_rte = $11 + $13;                                      # Émissions RTE existantes
        data[NR-1] = $1 "," $2 "," $3 "," $4 "," sprintf("%.2f", total_emission_ademe) "," sprintf("%.2f", total_emission_rte) "," sprintf("%.2f", gpu_emission_ademe) "," $11 "," sprintf("%.2f", cpu_emission_ademe) "," $13;
        total_emission_sum_ademe += total_emission_ademe;
        total_emission_sum_rte += total_emission_rte;
    }
    END {
        tm_float_ademe = total_emission_sum_ademe / tree_month;
        tm_float_rte = total_emission_sum_rte / tree_month;
        tree_months_ademe = formatText_treemonths(tm_float_ademe);
        tree_months_rte = formatText_treemonths(tm_float_rte);
        driving_km_ademe = total_emission_sum_ademe / car_perkm;
        driving_km_rte = total_emission_sum_rte / car_perkm;
        
        start = (NR-1 > line_limit) ? NR-line_limit : 1
        for (i = start; i <= NR-1; i++) {
            split(data[i], fields, ",")
            for (j = 1; j <= 10; j++) {
                printf "%s%s%s", color_arr[j], fields[j], nc
                if (j < 10) {
                    printf ","
                } else {
                    printf "\n"
                }
            }
        }
        
        printf "\n%sTotal Emissions%s: %.2f gCO2eq (ADEME) & %.2f gCO2eq (RTE)\n", color_arr[5], nc, total_emission_sum_ademe, total_emission_sum_rte  # Arrondi à 2 chiffres après la virgule
        printf "%sTree-months%s: %s (ADEME) & %s (RTE)\n", color_arr[8], nc, tree_months_ademe, tree_months_rte
        printf "%sEquivalent driving distance%s: %.2f km (ADEME) & %.2f km (RTE)\n", color_arr[9], nc, driving_km_ademe, driving_km_rte
        printf "%sEquivalent flight Paris-London%s: %.2f (ADEME) & %.2f (RTE)\n", color_arr[10], nc, total_emission_sum_ademe / flight_par_lon, total_emission_sum_rte / flight_par_lon
        printf "Showed %d jobs over %d recorded jobs\n", (NR-1 > line_limit) ? line_limit : NR-1, NR-1
        printf "\nADEME website : https://www.ademe.fr/, RTE website : https://www.rte-france.com/eco2mix\nThe main reason for the difference between the two emissions is the scope of the emissions taken into account.\n"
    }' "$FILE"
}

# Function to display all CSV content
display_all_csv() {
    cat "$FILE"
}

display_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -h, --help           Display this help message"
    echo "  -j, --job <job_id>   Display the log for the specific job ID"
    echo "  -a, --average        Display the average metrics"
    echo "  -c, --co2            Display CO2 emissions for the last jobs"
    echo "  -l, --limit <n>      Set the number of jobs to display (default is 5)"
    echo "  -s, --summary        Display summary columns"
    echo "  --all                Display all the CSV content"
    echo "  No options           Display the last 5 jobs"
}





display_summary() {
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


# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -j|--job)
        JOB_IDS=()
        shift
        while [[ $# -gt 0 && $1 != -* ]]; do
            JOB_IDS+=("$1")
            shift
        done
        ;;
        -a|--average)
        AVERAGE=true
        shift
        ;;
        -c|--co2)
        CO2=true
        shift
        ;;
        -l|--limit)
        LINE_LIMIT="$2"
        shift
        shift
        ;;
        -s|--summary)
        SUMMARY=true
        shift
        ;;
        --all)
        ALL=true
        shift
        ;;
        -h|--help)
        display_help
        exit 0
        ;;
        *)
        echo "Invalid option: $1"
        display_help
        exit 1
        ;;
    esac
done

# Execute based on parsed arguments
if [ -f "$FILE" ]; then
    if [ "$ALL" = true ]; then
        display_all_csv
    elif [ "$AVERAGE" = true ]; then
        calculate_averages
    elif [ "$CO2" = true ]; then
        display_co2_emissions
    elif [ "$SUMMARY" = true ]; then
        display_summary
    elif [ ! -z "${JOB_IDS+x}" ]; then
        header=$(head -n 1 "$FILE")
        colorize_line "$header"
        found_any=false
        for job_id in "${JOB_IDS[@]}"; do
            job_line=$(awk -F, -v job_id="$job_id" '$2 == job_id' "$FILE")
            if [ -n "$job_line" ]; then
                colorize_line "$job_line"
                found_any=true
            fi
        done
        if [ "$found_any" = false ]; then
            echo "No logs for the provided job IDs"
        fi
    else
        # Default behavior
        first_line=$(head -n 1 "$FILE")
        colorize_line "$first_line"
        tail_lines=$(tail -n +2 "$FILE")
        num_lines=$(echo "$tail_lines" | wc -l)
        if [ "$num_lines" -gt "$LINE_LIMIT" ]; then
            echo "$tail_lines" | tail -n "$LINE_LIMIT" | while IFS= read -r line; do colorize_line "$line"; done
        else
            echo "$tail_lines" | while IFS= read -r line; do colorize_line "$line"; done
        fi
        echo ""
        echo "Showed $(($num_lines > $LINE_LIMIT ? $LINE_LIMIT : $num_lines)) jobs over $num_lines recorded jobs"
    fi
else
    echo "No logs recorded"
fi

