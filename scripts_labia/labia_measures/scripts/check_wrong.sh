#!/bin/bash

# This script has the same functionment as "clean_jobs.sh", but instead of deleting the directories with missing files, it just shows them, and count them.

# Function to check directories and count those with missing files
check_and_report() {
    local parent_dir=$1
    local missing_count=0

    # Browse all directories at the specified depth
    for first_level_dir in "$parent_dir"/*/; do
        [ -d "$first_level_dir" ] || continue  # Check if it's a directory

        for second_level_dir in "$first_level_dir"/*/; do
            [ -d "$second_level_dir" ] || continue

            for third_level_dir in "$second_level_dir"/*/; do
                [ -d "$third_level_dir" ] || continue

                # Check that the third-level directory does not contain other subdirectories
                if [ -z "$(find "$third_level_dir" -mindepth 1 -type d)" ]; then
                    # Check the presence of the required files in the third-level directory
                    if [ ! -f "$third_level_dir/meta_data.json" ] || [ ! -f "$third_level_dir/power_metrics.json" ]; then
                        echo "In file $third_level_dir, files are missings"
                        # ls -lah "$third_level_dir"
                        missing_count=$((missing_count + 1))
                    fi
                fi
            done
        done
    done

    echo "Total numbers of incorrect directories  : $missing_count"
}

# Define the default directory
# This is why we need to browse third_level_dir, because :
# In this dir, there are directories named by date,
# Inside them, there are directories named by users,
# Inside them, there are directories named by the job id (which are supposed to contain meta_data.json and power_metrics.json)
default_dir="/mnt/beegfs/power_monitor/prolog_log"

# Use the default directory if no argument is passed
dir_to_check="${1:-$default_dir}"

# Launch the check from the directory
check_and_report "$dir_to_check"
