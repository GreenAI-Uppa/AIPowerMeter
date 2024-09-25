#!/bin/bash

# This script is made to be applied to the default dir, where we store json files for every recorded jobs.
# Some jobs may not have been recorded correctly, and this script deletes the records.
# However, not using this script will not affect the script "append_summary_last_jobs", it just corrects the warnings shown when using the program.

# Function to check and delete directories if required files are missing (meta_data.json and power_metrics.json are required, if one of them is missing, the directory is deleted)
check_and_delete() {
    local parent_dir=$1

    # Browse all directories at the specified depth
    for first_level_dir in "$parent_dir"/*/; do
        [ -d "$first_level_dir" ] || continue  # Check if it's a directory

        for second_level_dir in "$first_level_dir"/*/; do
            [ -d "$second_level_dir" ] || continue 

            for third_level_dir in "$second_level_dir"/*/; do
                [ -d "$third_level_dir" ] || continue  

                # Check that the third-level directory does not contain other subdirectories
                if [ ! -f "$third_level_dir/meta_data.json" ] || [ ! -f "$third_level_dir/power_metrics.json" ]; then
                    echo "We delete the following directory $third_level_dir because files are missing to make a summary."
                    rm -rf "$third_level_dir" # Delete the directory
                fi
            done
        done
    done
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
check_and_delete "$dir_to_check"
