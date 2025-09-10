#!/bin/bash

# SceneCrafter Auto Render Script
# Copyright (c) 2024 SceneCrafter Contributors
# Licensed under the MIT License - see LICENSE file for details
#
# This script automatically manages GPU resources and distributes rendering tasks
# across available GPUs based on memory availability. It monitors GPU status,
# launches rendering jobs, and handles cleanup on exit.
#
# Usage: ./auto_render.sh [experiment_name_suffix]
#   - Without arguments: Uses current timestamp as experiment name
#   - With argument: Uses provided suffix in traffic_gen_<suffix> format

# Check input parameters and set experiment name with timestamp
if [ -z "$1" ]; then
    now=$(date +"%Y%m%d_%H%M%S")
    exp_name="traffic_gen_$now"
else
    now=$(echo $1 | awk -F_ '{print $(NF-1)"_"$NF}')
    exp_name="traffic_gen_$now"
fi

# Create log directory for this experiment
mkdir -p "./scripts/logs/$now"

# Minimum free GPU memory required (in MB)
MIN_FREE_MEMORY=9000

# Task file containing rendering commands for this experiment
task_file="./scripts/tasks/render_tasks_$now.txt"

# Generate rendering task file
python ./scripts/tasks/generate_render_tasks.py \
    --exp_name "$exp_name" \
    --output "$task_file"

# Array tracking currently running tasks (format: gpu_index:pid:command)
running_tasks=()

# Function: Get list of available GPUs with sufficient memory
get_available_gpus() {
    local available_gpus=()
    local gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    while IFS=',' read -r index memory; do
        if (( memory >= MIN_FREE_MEMORY )); then
            available_gpus+=("$index")
        fi
    done <<< "$gpu_info"
    echo "${available_gpus[@]}"
}

# Function: Start a new rendering task on specified GPU
start_task() {
    local gpu_index=$1
    local task_command=$(head -n 1 "$task_file" && sed -i '1d' "$task_file")
    if [ -z "$task_command" ]; then
        return  # No more tasks available
    fi
    
    # Set GPU for this task
    export CUDA_VISIBLE_DEVICES=$gpu_index
    
    # Execute task and redirect output to log file
    $task_command > "./scripts/logs/$now/output_gpu_$gpu_index.log" 2>&1 &
    local pid=$!
    running_tasks+=("$gpu_index:$pid:$task_command")
    echo "TASK BEGIN: GPU $gpu_index -> PID $pid -> COMMAND $task_command"
}

# Function: Check task status and clean up completed tasks
check_running_tasks() {
    local new_running_tasks=()
    for task in "${running_tasks[@]}"; do
        IFS=':' read -r gpu_index pid command <<< "$task"
        if kill -0 $pid > /dev/null 2>&1; then
            new_running_tasks+=("$gpu_index:$pid:$command")
        else
            echo "TASK COMPLETE: GPU $gpu_index -> PID $pid -> COMMAND $command"
        fi
    done
    running_tasks=("${new_running_tasks[@]}")
}

# Function: Clean up all running tasks on exit
# This function is called when script receives interrupt signals or exits normally
cleanup_tasks() {
    echo "Cleaning up all running tasks..."
    for task in "${running_tasks[@]}"; do
        IFS=':' read -r gpu_index pid command <<< "$task"
        if kill -0 $pid > /dev/null 2>&1; then
            echo "Terminating task on GPU $gpu_index with PID $pid"
            kill -9 $pid > /dev/null 2>&1  # Force terminate task
        fi
    done
    running_tasks=()  # Clear task list
}

# Trap signals to ensure proper cleanup on script termination
trap 'echo "Interrupt signal received. Exiting..."; cleanup_tasks; exit 1' SIGINT SIGTERM EXIT

# Main execution loop - continuously monitor and manage GPU tasks
while true; do
    # Get list of currently available GPUs
    available_gpus=($(get_available_gpus))

    # Assign tasks to available GPUs
    for gpu in "${available_gpus[@]}"; do
        is_gpu_used=false
        for task in "${running_tasks[@]}"; do
            IFS=':' read -r gpu_index _ _ <<< "$task"
            if [[ "$gpu_index" == "$gpu" ]]; then
                is_gpu_used=true
                break
            fi
        done
        if ! $is_gpu_used; then
            start_task "$gpu"
        fi
    done

    # Check task status and clean up completed tasks
    check_running_tasks

    # Exit when all tasks are completed
    if [ ! -s "$task_file" ] && [ ${#running_tasks[@]} -eq 0 ]; then
        echo "ALL TASKS FINISHED."
        break
    fi

    # Wait before next check cycle
    sleep 5
done

# Final cleanup on normal script termination
cleanup_tasks