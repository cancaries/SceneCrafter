#!/bin/bash
# Dataset Conversion Script
# This script manages parallel execution of dataset conversion tasks with configurable concurrency.
# It reads commands from a task file and executes them in parallel while respecting the maximum job limit.
# 
# Usage: ./convert_dataset.sh [optional_experiment_name]
# - If no argument provided, uses current timestamp as experiment name
# - If argument provided, extracts last two underscore-separated fields as timestamp

# Generate experiment name based on input parameter or current timestamp
if [ -z "$1" ]; then
    now=$(date +"%Y%m%d_%H%M%S")
    exp_name="traffic_gen_$now"
else
    # Extract the last two underscore-separated fields as timestamp from input parameter
    now=$(echo $1 | awk -F_ '{print $(NF-1)"_"$NF}')
    exp_name="traffic_gen_$now"
fi

# Create log directory for this experiment
mkdir -p "./scripts/logs/$now"

# Path to the task file containing conversion commands
task_file="./scripts/tasks/convert_tasks_$exp_name.txt"

# Maximum number of concurrent jobs to run
max_jobs=10

# Check if the task file exists before proceeding
if [ ! -f "$task_file" ]; then
    echo "Error: Task file '$task_file' does not exist!"
    exit 1
fi

# Array to store PIDs of currently running tasks
running_pids=()

# Function: Start a new task from the task file
start_task() {
    # Read the next command from the task file based on current task index
    local command=$(awk "NR==$task_index" "$task_file")
    if [ -z "$command" ]; then
        return  # No more tasks to process
    fi
    echo "Starting task: $command"
    # Execute the command in background with output redirected to log file
    bash -c "$command" > "./scripts/logs/$now/convert_output_$$.log" 2>&1 &
    pid=$!
    running_pids+=("$pid")  # Store the PID for monitoring
}

# Function: Clean up all running tasks
# This is called when script exits or receives termination signals
cleanup_tasks() {
    echo "Cleaning up running tasks..."
    for pid in "${running_pids[@]}"; do
        if kill -0 $pid > /dev/null 2>&1; then
            echo "Terminating task with PID $pid"
            kill -9 $pid > /dev/null 2>&1  # Force terminate the task
        fi
    done
    running_pids=()  # Clear the PID list
}

# Set up signal handlers for graceful shutdown
# This ensures cleanup when user presses Ctrl+C or system sends termination signals
trap 'echo "Interrupt signal received. Exiting..."; cleanup_tasks; exit 1' SIGINT SIGTERM EXIT

# Main execution loop
# Count total number of tasks in the task file
total_tasks=$(wc -l < "$task_file")
task_index=1  # Initialize task counter

while true; do
    # Launch new tasks until max concurrency is reached or all tasks are scheduled
    while [ ${#running_pids[@]} -lt $max_jobs ] && [ $task_index -le $total_tasks ]; do
        start_task
        ((task_index++))  # Move to the next task
    done

    # Exit when all tasks are completed
    if [ ${#running_pids[@]} -eq 0 ] && [ $task_index -gt $total_tasks ]; then
        echo "All tasks have been completed!"
        break
    fi

    # Check for completed tasks and remove their PIDs from monitoring list
    for pid in "${running_pids[@]}"; do
        if ! kill -0 $pid > /dev/null 2>&1; then
            # Process has finished, remove PID from tracking array
            running_pids=(${running_pids[@]/$pid})
        fi
    done

    # Brief pause before next status check
    sleep 1
done

# Final cleanup on normal script exit
cleanup_tasks