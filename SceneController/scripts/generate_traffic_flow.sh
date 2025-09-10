#!/bin/bash
# SceneCrafter Traffic Flow Generation Script
#
# This script orchestrates the generation of traffic flow simulations across
# multiple scenes with configurable parallel processing capabilities.
#
# Features:
# - Configurable experiment naming with timestamp or custom names
# - Configurable number of simulations per scene
# - Parallel task execution with configurable concurrency
# - Comprehensive logging and error handling
# - Graceful cleanup on interruption
#
# Usage:
#   ./generate_traffic_flow.sh [experiment_name] [simulations_per_scene]
#
# Examples:
#   ./generate_traffic_flow.sh                    # Uses timestamp as experiment name
#   ./generate_traffic_flow.sh my_experiment      # Custom experiment name
#   ./generate_traffic_flow.sh my_exp 20          # Custom name with 20 simulations per scene

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Maximum number of concurrent simulation tasks
MAX_CONCURRENT_JOBS=10

# Base directories for the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOGS_DIR="$SCRIPT_DIR/logs"
TASKS_DIR="$SCRIPT_DIR/tasks"

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Determine experiment name based on input or timestamp
if [ -z "$1" ]; then
    # No experiment name provided - use timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXPERIMENT_NAME="traffic_gen_$TIMESTAMP"
else
    # Use provided experiment name, extract last two underscore-separated parts
    TIMESTAMP=$(echo "$1" | awk -F_ '{print $(NF-1)"_"$NF}')
    EXPERIMENT_NAME="traffic_gen_$TIMESTAMP"
fi

# Determine number of simulations per scene
if [ -z "$2" ]; then
    # Default simulations per scene
    SIMULATIONS_PER_SCENE=10
else
    SIMULATIONS_PER_SCENE=$2
fi

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

# Create necessary directories
mkdir -p "$LOGS_DIR/$TIMESTAMP"
mkdir -p "$TASKS_DIR"

# Configure output paths
TASK_FILE="$TASKS_DIR/traffic_flow_generation.txt"
LOG_PREFIX="$LOGS_DIR/$TIMESTAMP"

# =============================================================================
# TASK GENERATION
# =============================================================================

echo "=== SceneCrafter Traffic Flow Generation ==="
echo "Experiment: $EXPERIMENT_NAME"
echo "Simulations per scene: $SIMULATIONS_PER_SCENE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Generate task file with simulation commands
echo "Generating simulation tasks..."
python "$SCRIPT_DIR/tasks/traffic_flow_task_gen.py" \
    --exp_name "$EXPERIMENT_NAME" \
    --start_idx 0 \
    --count "$SIMULATIONS_PER_SCENE" \
    --output "$TASK_FILE"

# Brief pause to ensure task file is written
sleep 1

# =============================================================================
# TASK VALIDATION
# =============================================================================

# Verify task file was created successfully
if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: Task file '$TASK_FILE' was not created!"
    echo "Please check the traffic_flow_task_gen.py script for errors."
    exit 1
fi

# Count total tasks
TOTAL_TASKS=$(wc -l < "$TASK_FILE")
echo "Total tasks to execute: $TOTAL_TASKS"
echo ""

# =============================================================================
# PARALLEL EXECUTION SETUP
# =============================================================================

# Initialize task tracking variables
RUNNING_PIDS=()  # Array to store process IDs of running tasks
TASK_INDEX=1     # Current task index

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Function: Start a new simulation task
# Reads the next task from the task file and launches it in the background
start_task() {
    local command=$(awk "NR==$TASK_INDEX" "$TASK_FILE")
    
    if [ -z "$command" ]; then
        return  # No more tasks to process
    fi
    
    echo "[$(date '+%H:%M:%S')] Starting task $TASK_INDEX/$TOTAL_TASKS: $command"
    
    # Launch task in background with output redirection
    bash -c "$command" > "$LOG_PREFIX/task_output_${TASK_INDEX}.log" 2>&1 &
    local pid=$!
    
    RUNNING_PIDS+=("$pid")
    echo "  → PID: $pid"
}

# Function: Clean up all running tasks
# Called on script exit or interruption to ensure proper cleanup
cleanup_tasks() {
    echo ""
    echo "=== CLEANING UP ==="
    
    if [ ${#RUNNING_PIDS[@]} -gt 0 ]; then
        echo "Terminating ${#RUNNING_PIDS[@]} running tasks..."
        
        for pid in "${RUNNING_PIDS[@]}"; do
            if kill -0 "$pid" > /dev/null 2>&1; then
                echo "  → Terminating PID $pid"
                kill -9 "$pid" > /dev/null 2>&1
            fi
        done
        
        RUNNING_PIDS=()
    fi
    
    echo "Cleanup complete."
}

# =============================================================================
# SIGNAL HANDLING
# =============================================================================

# Set up signal handlers for graceful interruption
# Handles Ctrl+C and termination signals
trap 'echo ""; echo "INTERRUPT RECEIVED - initiating graceful shutdown..."; cleanup_tasks; exit 0' SIGINT SIGTERM

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================

echo "=== STARTING PARALLEL EXECUTION ==="
echo "Max concurrent jobs: $MAX_CONCURRENT_JOBS"
echo ""

# Main processing loop
while true; do
    # Launch new tasks until we reach max concurrency or run out of tasks
    while [ ${#RUNNING_PIDS[@]} -lt $MAX_CONCURRENT_JOBS ] && [ $TASK_INDEX -le $TOTAL_TASKS ]; do
        start_task
        ((TASK_INDEX++))
    done
    
    # Check if all tasks are complete
    if [ ${#RUNNING_PIDS[@]} -eq 0 ] && [ $TASK_INDEX -gt $TOTAL_TASKS ]; then
        echo ""
        echo "=== EXECUTION COMPLETE ==="
        echo "All $TOTAL_TASKS tasks have been successfully executed!"
        break
    fi
    
    # Clean up completed tasks
    ACTIVE_PIDS=()
    for pid in "${RUNNING_PIDS[@]}"; do
        if kill -0 "$pid" > /dev/null 2>&1; then
            ACTIVE_PIDS+=("$pid")
        fi
    done
    RUNNING_PIDS=("${ACTIVE_PIDS[@]}")
    
    # Display progress every 20 seconds
    CURRENT_TIME=$(date +%s)
    TIME_ELAPSED=$((CURRENT_TIME - LAST_PROGRESS_TIME))
    
    if [ $TIME_ELAPSED -ge 20 ]; then
        COMPLETED=$((TASK_INDEX - 1 - ${#RUNNING_PIDS[@]}))
        echo "[$(date '+%H:%M:%S')] Progress: $COMPLETED/$TOTAL_TASKS completed, ${#RUNNING_PIDS[@]} running"
        LAST_PROGRESS_TIME=$CURRENT_TIME
    fi
    
    # Shorter sleep for responsive task launching, but progress only every 20s
    sleep 2
done

# =============================================================================
# FINALIZATION
# =============================================================================

echo ""
echo "=== FINAL RESULTS ==="
echo "Experiment: $EXPERIMENT_NAME"
echo "Total simulations: $TOTAL_TASKS"
echo "Log directory: $LOG_PREFIX"
echo ""
echo "Traffic flow generation completed successfully!"

# Ensure cleanup is called on normal exit
cleanup_tasks