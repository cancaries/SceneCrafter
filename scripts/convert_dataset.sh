if [ -z "$1" ]; then
    now=$(date +"%Y%m%d_%H%M%S")
    exp_name="traffic_gen_$now"
else
    # now为输入字段被_分割的倒数两个字段
    now=$(echo $1 | awk -F_ '{print $(NF-1)"_"$NF}')
    exp_name="traffic_gen_$now"
fi

mkdir -p "./scripts/logs/$now"

# 任务文件路径
task_file="./scripts/tasks/convert_tasks_$exp_name.txt"

# 最大并发任务数
max_jobs=10

# 检查任务文件是否存在
if [ ! -f "$task_file" ]; then
    echo "Error: Task file '$task_file' does not exist!"
    exit 1
fi

# 当前正在运行的任务 PID 列表
running_pids=()

# 函数：启动一个新任务
start_task() {
    # 从任务文件中读取下一行任务
    local command=$(awk "NR==$task_index" "$task_file")
    if [ -z "$command" ]; then
        return  # 没有更多任务
    fi
    echo "Starting task: $command"
    bash -c "$command" > "./scripts/logs/$now/convert_output_$$.log" 2>&1 &  # 输出重定向到日志文件
    pid=$!
    running_pids+=("$pid")  # 记录 PID
}

# 函数：清理所有运行中的任务
cleanup_tasks() {
    echo "Cleaning up running tasks..."
    for pid in "${running_pids[@]}"; do
        if kill -0 $pid > /dev/null 2>&1; then
            echo "Terminating task with PID $pid"
            kill -9 $pid > /dev/null 2>&1  # 强制终止任务
        fi
    done
    running_pids=()  # 清空 PID 列表
}

# 捕获中断信号 (Ctrl+C) 和退出信号
trap 'echo "Interrupt signal received. Exiting..."; cleanup_tasks; exit 1' SIGINT SIGTERM EXIT

# 主循环
total_tasks=$(wc -l < "$task_file")  # 统计任务总数
task_index=1  # 初始化任务索引

while true; do
    # 启动新任务直到达到最大并发数或任务耗尽
    while [ ${#running_pids[@]} -lt $max_jobs ] && [ $task_index -le $total_tasks ]; do
        start_task
        ((task_index++))  # 移动到下一个任务
    done

    # 如果没有运行中的任务且所有任务已完成，则退出主循环
    if [ ${#running_pids[@]} -eq 0 ] && [ $task_index -gt $total_tasks ]; then
        echo "All tasks have been completed!"
        break
    fi

    # 检查并清理已完成的任务
    for pid in "${running_pids[@]}"; do
        if ! kill -0 $pid > /dev/null 2>&1; then
            # 如果进程已结束，从运行列表中移除其 PID
            running_pids=(${running_pids[@]/$pid})
        fi
    done

    # 等待一段时间后再次检查
    sleep 1
done

# 脚本正常退出时清理任务
cleanup_tasks