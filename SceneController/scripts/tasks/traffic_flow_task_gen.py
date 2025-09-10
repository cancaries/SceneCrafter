import argparse

def generate_tasks(scene_list, exp_name, start_idx, count, output_file="tasks.txt"):
    """
    Generate a tasks.txt file with commands for simulation.

    :param exp_name: The experiment name (same for all tasks)
    :param start_idx: The starting value of gen_start_idx
    :param count: The number of tasks to generate
    :param output_file: The output file name (default: tasks.txt)
    """
    with open(output_file, "w") as f:
        for i in range(count):
            gen_start_idx = start_idx + i
            command = (
                f"python ./SceneController/simulation/simulate_selected_scenes.py "
                f"--scene_list {scene_list} --exp_name {exp_name} --simulate_num 1 --gen_start_idx {gen_start_idx}\n"
            )
            f.write(command)
            print(command)
    print(f"Generated {count} tasks in '{output_file}'.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate tasks.txt for simulation.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--start_idx", type=int, required=True, help="Starting index for gen_start_idx")
    parser.add_argument("--count", type=int, required=True, help="Number of tasks to generate")
    parser.add_argument("--output", type=str, default="./SceneController/scripts/tasks/traffic_flow_generation.txt", help="Output file name (default: tasks.txt)")
    parser.add_argument("--scene_list", type=str, default="./SceneController/config/scene_config/selected_scenes.yaml")
    
    args = parser.parse_args()

    # Generate tasks
    generate_tasks(args.scene_list, args.exp_name, args.start_idx, args.count, args.output)