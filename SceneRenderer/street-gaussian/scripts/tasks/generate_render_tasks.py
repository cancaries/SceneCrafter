import argparse
import os
def generate_tasks(traffic_flow_path, exp_name, output_file="tasks.txt", ply_path=""):
    """
    Generate a tasks.txt file with commands for simulation.

    Args:
        traffic_flow_path (str): The path to the traffic flow data
        exp_name (str): The experiment name (same for all tasks)
        output_file (str, optional): The output file name (default: tasks.txt)
        ply_path (str, optional): The path to the ply models (default: "")
    Returns:
        None
    """
    # search traffic_flow_path for all experiments
    exp_path = os.path.join(traffic_flow_path, exp_name)
    all_gen = os.listdir(exp_path)
    all_gen.sort()
    data_to_gen = {}
    for gen_idx in all_gen:
        if not os.path.isdir(os.path.join(exp_path, gen_idx)):
            continue
        scenes_names = os.listdir(os.path.join(exp_path, gen_idx))
        scenes_names.sort()
        for scene_name in scenes_names:
            if not scene_name.isdigit():
                continue
            if not os.path.isdir(os.path.join(exp_path, gen_idx, scene_name)):
                continue
            if scene_name not in data_to_gen.keys():
                data_to_gen[scene_name] = []
            data_to_gen[scene_name].append(gen_idx)

    # write to output file
    with open(output_file, "w") as f:
        for scene_name, gen_idx_list in data_to_gen.items():
            for gen_idx in gen_idx_list:
                f.write(f"python render_waymo.py --config ./configs/example/waymo_render_{scene_name}.yaml --mode novel --render_name {exp_name} --scene_number {gen_idx} --ply_model_path {ply_path}\n")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate tasks.txt for simulation.")
    parser.add_argument("--traffic_flow_path", type=str, default='../../demo/traffic_flow')
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--output", type=str, default="./scripts/tasks/render_tasks.txt", help="Output file name (default: tasks.txt)")
    parser.add_argument("--ply_path", type=str, default="../../demo/agent_ply/model_w_shadow", help="Path to the ply models")
    
    args = parser.parse_args()

    # Generate tasks
    generate_tasks(args.traffic_flow_path, args.exp_name, args.output, args.ply_path)