import argparse
import os
def generate_tasks(render_output_path, exp_name, output_file="tasks.txt"):
    """
    Generate a tasks.txt file with commands for simulation.

    :param render_output_path: The path to the render output data
    :param exp_name: The experiment name (same for all tasks)
    :param output_file: The output file name (default: tasks.txt)
    """
    output_file = output_file.replace(".txt", f"_{exp_name}.txt")
    # search traffic_flow_path for all experiments
    scene_render_output_paths = os.listdir(render_output_path)
    scene_render_output_paths.sort()
    data_to_convert = {}
    #print(scene_render_output_paths)
    for scene_folder_name in scene_render_output_paths:
        scene_exp_folder = os.path.join(render_output_path, scene_folder_name, f"trajectory_{exp_name}")
        print(scene_exp_folder)
        if not os.path.exists(scene_exp_folder):
            continue
        if not os.path.isdir(scene_exp_folder):
            continue
        scenes_render_output_folders = os.listdir(scene_exp_folder)
        scenes_render_output_folders.sort()
        scene_name = scene_folder_name.split('_')[-1]
        if scene_name not in data_to_convert.keys():
            data_to_convert[scene_name] = []
        for scene_render_output_folder in scenes_render_output_folders:
            gen_idx = scene_render_output_folder.split('_')[-1]
            if not gen_idx.isdigit():
                continue
            data_to_convert[scene_name].append(gen_idx)
    print(data_to_convert)
    # write to output file
    with open(output_file, "w") as f:
        for scene_name, gen_idx_list in data_to_convert.items():
            for gen_idx in gen_idx_list:
                f.write(f"python ./data_utils/render2dataset.py --exp_name {exp_name} --specific_scene {scene_name} --sg_data_path /GPFS/public/junhaoge/data/SceneCrafter/dataset/waymo_dataset/ --dataset_output_path /GPFS/public/junhaoge/data/SceneCrafter/simulation_datasets/waymo_simulation_dataset_{exp_name}\n")
                # f.write(f"python ./data_utils/render2dataset.py --exp_name {exp_name} --specific_scene {scene_name} --sg_data_path /root/DATA2/yifanjiang/data --dataset_output_path /root/DATA2/junhaoge/sim2real/simulation_datasets/waymo_simulation_dataset_{exp_name}\n")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate tasks.txt for simulation.")
    parser.add_argument("--render_output_path", type=str, default='./SceneRenderer/street-gaussian/output/waymo_full_exp')
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--output", type=str, default="./scripts/tasks/convert_tasks.txt", help="Output file name (default: tasks.txt)")
    
    args = parser.parse_args()

    # Generate tasks
    generate_tasks(args.render_output_path, args.exp_name, args.output)