#!/usr/bin/env python3
"""
Task Generation Script for Dataset Conversion

This script generates task files containing commands for parallel dataset conversion processing.
It scans the render output directory to identify valid scene data and generates individual
conversion commands for each scene/generation combination.

Usage:
    python generate_convert_tasks.py --exp_name EXPERIMENT_NAME [options]

Example:
    python generate_convert_tasks.py --exp_name traffic_gen_20240101_120000 --output ./scripts/tasks/convert_tasks.txt
"""

import argparse
import os


def generate_tasks(render_output_path, sg_data_path, output_dataset_path, exp_name, output_file="tasks.txt"):
    """
    Generate a task file containing dataset conversion commands for parallel execution.
    
    This function scans the render output directory structure, identifies valid scene data,
    and creates individual conversion commands for each scene/generation index combination.

    Args:
        render_output_path (str): Path to the render output data directory
        exp_name (str): Experiment name used to identify trajectory folders
        output_file (str): Output task file path (default: tasks.txt)

    Directory Structure Expected:
        render_output_path/
            scene_folder_*/
                trajectory_{exp_name}/
                    scene_render_output_folder_*/
    """
    # Update output filename to include experiment name for unique identification
    output_file = output_file.replace(".txt", f"_{exp_name}.txt")
    
    # Scan render output directory for all scene folders
    scene_render_output_paths = os.listdir(render_output_path)
    scene_render_output_paths.sort()
    
    # Dictionary to store mapping of scene names to generation indices
    data_to_convert = {}
    
    # Process each scene folder
    for scene_folder_name in scene_render_output_paths:
        # Construct path to trajectory folder for this experiment
        scene_exp_folder = os.path.join(render_output_path, scene_folder_name, f"trajectory_{exp_name}")
        print(f"Processing: {scene_exp_folder}")
        
        # Skip if trajectory folder doesn't exist or isn't a directory
        if not os.path.exists(scene_exp_folder):
            continue
        if not os.path.isdir(scene_exp_folder):
            continue
            
        # Get all render output subdirectories within trajectory folder
        scenes_render_output_folders = os.listdir(scene_exp_folder)
        scenes_render_output_folders.sort()
        
        # Extract scene name from folder name (last part after underscore)
        scene_name = scene_folder_name.split('_')[-1]
        
        # Initialize scene entry if not exists
        if scene_name not in data_to_convert.keys():
            data_to_convert[scene_name] = []
        
        # Process each generation folder
        for scene_render_output_folder in scenes_render_output_folders:
            # Extract generation index from folder name
            gen_idx = scene_render_output_folder.split('_')[-1]
            
            # Skip if index is not a valid number
            if not gen_idx.isdigit():
                continue
                
            # Store valid generation index for this scene
            data_to_convert[scene_name].append(gen_idx)
    
    # Display collected data for verification
    print("Data to convert:")
    print(data_to_convert)
    
    # Generate conversion commands and write to output file
    with open(output_file, "w") as f:
        for scene_name, gen_idx_list in data_to_convert.items():
            for gen_idx in gen_idx_list:
                # Generate conversion command for each scene/gen_idx combination
                command = (f"python ./data_utils/render2dataset.py "
                          f"--exp_name {exp_name} "
                          f"--specific_scene {scene_name} "
                          f"--sg_data_path {sg_data_path} "
                          f"--render_path {render_output_path} "
                          f"--dataset_output_path {output_dataset_path}waymo_simulation_dataset_{exp_name}")
                f.write(command + "\n")
    
    print(f"Generated {sum(len(v) for v in data_to_convert.values())} tasks in {output_file}")


if __name__ == "__main__":
    # Configure command-line argument parser
    parser = argparse.ArgumentParser(
        description="Generate task file for parallel dataset conversion processing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Define command-line arguments
    parser.add_argument(
        "--sg_data_path", 
        type=str, 
        default='demo/waymo_dataset/',
        help="Path to waymo dataset directory (default: demo/waymo_dataset/)"
    )
    parser.add_argument(
        "--render_output_path", 
        type=str, 
        default='./SceneRenderer/street-gaussian/output/waymo_full_exp',
        help="Path to render output directory (default: ./SceneRenderer/street-gaussian/output/waymo_full_exp)"
    )
    parser.add_argument(
        "--output_dataset_path", 
        type=str, 
        default='demo/simulation_datasets/',
        help="Path to output dataset directory (default: demo/simulation_datasets/)"
    )
    parser.add_argument(
        "--exp_name", 
        type=str, 
        required=True,
        help="Experiment name used to identify trajectory folders"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./scripts/tasks/convert_tasks.txt",
        help="Output task file path (default: ./scripts/tasks/convert_tasks.txt)"
    )
    
    # Parse arguments and generate tasks
    args = parser.parse_args()
    generate_tasks(args.render_output_path, args.sg_data_path, args.output_dataset_path, args.exp_name, args.output)