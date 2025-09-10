"""
Traffic Simulation Data to Rendering Format Converter

This script converts traffic simulation data into a format suitable for rendering
in the SceneCrafter framework. It processes agent trajectories, map features,
and static actor data to prepare them for scene generation and visualization.
"""

import os
import json
import argparse

def main():
    # Configure command line argument parser
    parser = argparse.ArgumentParser(description='Convert traffic simulation data to rendering format')
    parser.add_argument("--sg_data_path", default="demo/waymo_dataset/", type=str,
                        help='Path to scene data directory (output)')
    parser.add_argument("--agent_data_path", default="demo/traffic_flow/", type=str,
                        help='Path to agent trajectory simulation data (input)')
    parser.add_argument("--waymo_scene_data_path", default="demo/waymo_scene_data/", type=str,
                        help='Path to Waymo data used for traffic generation')
    parser.add_argument("--exp_name", required=True, type=str,
                        help='Experiment name to filter processing (required)')
    parser.add_argument("--specific_scene", default=['019'], type=list,
                        help='List of specific scene IDs to process (default: ["019"])')
    parser.add_argument("--all_scene", action='store_true',
                        help='Process all available scenes instead of filtering')
    parser.add_argument("--text", default=None, type=str,
                        help='Optional text annotation for experiment naming')
    
    args = parser.parse_args()
    sg_data_path = args.sg_data_path
    agent_data_path = args.agent_data_path
    waymo_scene_data_path = args.waymo_scene_data_path
    specific_scene = args.specific_scene
    all_scene = args.all_scene
    text = args.text
    exp_name = args.exp_name
    
    # Process each experiment time directory
    for exp_time in os.listdir(agent_data_path):
        # Skip non-directory entries
        if not os.path.isdir(os.path.join(agent_data_path, exp_time)):
            continue
        
        # Filter by experiment name
        if exp_name not in exp_time:
            continue
        
        # Get experiment indices for this time period
        exp = os.listdir(os.path.join(agent_data_path, exp_time))
        print(f"Processing experiment time: {os.path.join(agent_data_path, exp_time)}")
        exp.sort()
        
        # Process each experiment index
        for exp_idx in exp:
            # Skip non-directory entries
            if not os.path.isdir(os.path.join(agent_data_path, exp_time, exp_idx)):
                continue
            
            print(f"Processing experiment index: {os.path.join(agent_data_path, exp_time, exp_idx)}")
            
            # Process each scene within experiment
            for scene_name in os.listdir(os.path.join(agent_data_path, exp_time, exp_idx)):
                # Skip non-numeric scene names
                if not scene_name.isdigit():
                    continue
                
                # Skip non-directory entries
                if not os.path.isdir(os.path.join(agent_data_path, exp_time, exp_idx, scene_name)):
                    continue
                
                # Process scene if matches criteria
                if all_scene or scene_name in specific_scene:
                    # Define source and destination paths
                    scene_path = os.path.join(agent_data_path, exp_time, exp_idx, scene_name)
                    scene_data_path = os.path.join(sg_data_path, scene_name)
                    
                    # Create destination directories
                    os.makedirs(scene_data_path, exist_ok=True)
                    car_info_path = os.path.join(sg_data_path, scene_name, 'car_info')
                    os.makedirs(car_info_path, exist_ok=True)
                    
                    # Define file paths for copying
                    map_feature_path = os.path.join(car_info_path, 'map_feature.json')
                    ego_pose_ori_path = os.path.join(car_info_path, 'ego_pose_ori.json')
                    
                    # Construct car dictionary sequence filename with optional text annotation
                    if text is not None:
                        car_dict_sequence_path = os.path.join(car_info_path, 
                            f'car_dict_sequence_{str(int(exp_idx)).zfill(3)}_{text}.json')
                    else:
                        car_dict_sequence_path = os.path.join(car_info_path, 
                            f'car_dict_sequence_{str(int(exp_idx)).zfill(3)}.json')
                    
                    # Remove existing file if no backup exists (prevent conflicts)
                    if os.path.exists(car_dict_sequence_path):
                        backup_path = f'{car_dict_sequence_path.split(".")[0]}_before_{exp_time}.json'
                        if not os.path.exists(backup_path):
                            os.system(f'rm {car_dict_sequence_path}')
                    
                    # Define static actor data path
                    static_actor_path = os.path.join(car_info_path, 'all_static_actor_data.json')
                    
                    # Copy essential data files to rendering format
                    os.system(f'cp {scene_path}/car_info_dict.json {car_dict_sequence_path}')
                    os.system(f'cp {scene_path}/map_feature.json {map_feature_path}')
                    os.system(f'cp {waymo_scene_data_path}/{scene_name}/ego_pose.json {ego_pose_ori_path}')
                    os.system(f'cp {waymo_scene_data_path}/{scene_name}/all_static_actor_data.json {static_actor_path}')


if __name__ == '__main__':
    main()