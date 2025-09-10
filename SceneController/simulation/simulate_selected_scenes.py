#!/usr/bin/env python3
"""
Scene Simulation Engine for SceneCrafter

This module provides a comprehensive simulation framework for generating traffic scenes
using the SceneController system. It processes Waymo scene data to create realistic
traffic simulations with configurable parameters.
"""

import sys
import os

# Add project root to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import time
import random
import yaml
import json
import argparse

import numpy as np
from tqdm import tqdm
from PIL import Image
from SceneController.scene import Scene
from SceneController.agents.navigation.waypoint import Transform
from visualization.video_utils import create_video_from_images


def main():
    parser = argparse.ArgumentParser(description='SceneCrafter Traffic Simulation Engine')
    parser.add_argument(
        '--scene_setting_config', 
        type=str, 
        default='./SceneController/config/agent_config/example_datagen.yaml',
        help='YAML configuration file for scene settings and agent behaviors'
    )
    parser.add_argument(
        '--scene_list', 
        type=str, 
        default='./SceneController/config/scene_config/selected_scenes.yaml',
        help='YAML file listing scenes to process'
    )
    parser.add_argument(
        '--exp_name', 
        type=str, 
        default=None, 
        help='Experiment name for output organization. Auto-generated if not provided'
    )
    parser.add_argument(
        '--simulate_num', 
        type=int, 
        default=1, 
        help='Number of simulation runs per scene'
    )
    parser.add_argument(
        '--traffic_video', 
        action='store_true', 
        help='Generate traffic flow visualization videos'
    )
    parser.add_argument(
        '--video_save_fps', 
        type=int, 
        default=10, 
        help='Frame rate for saved videos'
    )
    parser.add_argument(
        '--gen_start_idx', 
        type=int, 
        default=0, 
        help='Starting index for generation numbering'
    )
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=None, 
        help='Base random seed for reproducibility'
    )
    parser.add_argument(
        '--map_mode', 
        type=str, 
        default='model', 
        choices=['original', 'model'],
        help='Map processing mode: "original" uses raw maps, "model" uses ground-refined maps'
    )

    args = parser.parse_args()
    data_type = "demo" # Options: "demo" or "data"
    scene_setting_config = yaml.load(open(args.scene_setting_config, 'r'), Loader=yaml.FullLoader)
    scene_file = yaml.load(open(args.scene_list, 'r'), Loader=yaml.FullLoader)
    
    # Build comprehensive scene list from includes
    scene_list = []
    if len(scene_file['include_scene_files']) > 0:
        print('Include scene files: ', scene_file['include_scene_files'])
        config_root = os.path.dirname(args.scene_list)
        for file_name in scene_file['include_scene_files']:
            file_name += '.yaml'
            file = os.path.join(config_root, file_name)
            scene_list += yaml.load(open(file, 'r'), Loader=yaml.FullLoader)['include_scene_idx']
    
    # Add direct scene indices
    scene_list += scene_file['include_scene_idx']
    
    if args.exp_name is None:
        exp_name = 'normal_sim'
    else:
        exp_name = args.exp_name
    
    simulate_num = args.simulate_num
    save_traffic_video = args.traffic_video
    video_save_fps = args.video_save_fps
    random_seed = args.random_seed
    gen_idx = args.gen_start_idx
    
    scene_setting_config['simulation']['FPS'] = 10
    skip_frames = int(scene_setting_config['simulation']['FPS'] / video_save_fps)
    
    # Configure data paths
    data_path = os.path.join(project_root, f'{data_type}/waymo_scene_data')
    
    # Main simulation loop
    for _ in range(simulate_num):
        # Generate unique random seed for this simulation
        if random_seed is not None:
            scene_setting_config['simulation']['random_seed'] = random_seed
            random.seed(scene_setting_config['simulation']['random_seed'])
        else:
            scene_setting_config['simulation']['random_seed'] = random.randint(0, 10000)
            random.seed(scene_setting_config['simulation']['random_seed'])
        
        # Configure output directories
        save_folder_name = f'{exp_name}/{gen_idx}'
        scenes_folder = os.path.join(project_root, f'{data_type}/traffic_flow/{save_folder_name}')
        
        # Create output directory structure
        gen_idx += 1
        os.makedirs(scenes_folder, exist_ok=True)
        
        # Configure video output directory
        video_path = os.path.join(scenes_folder, 'video')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        
        # Save configuration for reproducibility
        with open(os.path.join(scenes_folder, 'config.yaml'), 'w') as f:
            yaml.dump(scene_setting_config, f)
        
        # Process scenes in sorted order
        data_path_sort = os.listdir(data_path)
        data_path_sort.sort()
        
        for scene_name in data_path_sort:
            # Skip non-numeric scene directories
            if not scene_name.isdigit():
                continue
            
            # Skip scenes not in the processing list
            if scene_name not in scene_list:
                continue
                
            print('Simulating scene: ', scene_name)
            
            # Configure scene-specific paths
            _data_root = os.path.join(data_path, scene_name)
            save_dir = os.path.join(scenes_folder, scene_name)
            
            # Clean existing output
            if os.path.exists(save_dir):
                os.system(f'rm -r {save_dir}')
            
            # Configure map data paths based on mode
            if args.map_mode == 'original':
                map_config_path = os.path.join(_data_root, 'map_feature.json')
            elif args.map_mode == 'model':
                map_config_path = os.path.join(_data_root, 'map_feature_w_model.json')
            
            # Configure additional data paths
            ego_pose_path = os.path.join(_data_root, 'ego_pose.json')
            static_actor_config_path = os.path.join(_data_root, 'all_static_actor_data.json')
            
            # Build scene configuration dictionary
            scene_config = dict()
            scene_config['_data_root'] = _data_root
            scene_config['_save_dir'] = save_dir
            scene_config['_available_asset_dir'] = scene_setting_config['data']['available_asset_dir']
            scene_config['_map_config_path'] = map_config_path
            scene_config['_ego_pose_path'] = ego_pose_path
            scene_config['_static_actor_config_path'] = static_actor_config_path
            scene_config['_scene_name'] = _data_root.split('/')[-1]
            scene_config['_FPS'] = scene_setting_config['simulation']['FPS']
            scene_config['mode'] = scene_setting_config['simulation']['mode']
            scene_config['ego_vehicle_config'] = scene_setting_config['ego_vehicle']
            scene_config['other_agents_config'] = scene_setting_config['other_agents']
            scene_config['agent_spawn_config'] = scene_setting_config['simulation']['agent_spawn']
            
            # Initialize scene simulation
            Scene.initialize_scene(scene_config)
            
            # Configure simulation parameters
            scene_length = scene_setting_config['simulation']['max_steps']
            stay_time_steps = scene_setting_config['simulation']['ego_control_steps']
            
            # Run simulation for this scene
            for i in tqdm(range(scene_length), desc=f'Simulating {scene_name}'):
                # Check for early termination
                if Scene._end_scene:
                    break
                
                # Process all agents
                for agent_name, agent in Scene._agent_dict.items():
                    agent_control = agent.run_step()
                    Scene._agent_control_dict[agent_name] = agent_control
                
                # Control ego vehicle behavior
                if i < stay_time_steps:
                    # Initial phase: controlled movement
                    Scene.run_step_w_ego(i)
                else:
                    # Later phase: autonomous behavior
                    Scene.run_step_w_ego(i, True)
            
            # Generate traffic flow visualizations
            if Scene._mode == 'debug':
                # Debug mode: show all traffic
                pic_save_path = os.path.join(scenes_folder, scene_name, 'traffic_pic')
                Scene._map.draw_map_w_traffic_flow_sequence(
                    Scene._car_dict_sequence,
                    save_path=pic_save_path,
                    skip_frames=skip_frames
                )
            elif Scene._mode == 'datagen':
                # Data generation mode: show only ego vehicle
                pic_save_path = os.path.join(scenes_folder, scene_name, 'traffic_pic')
                Scene._map.draw_map_w_traffic_flow_sequence(
                    Scene._car_dict_sequence,
                    save_path=pic_save_path,
                    skip_frames=skip_frames,
                    only_ego=True
                )
                
                # Save traffic flow data 
                Scene.save_traffic_flow()
                
                # Generate video if requested
                if save_traffic_video and scene_name != 'video' and not os.path.isfile(os.path.join(scenes_folder, scene_name)):
                    scene_img_folder = os.path.join(scenes_folder, scene_name, 'traffic_pic')
                    output_path = os.path.join(video_path, f'{scene_name}.mp4')
                    
                    # Create video from simulation frames
                    valid = create_video_from_images(
                        scene_img_folder, 
                        output_path, 
                        int(scene_setting_config['simulation']['FPS'] / skip_frames)
                    )
            
            # Reset scene for next iteration
            Scene.reset()
                
                
if __name__ == '__main__':
    main()