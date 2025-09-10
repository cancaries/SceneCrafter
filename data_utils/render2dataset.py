"""
Render Output to Dataset Converter for SceneCrafter

This module converts rendered simulation outputs into structured datasets for 
autonomous driving research and scene understanding tasks. It processes multi-view
rendered images, depth maps, and vehicle/agent information to create organized
datasets compatible with standard perception frameworks.

Key Features:
- Multi-camera processing (5 camera setup)
- 3D object projection and validation using depth maps
- Automatic filtering of occluded objects
- Support for both specific scene processing and batch processing
- Integration with Waymo dataset format
- Timestamp-based frame organization
- Dynamic object visibility calculation across cameras

Input Structure:
- Rendered images from multiple viewpoints
- Depth maps for occlusion testing
- Agent trajectory data (JSON format)
- Camera calibration parameters (intrinsic/extrinsic)

Output Structure:
- Organized frame-by-frame dataset
- Validated 3D bounding boxes with visibility information
- Camera calibration data per frame
- Ego vehicle poses and agent information
"""

import os
import re
import imageio
import copy
import time
import subprocess
import json
import argparse
import math
import cv2
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d
from PIL import Image
from cal_utils import calculate_distance, rpy2R, bbox_to_corner3d, disassemble_matrix
from graphics_utils import project_numpy, project_label_to_image
from waymo_utils import load_extrinsics, load_intrinsics

# Coordinate system transformation: OpenCV camera to standard camera system
# This matrix transforms points from OpenCV's camera coordinate system (Z-forward, Y-down, X-right)
# to the standard camera system used in computer graphics
opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])


def get_ego_vehicle_matrices(json_file, ego_poses_ori_json=None):
    """
    Generate 4x4 transformation matrices for ego vehicle from trajectory data.
    
    This function processes ego vehicle trajectory data to create smooth transformation
    matrices for each frame. It applies Gaussian filtering to reduce noise in the
    z-coordinate (height) and retrieves accurate roll/pitch angles from reference data.
    
    Args:
        json_file (str): Path to JSON file containing ego vehicle trajectory data
        ego_poses_ori_json (str, optional): Path to original Waymo ego poses for roll/pitch reference
        
    Returns:
        list: List of 4x4 transformation matrices (vehicle-to-world) for each frame
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    matrices = []
    ego_loc = []
    ego_yaw = []
    
    # Extract ego vehicle positions and yaw angles from trajectory data
    for item in data:
        if 'ego_vehicle' in item:
            ego_vehicle = item['ego_vehicle']
            loc = np.array(ego_vehicle['loc']) 
            ego_loc.append(loc)
            theta = ego_vehicle['rot'][2]  # Yaw angle
            ego_yaw.append(theta)

    # Apply Gaussian filtering to smooth z-coordinate (height) variations
    ego_z = [loc[2] for loc in ego_loc]
    sigma = 2.0
    filtered_z = gaussian_filter1d(ego_z, sigma=sigma)
    ego_loc_new = []

    for idx, loc_ori in enumerate(ego_loc):
        loc = loc_ori.copy()
        loc[2] = filtered_z[idx]  # Replace with smoothed height
        ego_loc_new.append(loc)

    # Load reference poses for accurate roll/pitch angles if available
    ego_poses_ref = []
    ego_poses_indices = []
    if ego_poses_ori_json is not None:
        with open(ego_poses_ori_json, 'r') as f:
            ego_poses_ori = json.load(f)
        for idx, item in enumerate(ego_poses_ori):
            loc = item['location']
            rot = item['rotation']
            ego_poses_ref.append(loc)
            ego_poses_indices.append(rot)
        kd_tree_rot = KDTree(ego_poses_ref)

    # Query closest reference pose for roll angles
    rolls = []
    for idx, loc in enumerate(ego_loc_new):
        roll = ego_poses_indices[kd_tree_rot.query(ego_loc_new[idx])[1]][0]
        rolls.append(roll)

    # Query closest reference pose for pitch angles
    pitchs = []
    for idx, loc in enumerate(ego_loc_new):
        pitch = ego_poses_indices[kd_tree_rot.query(ego_loc_new[idx])[1]][1]
        pitchs.append(pitch)
    
    # Apply additional smoothing to roll and pitch angles
    sigma = 0.5
    rolls = gaussian_filter1d(rolls, sigma=sigma)
    pitchs = gaussian_filter1d(pitchs, sigma=sigma)
    
    # Construct final transformation matrices
    for idx, loc in enumerate(ego_loc_new):
        theta = ego_yaw[idx]
        roll = rolls[idx]
        pitch = pitchs[idx]
        R_z = rpy2R([roll, pitch, theta])
        T_vehicle_to_world = np.eye(4)
        T_vehicle_to_world[:3, :3] = R_z  # Rotation matrix
        T_vehicle_to_world[:3, 3] = loc  # Translation vector
        T_vehicle_to_world[3, 3] = 1
        matrices.append(T_vehicle_to_world)

    return matrices

def get_agent_vehicle_matrices(json_file):
    """
    Extract and process agent vehicle information from trajectory data.
    
    This function processes agent vehicle data (non-ego vehicles, pedestrians, etc.)
    and adjusts their positions to account for bounding box center alignment.
    
    Args:
        json_file (str): Path to JSON file containing agent vehicle trajectory data
        
    Returns:
        list: List of dictionaries, each containing agent vehicle information for one frame
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    agent_vehicle_info = []
    for item in data:
        tmp_agent_vehcile_info = dict()
        for key, vehicle_info in item.items():
            if key == 'ego_vehicle':
                continue  # Skip ego vehicle as it's handled separately
            
            loc = np.array(vehicle_info['loc'])
            # Adjust z-coordinate for dynamic objects to account for bounding box center
            if not 'static' in key:
                loc[2] = loc[2] + vehicle_info['bbox'][2] / 2
            
            tmp_agent_vehcile_info[key] = vehicle_info
            tmp_agent_vehcile_info[key]['loc'] = loc

        agent_vehicle_info.append(tmp_agent_vehcile_info)

    return agent_vehicle_info

def main():
    parser = argparse.ArgumentParser(description='Convert rendered outputs to structured dataset')
    parser.add_argument("--sg_data_path", default="data/SceneCrafter/dataset/waymo_dataset/", 
                        type=str, help='Path to scene graph data directory')
    parser.add_argument("--render_path", default="data/SceneCrafter/sg_output", 
                        type=str, help='Path to rendered output directory')
    parser.add_argument("--dataset_output_path", 
                        default='data/SceneCrafter/simulation_datasets/waymo_simulation_dataset', 
                        type=str, help='Output path for generated dataset')
    parser.add_argument("--exp_name", default='trajectory', type=str, 
                        help='Experiment name for trajectory identification')
    parser.add_argument("--specific_scene", nargs='+', default=['019'], 
                        help='List of specific scene IDs to process')
    parser.add_argument("--specific_scene_idx", nargs='+', default=[], 
                        help='List of specific experiment indices to process')
    parser.add_argument("--all_scene", default=False, type=bool, 
                        help='Process all available scenes')
    parser.add_argument("--gpu", default="0", type=str, 
                        help="GPU device ID to use for processing")
    
    args = parser.parse_args()

    # Set GPU device for processing
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Extract configuration parameters
    sg_data_path = args.sg_data_path
    render_path = args.render_path
    dataset_output_path = args.dataset_output_path
    specific_scene = args.specific_scene
    all_scene = args.all_scene
    specific_scene_idx = args.specific_scene_idx
    camera_ids = [0, 1, 2, 3, 4]  # Multi-camera setup
    text = args.exp_name
    trajectory_name = f'trajectory_{text}'
    
    print('Processing scenes:', specific_scene)

    # Process each render output directory
    for render_output_path in os.listdir(render_path):
        scene_name = render_output_path.split('_')[-1]
        print('Scene name:', scene_name)
        
        # Skip non-directory entries
        if not os.path.isdir(os.path.join(args.render_path, render_output_path)):
            print(f'{os.path.join(args.render_path, render_output_path)} is not a directory, skip')
            continue
        
        # Filter for Waymo scenes
        if 'waymo' not in render_output_path:
            continue
        
        # Process scene if matches criteria
        if all_scene or scene_name in specific_scene:
            dataset_output_path_scene = os.path.join(dataset_output_path, scene_name)
            os.makedirs(dataset_output_path_scene, exist_ok=True)
            
            # Setup paths for source data
            sg_data_path_scene = os.path.join(args.sg_data_path, scene_name)
            sg_data_path_scene_car_info = os.path.join(sg_data_path_scene, 'car_info')
            map_feature_path = os.path.join(sg_data_path_scene_car_info, 'map_feature.json')
            
            # Load camera calibration data
            ext_path = os.path.join(sg_data_path_scene, "extrinsics")
            int_path = os.path.join(sg_data_path_scene, "intrinsics")
            camera_extrinsics = []
            camera_intrinsics = []
            
            for cam_id in camera_ids:
                with open(os.path.join(ext_path, f'{cam_id}.txt'), 'r') as f:
                    data = np.loadtxt(f)
                    camera_extrinsics.append(data)
                with open(os.path.join(int_path, f'{cam_id}.txt'), 'r') as f:
                    data = load_intrinsics(f)
                    camera_intrinsics.append(data)
            
            # Process trajectory experiments
            scene_data_path = os.path.join(args.render_path, render_output_path, trajectory_name)
            scene_data_path_list = os.listdir(scene_data_path)
            scene_data_path_list.sort()
            
            for scene_exp in scene_data_path_list:
                # Filter by specific experiment indices if provided
                if len(specific_scene_idx) > 0:
                    if str(scene_exp.split('_')[-1]) not in specific_scene_idx:
                        print(f'{scene_exp} is not in specific_scene_idx, skip')
                        continue
                
                # Skip non-directory entries
                if not os.path.isdir(os.path.join(scene_data_path, scene_exp)):
                    continue
                
                print(f'Processing scene {scene_name} experiment {scene_exp}')
                start_time = time.time()
                cur_time = start_time
                
                scene_exp_path = os.path.join(scene_data_path, scene_exp)
                
                # Collect valid frame indices from rendered outputs
                valid_frame_idx = []
                for file_name in os.listdir(scene_exp_path):
                    if file_name.endswith('rgb.png'):
                        valid_frame_idx.append(int(file_name.split('_')[0]))
                valid_frame_idx = sorted(valid_frame_idx)
                
                # Remove duplicate frame indices
                valid_frame_idx = list(set(valid_frame_idx))
                unique_frame_list = []
                [unique_frame_list.append(x) for x in valid_frame_idx if x not in unique_frame_list]
                unique_frame_list.sort()
                
                # Setup output directory for this experiment
                scene_idx = scene_exp.split('_')[-1]
                scene_save_path = os.path.join(dataset_output_path_scene, scene_idx)
                
                if os.path.exists(scene_save_path):
                    print(f'{scene_save_path} already exists, skip')
                    continue
                
                os.makedirs(scene_save_path, exist_ok=True)
                
                # Copy essential data files
                map_feature_save_path = os.path.join(scene_save_path, 'map_feature.json')
                os.system(f'cp {map_feature_path} {map_feature_save_path}')
                
                # Load and process agent trajectory data
                car_dict_sequence_path = os.path.join(sg_data_path_scene_car_info, 
                                                   f'car_dict_sequence_{scene_idx}.json')
                car_dict_sequence_save_path = os.path.join(scene_save_path, 'car_dict_sequence.json')
                
                with open(car_dict_sequence_path, 'r') as f:
                    car_dict_sequence_ori_data = json.load(f)
                
                # Normalize vehicle types and adjust positions
                for idx, item in enumerate(car_dict_sequence_ori_data):
                    for key, car_info in item.items():
                        # Adjust z-coordinate for dynamic objects
                        if not 'static' in key and not 'ego' in key:
                            car_info['loc'][2] = car_info['loc'][2] + car_info['bbox'][2] / 2
                        
                        # Normalize vehicle type classifications
                        valid_types = ['pedestrian', 'vehicle', 'sign', 'traffic_light', 
                                     'bicyle', 'motorcycle', 'bike', 'CYCLIST', 
                                     'PEDESTRIAN', 'VEHICLE', 'cyclist', 'misc']
                        if car_info['type'] not in valid_types:
                            car_info['type'] = 'vehicle'
                
                os.system(f'cp {car_dict_sequence_path} {car_dict_sequence_save_path}')
                
                # Load camera calibration
                intrinsic_path = os.path.join(sg_data_path_scene, 'intrinsics', '0.txt')
                camera_intrinsic = np.loadtxt(intrinsic_path)
                
                # Generate ego vehicle poses and agent information
                ego_pose_ori_path = os.path.join(sg_data_path_scene_car_info, 'ego_pose_ori.json')
                ego_poses = get_ego_vehicle_matrices(car_dict_sequence_save_path, ego_pose_ori_path)
                agent_car_info = get_agent_vehicle_matrices(car_dict_sequence_save_path)
                
                # Process each frame
                for idx, frame_idx in enumerate(unique_frame_list):
                    cur_timestamp = cur_time + 5000
                    save_timestamp = round(cur_timestamp * 1000)
                    
                    frame_save_path = os.path.join(scene_save_path, f'{idx}')
                    os.makedirs(frame_save_path, exist_ok=True)
                    
                    # Skip frames beyond available data
                    if frame_idx > len(car_dict_sequence_ori_data) - 1:
                        if os.path.exists(frame_save_path):
                            os.system(f'rm -r {frame_save_path}')
                        continue
                    
                    # Load frame-specific data
                    frame_car_dict = car_dict_sequence_ori_data[frame_idx].copy()
                    frame_ego_dict = dict()
                    
                    if 'ego_vehicle' in frame_car_dict.keys():
                        frame_ego_dict = frame_car_dict['ego_vehicle']
                        del frame_car_dict['ego_vehicle']
                    
                    # Initialize visibility tracking for all agents
                    for key, car_info in frame_car_dict.items():
                        if 'visible_camera_id' not in car_info.keys():
                            frame_car_dict[key]['visible_camera_id'] = []
                    
                    # Calculate ego vehicle pose
                    cur_ego_pose = ego_poses[frame_idx]
                    ego_pose_ori_cor = cur_ego_pose
                    roll, pitch, yaw, loc = disassemble_matrix(ego_pose_ori_cor)
                    frame_ego_dict['rot'] = [roll, pitch, yaw]
                    
                    # Setup camera calibration storage
                    camera_calib_dict = dict()
                    camera_calib_dict['timestamp'] = save_timestamp
                    
                    # Process each camera view
                    for cam_id in camera_ids:
                        rgb_path = os.path.join(scene_exp_path, f'{frame_idx}_{cam_id}_rgb.png')
                        rgb_save_path = os.path.join(frame_save_path, f'{idx}_{cam_id}_rgb.png')
                        
                        # Load and copy RGB image
                        img = cv2.imread(rgb_path)
                        if img is None:
                            continue
                            
                        img_width, img_height = img.shape[1], img.shape[0]
                        os.system(f'cp {rgb_path} {rgb_save_path}')
                        
                        # Load depth map for occlusion testing
                        depth_path = os.path.join(scene_exp_path, f'{frame_idx}_{cam_id}_depth.npy')
                        depth = np.load(depth_path)
                        
                        # Calculate camera pose and calibration
                        camera_extrinsic = camera_extrinsics[cam_id]
                        camera_intrinsic = camera_intrinsics[cam_id]
                        c2w = cur_ego_pose @ camera_extrinsic
                        extrinsic = c2w
                        cam_loc = extrinsic[:3, 3]
                        
                        # Setup calibration data
                        calibration = {
                            'intrinsic': camera_intrinsic,
                            'extrinsic': extrinsic
                        }
                        
                        # Transform to standard camera coordinate system
                        extrinsic_to_save = extrinsic @ np.linalg.inv(opencv2camera)
                        calibration_to_save = {
                            'intrinsic': camera_intrinsic.tolist(),
                            'extrinsic': extrinsic_to_save.tolist()
                        }
                        camera_calib_dict[cam_id] = calibration_to_save
                        
                        # Validate agent visibility using depth-based occlusion testing
                        agent_valid = []
                        points_valid = []
                        name_valid = []
                        cur_agent_car_info = agent_car_info[frame_idx]
                        
                        for key, car_info in cur_agent_car_info.items():
                            dim = car_info['bbox']
                            loc = car_info['loc']
                            
                            # Construct object pose matrix
                            R_z = rpy2R([car_info['rot'][0], car_info['rot'][1], car_info['rot'][2]])
                            T_vehicle_to_world = np.eye(4)
                            T_vehicle_to_world[:3, :3] = R_z
                            T_vehicle_to_world[:3, 3] = loc
                            T_vehicle_to_world[3, 3] = 1
                            
                            # Project 3D bounding box to 2D
                            points_uv, valid = project_label_to_image(
                                dim, T_vehicle_to_world, calibration, img_width, img_height
                            )
                            
                            # Check if all corners are valid (within image bounds)
                            if valid.all():
                                points_valid.append(points_uv)
                                name_valid.append(key)
                                agent_valid.append(car_info)
                        
                        # Depth-based occlusion testing
                        for idx_valid, points_uv in enumerate(points_valid):
                            car_info = agent_valid[idx_valid]
                            car_loc = car_info['loc']
                            distance = calculate_distance(car_loc, cam_loc)
                            
                            # Skip distant objects (>100m)
                            if distance > 100:
                                continue
                            
                            # Calculate object center in image coordinates
                            car_center_pixel = np.mean(points_uv, axis=0)
                            pixel_x = max(0, min(int(car_center_pixel[0]), img_width - 1))
                            pixel_y = max(0, min(int(car_center_pixel[1]), img_height - 1))
                            
                            # Check depth consistency for occlusion validation
                            car_center_depth = depth[pixel_y, pixel_x][0]
                            depth_diff = abs(distance - car_center_depth)
                            
                            if depth_diff < 5.0:
                                # Object is visible (depth matches projection)
                                frame_car_dict[name_valid[idx_valid]]['visible_camera_id'].append(cam_id)
                            else:
                                # Check multiple points for partial visibility
                                valid_num = 0
                                for point in points_uv:
                                    pixel_x = max(0, min(int(point[0]), img_width - 1))
                                    pixel_y = max(0, min(int(point[1]), img_height - 1))
                                    
                                    if 0 <= pixel_x < img_width and 0 <= pixel_y < img_height:
                                        depth_value = depth[pixel_y, pixel_x][0]
                                        if abs(distance - depth_value) < 5.0:
                                            valid_num += 1
                                
                                # Consider visible if sufficient points pass depth test
                                if valid_num >= 4:
                                    frame_car_dict[name_valid[idx_valid]]['visible_camera_id'].append(cam_id)
                    
                    # Save processed frame data
                    frame_car_dict_save_path = os.path.join(frame_save_path, 'agent_info.json')
                    with open(frame_car_dict_save_path, 'w') as f:
                        json.dump(frame_car_dict, f, indent=2)
                    
                    frame_ego_dict_save_path = os.path.join(frame_save_path, 'ego_pose.json')
                    with open(frame_ego_dict_save_path, 'w') as f:
                        json.dump(frame_ego_dict, f, indent=2)
                    
                    camera_calib_dict_save_path = os.path.join(frame_save_path, 'camera_calib.json')
                    with open(camera_calib_dict_save_path, 'w') as f:
                        json.dump(camera_calib_dict, f, indent=2)


# save simulation dataset
if __name__ == '__main__':
    main()