"""
Waymo Dataset Preparation for Simulation Data Generation

This module processes Waymo Open Dataset sequences to extract and prepare data for
simulation and neural rendering applications. It handles map features, ego vehicle poses,
static/dynamic actor data extraction, and light direction estimation from Waymo TFRecord files.

Usage Example:
    python prepare_simulatation_data.py \
        --process_list map ego_pose static_vehicle light_direction \
        --split_file data/data_split/map_gen.txt \
        --root_dir /path/to/waymo/tfrecords \
        --save_dir /path/to/save_dir \
        --segment_file /path/to/segment_file.txt
"""

import sys
import os
import numpy as np
import math
from matplotlib import pyplot as plt
import tensorflow as tf
import json
import argparse
import torch
from ground_model import MLP_model
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import frame_utils
from cal_utils import get_translation_from_matrix, get_rotation_from_matrix
from estimate_light_direction import process_scene_light_direction


def parse_seq_rawdata(process_list, root_dir, seq_name, seq_save_dir, track_file, 
                     scene_ground_model_path, scene_dir=None, start_idx=None, end_idx=None, debug=False):
    """
    Process a single Waymo sequence to extract simulation data.
    
    This function processes a Waymo TFRecord sequence to extract map features,
    ego poses, and actor data for simulation purposes. It handles coordinate
    transformations, ground elevation refinement, and data validation.
    
    Args:
        process_list (list): List of processing tasks ['map', 'egp_pose', 'static_vehicle', 'light_direction']
        root_dir (str): Root directory containing Waymo TFRecord files
        seq_name (str): Name of the sequence to process
        seq_save_dir (str): Directory to save processed data
        track_file (str): Path to tracking information file (optional)
        scene_ground_model_path (str): Path to ground elevation model file
        start_idx (int): Starting frame index (default: 0)
        end_idx (int): Ending frame index (default: last frame)
        debug (bool): Enable debug visualization
        
    Returns:
        bool: True if processing successful, False otherwise
        
    Output Files:
        - map_feature_w_model.json: Map features with ground-refined elevation
        - ego_pose.json: Ego vehicle pose sequence
        - actor_data/*.json: Per-frame actor information
        - all_static_actor_data.json: Static actors across the sequence
        - light_direction.json: Estimated light direction vector
        - map_feature.png: Debug visualization (if debug=True)
    """
    print(f'Processing sequence {seq_name}...')
    print(f'Saving to {seq_save_dir}')

    # Create output directory
    os.makedirs(seq_save_dir, exist_ok=True)
    
    # Load TFRecord dataset
    seq_path = os.path.join(root_dir, seq_name + '.tfrecord')
    error_flag = False
    
    # Read all frames from TFRecord
    dataset = tf.data.TFRecordDataset(seq_path, compression_type='')
    frames = []
    for data in dataset:
        frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
        frames.append(frame)
    num_frames = len(frames)
    
    # Load ground elevation model if provided
    if 'ground' in process_list and scene_ground_model_path:
        pc_range = [-200, -200, -3, 200, 200, 2]
        batch_size = 512
        g_model = MLP_model(
            hidden_dim=128,
            pc_range=pc_range,
            batch_size=batch_size,
            lr=0.005
        )
        print(f"Loading ground model from: {scene_ground_model_path}")
        g_model.model = torch.load(scene_ground_model_path)

    # Set frame processing range
    start_idx = start_idx or 0
    end_idx = end_idx or num_frames - 1
    
    # Calculate reference transformation for coordinate normalization
    transform_all = []
    for frame_id, frame in enumerate(frames):
        if frame_id < start_idx:
            continue
        if frame_id > end_idx:
            break
        transform = np.array(frame.pose.transform).reshape(4, 4)
        transform_all.append(transform)
    
    # Compute mean transformation as reference
    transform_all = np.stack(transform_all)
    ref_transform = np.mean(transform_all[:, :3, 3], axis=0)
    
    # Save reference transformation for later use
    with open(os.path.join(seq_save_dir, 'ref_transform.json'), 'w') as f:
        json.dump(ref_transform.tolist(), f, indent=2)

    # Process map features if requested
    if 'map' in process_list:
        # Initialize map feature containers
        road_edges = []
        lanes = []
        driveway = []
        road_line = []
        crosswalk = []
        map_feature_dict = {}
        
        # Process first frame for map features
        for i in range(len(frames[0].map_features)):
            feature_id = frames[0].map_features[i].id
            map_feature_dict[feature_id] = {}
            
            # Process lane features
            if len(frames[0].map_features[i].lane.polyline) > 0:
                map_feature_dict[feature_id]['feature_type'] = 'lane'
                
                # Extract lane metadata
                if frames[0].map_features[i].lane.type:
                    map_feature_dict[feature_id]['lane_type'] = frames[0].map_features[i].lane.type
                else:
                    map_feature_dict[feature_id]['lane_type'] = None
                
                # Extract lane polyline coordinates
                curr_lane = []
                for node in frames[0].map_features[i].lane.polyline:
                    node_position = np.ones(4)
                    node_position[0] = node.x
                    node_position[1] = node.y
                    node_position[2] = node.z
                    curr_lane.append(node_position)
                
                curr_lane = np.stack(curr_lane)
                curr_lane = curr_lane[:, :3] - ref_transform  # Normalize to reference frame
                map_feature_dict[feature_id]['polyline'] = curr_lane.tolist()
                lanes.append(curr_lane)
                
                # Extract additional lane properties
                if frames[0].map_features[i].lane.interpolating:
                    map_feature_dict[feature_id]['interpolating'] = frames[0].map_features[i].lane.interpolating
                
                if frames[0].map_features[i].lane.speed_limit_mph:
                    map_feature_dict[feature_id]['speed_limit_mph'] = frames[0].map_features[i].lane.speed_limit_mph
                
                # Extract lane connectivity information
                if frames[0].map_features[i].lane.entry_lanes:
                    entry_lanes = [str(entry_lane) for entry_lane in frames[0].map_features[i].lane.entry_lanes]
                    map_feature_dict[feature_id]['entry_lanes'] = entry_lanes
                
                if frames[0].map_features[i].lane.exit_lanes:
                    exit_lanes = [str(exit_lane) for exit_lane in frames[0].map_features[i].lane.exit_lanes]
                    map_feature_dict[feature_id]['exit_lanes'] = exit_lanes
                
                # Extract lane neighbor information
                if frames[0].map_features[i].lane.left_neighbors:
                    map_feature_dict[feature_id]['left_neighbors'] = []
                    for left_neighbor in frames[0].map_features[i].lane.left_neighbors:
                        left_neighbor_dict = {
                            'feature_id': str(left_neighbor.feature_id),
                            'self_start_index': left_neighbor.self_start_index,
                            'self_end_index': left_neighbor.self_end_index,
                            'neighbor_start_index': left_neighbor.neighbor_start_index,
                            'neighbor_end_index': left_neighbor.neighbor_end_index
                        }
                        map_feature_dict[feature_id]['left_neighbors'].append(left_neighbor_dict)
                
                if frames[0].map_features[i].lane.right_neighbors:
                    map_feature_dict[feature_id]['right_neighbors'] = []
                    for right_neighbor in frames[0].map_features[i].lane.right_neighbors:
                        right_neighbor_dict = {
                            'feature_id': str(right_neighbor.feature_id),
                            'self_start_index': right_neighbor.self_start_index,
                            'self_end_index': right_neighbor.self_end_index,
                            'neighbor_start_index': right_neighbor.neighbor_start_index,
                            'neighbor_end_index': right_neighbor.neighbor_end_index
                        }
                        map_feature_dict[feature_id]['right_neighbors'].append(right_neighbor_dict)

            # Process road edge features
            if len(frames[0].map_features[i].road_edge.polyline) > 0:
                map_feature_dict[feature_id]['feature_type'] = 'road_edge'
                
                if frames[0].map_features[i].road_edge.type:
                    map_feature_dict[feature_id]['road_edge_type'] = frames[0].map_features[i].road_edge.type
                
                curr_edge = []
                for node in frames[0].map_features[i].road_edge.polyline:
                    node_position = np.ones(4)
                    node_position[0] = node.x
                    node_position[1] = node.y
                    node_position[2] = node.z
                    curr_edge.append(node_position)
                
                curr_edge = np.stack(curr_edge)
                curr_edge = curr_edge[:, :3] - ref_transform
                road_edges.append(curr_edge)
                map_feature_dict[feature_id]['polyline'] = curr_edge.tolist()

            # Process crosswalk features
            if len(frames[0].map_features[i].crosswalk.polygon) > 0:
                map_feature_dict[feature_id]['feature_type'] = 'crosswalk'
                
                curr_polygon = []
                for node in frames[0].map_features[i].crosswalk.polygon:
                    node_position = np.ones(4)
                    node_position[0] = node.x
                    node_position[1] = node.y
                    node_position[2] = node.z
                    curr_polygon.append(node_position)
                
                curr_polygon = np.stack(curr_polygon)
                curr_polygon = curr_polygon[:, :3] - ref_transform
                crosswalk.append(curr_polygon)
                map_feature_dict[feature_id]['polyline'] = curr_polygon.tolist()

            # Process road line features
            if len(frames[0].map_features[i].road_line.polyline) > 0:
                map_feature_dict[feature_id]['feature_type'] = 'road_line'
                
                curr_polygon = []
                for node in frames[0].map_features[i].road_line.polyline:
                    node_position = np.ones(4)
                    node_position[0] = node.x
                    node_position[1] = node.y
                    node_position[2] = node.z
                    curr_polygon.append(node_position)
                
                curr_polygon = np.stack(curr_polygon)
                curr_polygon = curr_polygon[:, :3] - ref_transform
                road_line.append(curr_polygon)
                map_feature_dict[feature_id]['polyline'] = curr_polygon.tolist()

        # Validate map features
        if len(map_feature_dict.keys()) < 1:
            print('No map features found in the first frame')
            error_flag = True
            return False
        
        # save original map features
        map_json_save_path = os.path.join(seq_save_dir, 'map_feature.json')
        with open(map_json_save_path, 'w') as f:
            json.dump(map_feature_dict, f, indent=2)

        # Refine map feature elevation using ground model
        if 'ground' in process_list and scene_ground_model_path:
            print("Refining map feature elevation with ground model...")
            for map_feature_id, map_feature in map_feature_dict.items():
                if 'polyline' in map_feature.keys():
                    # Load polyline coordinates
                    feature_polyline = torch.tensor(map_feature['polyline'].copy(), dtype=torch.float32)
                    
                    # Predict ground elevation using neural model
                    z_new = g_model.inference(feature_polyline)
                    z_new = z_new.reshape(-1)
                    
                    # Update elevation coordinates
                    feature_polyline[:, 2] = z_new
                    map_feature['polyline'] = feature_polyline.tolist()

            # Save refined map features
            map_json_save_path = os.path.join(seq_save_dir, 'map_feature_w_model.json')
            with open(map_json_save_path, 'w') as f:
                json.dump(map_feature_dict, f, indent=2)

        # Generate debug visualization if requested
        if debug:
            # Define visualization bounds
            x_min, x_max = -300, 500
            y_min, y_max = -200, 200
            
            # Crop map features to visualization bounds
            cropped_road_edges = []
            for edge in road_edges:
                new_road_edge = []
                for i in range(edge.shape[0]):
                    if (x_min <= edge[i, 0] <= x_max and 
                        y_min <= edge[i, 1] <= y_max):
                        new_road_edge.append(edge[i])
                if len(new_road_edge) > 0:
                    new_road_edge = np.stack(new_road_edge)
                    cropped_road_edges.append(new_road_edge)

            cropped_lanes = []
            for lane in lanes:
                new_lane = []
                for i in range(lane.shape[0]):
                    if (x_min <= lane[i, 0] <= x_max and 
                        y_min <= lane[i, 1] <= y_max):
                        new_lane.append(lane[i])
                if len(new_lane) > 0:
                    new_lane = np.stack(new_lane)
                    cropped_lanes.append(new_lane)

            # Create visualization
            plt.figure(figsize=(24, 16), dpi=200)
            
            # Plot road edges in red
            for edge in cropped_road_edges:
                edge = np.array(edge)
                plt.plot(edge[:, 0], edge[:, 1], c='red')

            # Plot lanes in green
            for lane in cropped_lanes:
                lane = np.array(lane)
                plt.plot(lane[:, 0], lane[:, 1], c='green')

            # Plot driveways in blue
            for driveway_edge in driveway:
                driveway_edge = np.array(driveway_edge)
                plt.plot(driveway_edge[:, 0], driveway_edge[:, 1], c='blue')

            # Plot road lines in yellow
            for road_line_poly in road_line:
                edge = np.array(road_line_poly)
                plt.plot(edge[:, 0], edge[:, 1], c='yellow', linewidth=1)

            # Plot crosswalks in black
            for crosswalk_edge in crosswalk:
                crosswalk_edge = np.array(crosswalk_edge)
                crosswalk_edge = np.concatenate([crosswalk_edge, crosswalk_edge[0:1]], axis=0)
                plt.plot(crosswalk_edge[:, 0], crosswalk_edge[:, 1], c='black')

            # Save visualization
            plt_save_path = os.path.join(seq_save_dir, 'map_feature.png')
            plt.savefig(plt_save_path)
            plt.close()

    # Process ego vehicle poses if requested
    if 'egp_pose' in process_list:
        ego_pose_data = []
        print("Processing ego pose...")
        
        for frame_id, frame in enumerate(frames):
            cur_ego_pose_data = {}
            
            # Extract ego vehicle pose from frame
            cur_ego_pose_global = np.reshape(np.array(frame.pose.transform), [4, 4])
            
            # Normalize to reference frame
            cur_ego_pose = cur_ego_pose_global.copy()
            cur_ego_pose[:3, 3] = cur_ego_pose_global[:3, 3] - ref_transform.reshape(1, 3)
            
            # Extract pose components
            cur_ego_pose_data['location'] = get_translation_from_matrix(cur_ego_pose)
            cur_ego_pose_data['rotation'] = get_rotation_from_matrix(cur_ego_pose)
            
            ego_pose_data.append(cur_ego_pose_data)

        # Save ego pose data
        ego_pose_save_dir = os.path.join(seq_save_dir, 'ego_pose.json')
        with open(ego_pose_save_dir, 'w') as f:
            json.dump(ego_pose_data, f, indent=2)

    # Process static vehicle data if requested
    if 'static_vehicle' in process_list:
        all_static_actor_data = {}
        actor_data_output_path = os.path.join(seq_save_dir, 'actor_data')
        
        if not os.path.exists(actor_data_output_path):
            os.makedirs(actor_data_output_path)
        
        frame_obj_dict = {}
        
        # Process each frame
        for frame_id, frame in enumerate(frames):
            actor_data = []
            
            # Process all labeled objects
            for label in frame.laser_labels:
                label_id = label.id
                
                # Skip if already processed this object
                if label_id in frame_obj_dict.keys():
                    continue
                else:
                    frame_obj_dict[label_id] = []
                
                # Extract object information
                box = label.box
                meta = label.metadata
                
                # Determine object class
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    obj_class = "vehicle"
                elif label.type == label_pb2.Label.Type.TYPE_PEDESTRIAN:
                    obj_class = "pedestrian"
                elif label.type == label_pb2.Label.Type.TYPE_SIGN:
                    obj_class = "sign"
                elif label.type == label_pb2.Label.Type.TYPE_CYCLIST:
                    obj_class = "cyclist"
                else:
                    obj_class = "misc"
                
                # Calculate object properties
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])
                point_num_in_lidar = label.num_lidar_points_in_box
                most_visible_camera_name = label.most_visible_camera_name
                
                # Filter by camera visibility
                if most_visible_camera_name not in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']:
                    continue
                
                # Determine motion state (following EmerNeRF threshold)
                is_dynamic = bool(speed > 1.0)
                is_vehicle = bool(label.type == label_pb2.Label.Type.TYPE_VEHICLE)
                
                # Build 3D bounding box
                length, width, height = box.length, box.width, box.height
                size = [width, length, height]
                
                # Build 3D bounding box pose
                tx, ty, tz = box.center_x, box.center_y, box.center_z
                heading = box.heading
                c, s = math.cos(heading), math.sin(heading)
                rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                
                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = rotz_matrix
                obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
                
                # Transform to global coordinates
                cur_frame_transform = np.reshape(np.array(frame.pose.transform), [4, 4])
                obj_pose_global = np.dot(cur_frame_transform, obj_pose_vehicle)
                
                # Normalize to reference frame
                obj_pose_ref = obj_pose_global.copy()
                obj_pose_ref[:3, 3] = obj_pose_global[:3, 3] - ref_transform
                
                # Create actor data dictionary
                tmp_dict = dict()
                tmp_dict['label_id'] = label_id
                tmp_dict['size'] = size
                tmp_dict['obj_class'] = obj_class
                tmp_dict['point_num_in_lidar'] = point_num_in_lidar
                tmp_dict['speed'] = speed
                tmp_dict['if_vehicle'] = is_vehicle
                tmp_dict['motion_state'] = is_dynamic
                tmp_dict['pose'] = obj_pose_ref.tolist()
                tmp_dict['location'] = get_translation_from_matrix(obj_pose_ref)
                tmp_dict['rotation'] = get_rotation_from_matrix(obj_pose_ref)
                
                actor_data.append(tmp_dict)
                
                # Track static objects
                if not is_dynamic:
                    all_static_actor_data[label_id] = tmp_dict
            
            # Save per-frame actor data
            frame_id_name = str(frame_id).zfill(3)
            with open(os.path.join(actor_data_output_path, frame_id_name + '.json'), 'w') as f:
                json.dump(actor_data, f, indent=2)

        # Filter out dynamic objects from static actor list
        for obj_id in frame_obj_dict.keys():
            if len(frame_obj_dict[obj_id]) < 1:
                continue
            
            # Calculate object movement
            positions = np.array(frame_obj_dict[obj_id])
            distance = np.linalg.norm(positions[0] - positions[-1])
            is_moving = np.any(np.std(positions, axis=0) > 1) or distance > 2
            
            # Remove dynamic objects from static list
            if is_moving and obj_id in all_static_actor_data.keys():
                all_static_actor_data.pop(obj_id)

        # Save static actor data
        with open(os.path.join(seq_save_dir, 'all_static_actor_data.json'), 'w') as f:
            json.dump(all_static_actor_data, f, indent=2)

    # Process light direction estimation if requested
    if 'light_direction' in process_list and scene_dir is not None:
        print("Processing light direction estimation...")
        light_success = process_scene_light_direction(scene_dir)
        if light_success:
            print("Light direction estimation completed successfully")
        else:
            print("Warning: Light direction estimation failed for this scene")

    return True


def main():
    parser = argparse.ArgumentParser(description='Process Waymo sequences for simulation data')
    parser.add_argument('--process_list', type=str, nargs='+', 
                       default=['map', 'egp_pose', 'static_vehicle', 'ground', 'light_direction'],
                       help='List of processing tasks to perform: map, ego_pose, static_vehicle, ground, light_direction')
    parser.add_argument('--root_dir', type=str, 
                       default='demo/waymo_tfrecord/training/',
                       help='Root directory containing Waymo TFRecord files')
    parser.add_argument('--processed_waymo_dir', type=str, 
                       default='demo/waymo_dataset/',
                       help='Root directory containing processed Waymo dataset')
    parser.add_argument('--save_dir', type=str, 
                       default='demo/waymo_scene_data',
                       help='Directory to save processed data')
    parser.add_argument('--split_file', type=str, 
                       default='demo/data_split/train_waymo.txt',
                       help='File containing scene ID and sequence name mappings')
    parser.add_argument('--ground_model_path', type=str,
                       default="demo/ground_model/",
                       help='Directory containing ground elevation models')
    parser.add_argument('--segment_file', type=str,
                       default='demo/data_split/segment_list_train.txt',
                       help='File containing segment list')
    parser.add_argument('--gpu', type=str, default=None,
                       help='GPU device ID to use for processing')

    args = parser.parse_args()
    
    # Set up processing parameters
    process_list = args.process_list
    root_dir = args.root_dir
    processed_waymo_dir = args.processed_waymo_dir
    save_dir = args.save_dir
    ground_model_path = args.ground_model_path
    gpu_id = args.gpu
    
    # Configure GPU if specified
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Load sequence mappings
    split_file = open(args.split_file, "r").readlines()[1:]
    scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    seq_names = [line.strip().split(",")[1] for line in split_file]
    
    segment_file = args.segment_file
    seq_lists = open(segment_file).read().splitlines()
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Process sequences
    error_seq_list = []
    scene_ids_list = ['019']  # Override for testing
    
    for i, scene_id in enumerate(scene_ids_list):
        print(f'Processing scene {scene_id}...')
        
        # Validate sequence name consistency
        print(seq_names[i][3:])
        print(seq_lists[int(scene_id)][8:14])
        assert seq_names[i][3:] == seq_lists[int(scene_id)][8:14]
        
        # Set up paths
        seq_save_dir = os.path.join(save_dir, str(scene_id).zfill(3))
        tf_record_dir = os.path.join(root_dir, seq_lists[int(scene_id)] + '.tfrecord')
        scene_dir = os.path.join(processed_waymo_dir, str(scene_id).zfill(3))
        
        # Check if sequence exists
        if not os.path.exists(tf_record_dir):
            print(f'{tf_record_dir} does not exist')
            error_seq_list.append(seq_lists[int(scene_id)])
            continue
        
        # Load ground model if available
        if ground_model_path:
            scene_ground_model_path = os.path.join(args.ground_model_path, scene_id, 'ground_model.pth')
        else:
            scene_ground_model_path = None
        
        # Process sequence
        valid = parse_seq_rawdata(
            process_list=process_list,
            root_dir=root_dir,
            seq_name=seq_lists[int(scene_id)],
            seq_save_dir=seq_save_dir,
            track_file=None,
            scene_ground_model_path=scene_ground_model_path,
            scene_dir=scene_dir,
            debug=True
        )
        
        if not valid:
            error_seq_list.append(seq_lists[scene_id])

    # Report processing errors
    if len(error_seq_list) > 0:
        import time
        time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        
        with open(os.path.join(save_dir, 'error_sequence_list.txt'), 'w') as f:
            f.write(f'Error sequence list at {time_now}\n')
            for seq in error_seq_list:
                f.write(seq + '\n')


if __name__ == '__main__':
    main()