"""
Copyright (c) 2024 SceneCrafter Contributors
Licensed under the MIT License - see LICENSE file for details.

render_waymo.py - Waymo Dataset Rendering Module
"""

import os
import torch
import json
import numpy as np
import math
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import CameraSimple
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time
import torch.cuda as cuda
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree
def moving_average(z, window_size):
    """
    Apply moving average smoothing to a 1D array.
    
    Args:
        z (np.ndarray): Input array to smooth
        window_size (int): Size of the moving average window
        
    Returns:
        np.ndarray: Smoothed array with same length as input
    """
    pad_size = window_size // 2
    z_padded = np.pad(z, (pad_size, pad_size), mode='edge')
    return np.convolve(z_padded, np.ones(window_size)/window_size, mode='valid')

def rotate_matrix_by_deg(matrix, yaw_deg, pitch_deg=0, roll_deg=0):
    """
    Rotate a 4x4 transformation matrix by specified angles around all three axes.

    Parameters:
        matrix (np.ndarray): The original 4x4 transformation matrix.
        yaw_deg (float): Rotation angle around Z-axis in degrees.
        pitch_deg (float): Rotation angle around Y-axis in degrees (default: 0).
        roll_deg (float): Rotation angle around X-axis in degrees (default: 0).

    Returns:
        np.ndarray: The new 4x4 transformation matrix after rotation.
    """
    # Convert angle from degrees to radians
    yaw_rad = np.deg2rad(yaw_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    roll_rad = np.deg2rad(roll_deg)
    
    # Define rotation matrices for each axis
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0,                  0,                 1]
    ])
    Ry = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    # Extract the original rotation and translation components
    original_rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    
    # Apply rotation transformations
    new_rotation = np.dot(Rz, original_rotation)
    new_rotation = pitch_rotation_matrix(-pitch_deg, new_rotation)
    
    # Reconstruct the new 4x4 transformation matrix
    new_matrix = np.eye(4)
    new_matrix[:3, :3] = new_rotation
    new_matrix[:3, 3] = translation
    
    return new_matrix

def rotate_matrix(matrix, R):
    """
    Rotate a 4x4 transformation matrix using a provided rotation matrix.

    Parameters:
        matrix (np.ndarray): The original 4x4 transformation matrix.
        R (np.ndarray): 3x3 rotation matrix to apply.

    Returns:
        np.ndarray: The new 4x4 transformation matrix after rotation.
    """
    # Extract rotation component from provided matrix
    Rz = R[:3, :3]
    
    # Extract the original rotation and translation components
    original_rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    
    # Apply rotation transformation
    new_rotation = np.dot(Rz, original_rotation)
    
    # Reconstruct the new 4x4 transformation matrix
    new_matrix = np.eye(4)
    new_matrix[:3, :3] = new_rotation
    new_matrix[:3, 3] = translation
    
    return new_matrix


# OpenCV to camera coordinate system transformation matrix
opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])

def camera_intrinsic_transform(vfov=40, hfov=66, pixel_width=1600, pixel_height=900):
    """
    Create camera intrinsic matrix from field of view parameters.
    
    Args:
        vfov (float): Vertical field of view in degrees
        hfov (float): Horizontal field of view in degrees  
        pixel_width (int): Image width in pixels
        pixel_height (int): Image height in pixels
        
    Returns:
        np.ndarray: 3x3 camera intrinsic matrix
    """
    camera_intrinsics = np.zeros((3, 3))
    camera_intrinsics[2, 2] = 1
    camera_intrinsics[0, 0] = (pixel_width / 2.0) / math.tan(math.radians(hfov / 2.0))
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = (pixel_height / 2.0) / math.tan(math.radians(vfov / 2.0))
    camera_intrinsics[1, 2] = pixel_height / 2.0
    return camera_intrinsics

def camera_intrinsic_fov(intrinsic):
    """
    Calculate horizontal and vertical field of view from camera intrinsic matrix.
    
    Args:
        intrinsic (np.ndarray): 3x3 camera intrinsic matrix
        
    Returns:
        tuple: (fov_x, fov_y) field of view angles in degrees
    """
    w, h = intrinsic[0][2] * 2, intrinsic[1][2] * 2
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    
    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))
    return fov_x, fov_y

def rpy2R(rpy):
    """
    Convert roll-pitch-yaw angles to rotation matrix.
    
    Args:
        rpy (list): [roll, pitch, yaw] angles in radians
        
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    rot_x = np.array([[1, 0, 0],
                    [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                    [0, math.sin(rpy[0]), math.cos(rpy[0])]])
    rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                    [0, 1, 0],
                    [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
    rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                    [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                    [0, 0, 1]])
    R = np.dot(rot_z, np.dot(rot_y, rot_x))
    return R

def get_ego_vehicle_matrices(json_file, ego_poses_ori_json=None):
    """
    Load and process ego vehicle pose data from JSON files.
    
    This function loads ego vehicle position and orientation data, applies smoothing
    to the z-coordinate (height), and optionally incorporates roll and pitch angles
    from reference pose data using nearest neighbor lookup.
    
    Args:
        json_file (str): Path to ego vehicle JSON file with location and rotation data
        ego_poses_ori_json (str, optional): Path to original ego poses JSON file for roll/pitch data
        
    Returns:
        list: List of 4x4 transformation matrices representing ego vehicle poses
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    matrices = []
    ego_loc = []
    ego_yaw = []
    
    # Extract location and yaw data from JSON
    for item in data:
        if 'ego_vehicle' in item:
            ego_vehicle = item['ego_vehicle']
            loc = np.array(ego_vehicle['loc']) 
            ego_loc.append(loc)
            theta = ego_vehicle['rot'][2]  # Extract yaw angle
            ego_yaw.append(theta)

    # Smooth z-coordinate (height) using Gaussian filter
    ego_z = [loc[2] for loc in ego_loc]
    sigma = 2.0
    filtered_z = gaussian_filter1d(ego_z, sigma=sigma)
    ego_loc_new = []

    # Update locations with smoothed z-coordinates
    for idx, loc_ori in enumerate(ego_loc):
        loc = loc_ori.copy()
        loc[2] = filtered_z[idx]
        ego_loc_new.append(loc)

    # Load reference poses for roll and pitch angles if provided
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

    # Get roll and pitch angles using nearest neighbor lookup
    rolls = []
    for idx, loc in enumerate(ego_loc_new):
        roll = ego_poses_indices[kd_tree_rot.query(ego_loc_new[idx])[1]][0]
        rolls.append(roll)

    pitchs = []
    for idx, loc in enumerate(ego_loc_new):
        pitch = ego_poses_indices[kd_tree_rot.query(ego_loc_new[idx])[1]][1]
        pitchs.append(pitch)
    
    # Smooth roll and pitch angles
    sigma = 0.5
    rolls = gaussian_filter1d(rolls, sigma=sigma)
    pitchs = gaussian_filter1d(pitchs, sigma=sigma)
    
    # Construct transformation matrices
    for idx, loc in enumerate(ego_loc_new):
        theta = ego_yaw[idx]
        roll = rolls[idx]
        pitch = pitchs[idx]
        R_z = rpy2R([roll, pitch, theta])
        T_vehicle_to_world = np.eye(4)
        T_vehicle_to_world[:3, :3] = R_z  # Set rotation
        T_vehicle_to_world[:3, 3] = loc   # Set translation
        T_vehicle_to_world[3, 3] = 1
        matrices.append(T_vehicle_to_world)

    return matrices

def pitch_rotation_matrix(pitch_deg, R_current):
    """
    Apply pitch rotation to a current rotation matrix around the right axis.
    
    This function implements Rodrigues' rotation formula to rotate a matrix
    around an arbitrary axis (the right vector of the current rotation).
    
    Args:
        pitch_deg (float): Pitch angle in degrees for the rotation
        R_current (np.ndarray): Current 3x3 rotation matrix
        
    Returns:
        np.ndarray: New 3x3 rotation matrix after pitch rotation
    """
    pitch_rad = np.deg2rad(pitch_deg)
    
    # Extract right vector (local X-axis) from current rotation matrix
    right_vector = R_current[:, 0]
    
    # Apply Rodrigues' rotation formula
    x, y, z = right_vector
    c = np.cos(pitch_rad)
    s = np.sin(pitch_rad)
    R_pitch = np.array([
        [c + (1 - c) * x * x, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
        [(1 - c) * y * x + s * z, c + (1 - c) * y * y, (1 - c) * y * z - s * x],
        [(1 - c) * z * x - s * y, (1 - c) * z * y + s * x, c + (1 - c) * z * z]
    ])
    
    # Combine rotations: apply pitch rotation after current rotation
    R_final = np.dot(R_pitch, R_current)
    
    return R_final

def focal2fov(focal, pixels):
    """
    Convert focal length to field of view angle.
    
    Args:
        focal (float): Focal length in pixels
        pixels (float): Sensor dimension in pixels
        
    Returns:
        float: Field of view angle in radians
    """
    return 2 * math.atan(pixels / (2 * focal))

def load_extrinsics(file_path):
    """
    Load camera extrinsic matrix from text file.
    
    Args:
        file_path (str): Path to extrinsic matrix file (4x4)
        
    Returns:
        tuple: (rotation_matrix, translation_vector) where rotation_matrix is 3x3
               and translation_vector is 3x1
    """
    matrix = np.loadtxt(file_path)
    rot_metric = matrix[:3, :3]  # Extract 3x3 rotation matrix
    trans_offset = matrix[:3, 3]  # Extract 3x1 translation vector
    return rot_metric, trans_offset

def load_intrinsics(file_path):
    """
    Load camera intrinsic parameters from text file.
    
    Args:
        file_path (str): Path to intrinsic parameters file
        
    Returns:
        np.ndarray: 3x3 camera intrinsic matrix in standard format
    """
    intrinsic = np.loadtxt(file_path)
    
    # Extract focal lengths and principal point coordinates
    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    return intrinsic_matrix

def create_pitch_rotation_matrix(angle_degrees):
    """
    Create a rotation matrix for pitch rotation around the X-axis.
    
    Args:
        angle_degrees (float): Pitch angle in degrees
        
    Returns:
        np.ndarray: 3x3 rotation matrix for pitch rotation
    """
    theta = np.radians(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    return np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])

def render_novel():
    """
    Main rendering function for Waymo dataset scenes.
    
    This function orchestrates the complete rendering pipeline:
    1. Loads camera parameters and ego-vehicle trajectories
    2. Initializes Street Gaussian models and scene
    3. Generates novel views along the trajectory
    4. Renders and saves images/videos for multiple cameras
    
    Configuration is provided through the global 'cfg' object.
    """
    # Configure rendering settings
    cfg.render.save_image = True
    cfg.render.save_video = True
    
    # Extract configuration parameters
    text = cfg.render_name
    source_path = cfg.source_path
    scene_name = source_path.split('/')[-1]
    scene_number = cfg.scene_number
    data_path = source_path
    
    # Define camera configuration for Waymo dataset
    concat_cameras = [0, 1, 2, 3, 4]  # Camera IDs to render
    image_sizes = [(1280, 1920), (1280, 1920), (1280, 1920), (886, 1920), (886, 1920)]
    
    # Load camera extrinsic and intrinsic parameters
    ext_path = os.path.join(data_path, "extrinsics")
    int_path = os.path.join(data_path, "intrinsics")
    camera_extrinsics = []
    camera_intrinsics = []
    
    for cam_id in concat_cameras:
        # Load extrinsic matrix
        with open(os.path.join(ext_path, f'{cam_id}.txt'), 'r') as f:
            data = np.loadtxt(f)
            camera_extrinsics.append(data)
        
        # Load intrinsic parameters
        with open(os.path.join(int_path, f'{cam_id}.txt'), 'r') as f:
            data = load_intrinsics(f)
            camera_intrinsics.append(data)

    # Setup output directory
    trajectory_name = f'trajectory_{text}'
    
    # Initialize rendering pipeline with no gradient computation
    with torch.no_grad():
        # Load dataset and initialize Street Gaussian models
        dataset = Dataset(render_flag=True)
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        # Create scene with loaded Gaussians
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        # Setup output directory and clean if exists
        save_dir = os.path.join(cfg.model_path, trajectory_name, f"{scene_name}_{scene_number}")
        if os.path.exists(save_dir):
            os.system(f'rm -rf {save_dir}')
        visualizer = StreetGaussianVisualizer(save_dir)

        # Load ego-vehicle trajectory data
        if scene_number is None:
            ego_car_dict = os.path.join(source_path, 'car_info/car_dict_sequence_0.json')
        else:
            ego_car_dict = os.path.join(source_path, f'car_info/car_dict_sequence_{str(scene_number).zfill(3)}.json')
        print(ego_car_dict)
        print(f"Loading ego vehicle data from: {ego_car_dict}")

        # Load training and test cameras
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))
        
        # Load ego poses with optional reference data
        ego_poses_ori_json = os.path.join(source_path, 'car_info/ego_pose_ori.json')
        ego_poses = get_ego_vehicle_matrices(ego_car_dict, ego_poses_ori_json)

        # Get metadata from first camera
        basic_camera = cameras[0]
        meta = basic_camera.meta

        # Render trajectory with progress tracking
        for idx, cur_ego_pose in enumerate(tqdm(ego_poses, desc="Rendering Trajectory")):
            # Skip every 5th frame for efficiency
            if idx % 5 != 0:
                continue
                
            # Render for each camera
            for num_cam in range(len(concat_cameras)):
                cam_id = concat_cameras[num_cam]
                camera_extrinsic = camera_extrinsics[cam_id]
                camera_intrinsic = camera_intrinsics[cam_id]
                img_height, img_width = image_sizes[num_cam]
                
                # Compute camera-to-world transformation
                c2w = cur_ego_pose @ camera_extrinsic
                fx = camera_intrinsic[0, 0]
                fy = camera_intrinsic[1, 1]
                FovY = focal2fov(fx, img_height)
                FovX = focal2fov(fy, img_width)
                
                # Compute world-to-camera transformation
                RT = np.linalg.inv(c2w)
                R = RT[:3, :3].T
                T = RT[:3, 3]
                
                # Create camera object for rendering
                cur_camera = CameraSimple(
                    id=0,
                    ego_pose=cur_ego_pose,
                    extrinsic=camera_extrinsic,
                    R=R,
                    T=T,
                    H=img_height,
                    W=img_width,
                    FoVx=FovX,
                    FoVy=FovY,
                    K=camera_intrinsic,
                    image_name=f"{idx}_{cam_id}",
                    metadata=meta,
                )
                cur_camera.set_render_frame_idx(idx)

                # Update camera ID in metadata
                basic_camera.meta['cam'] = cam_id

                # Render and visualize results
                result = renderer.render_all(cur_camera, gaussians)
                visualizer.visualize(result, cur_camera)

        # Generate summary visualization
        visualizer.summarize()

if __name__ == "__main__":
    with torch.cuda.device(0):
        print("Rendering " + cfg.model_path)
        safe_state(cfg.eval.quiet)
        render_novel()