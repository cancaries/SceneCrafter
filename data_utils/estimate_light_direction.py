"""
Light Direction Estimation for SceneCrafter

This module provides functionality to estimate light direction from images
using gradient-based analysis. It processes reference images to extract
illumination direction vectors for use in simulation and rendering.

Usage:
    python estimate_light_direction.py --image_path /path/to/image.jpg
"""

import os
import cv2
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from transforms3d.euler import euler2mat, mat2euler

# Camera coordinate transformation matrix
opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])

def estimate_light_direction(image_path):
    """
    Estimate light direction from a single image using gradient analysis.
    
    This function analyzes image gradients to determine the dominant light
    direction based on the assumption that strong gradients often correspond
    to lighting changes.
    
    Args:
        image_path (str): Path to the input image file
        
    Returns:
        np.ndarray: Light direction vector [lx, ly, lz] normalized to unit length
        
    Raises:
        ValueError: If image cannot be loaded or path is invalid
    """
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or invalid path: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = gaussian_filter(gray, sigma=2)
    
    # Calculate gradients in x and y directions
    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x)
    
    # Find strongest gradient points (assumed to correlate with lighting)
    max_magnitude_index = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    dominant_direction = direction[max_magnitude_index]
    
    # Convert angle to unit vector [lx, ly, lz]
    lx = -np.cos(dominant_direction)
    ly = -np.sin(dominant_direction)
    lz = -1.0  # Assume light comes from above (positive z direction)
    
    light_direction = np.array([lx, ly, lz])
    light_direction /= np.linalg.norm(light_direction)  # Normalize to unit vector
    
    return light_direction

def transform_to_global(light_direction, camera_position, camera_orientation):
    """
    Transform local light direction to global coordinate system.
    
    This function converts a light direction vector from camera-local coordinates
    to global world coordinates using camera pose information.
    
    Args:
        light_direction (np.ndarray): Local light direction vector [lx, ly, lz]
        camera_position (np.ndarray): Camera position [cx, cy, cz]
        camera_orientation: Camera orientation as Euler angles [roll, pitch, yaw] 
                         or rotation matrix (3x3)
    
    Returns:
        np.ndarray: Global light direction vector [Lx, Ly, Lz]
        
    Raises:
        ValueError: If camera orientation format is invalid
    """
    # Convert Euler angles to rotation matrix if necessary
    if isinstance(camera_orientation, list) and len(camera_orientation) == 3:
        roll, pitch, yaw = camera_orientation
        rotation_matrix = euler2mat(roll, pitch, yaw)
    elif isinstance(camera_orientation, np.ndarray) and camera_orientation.shape == (3, 3):
        rotation_matrix = camera_orientation
    else:
        raise ValueError("Invalid camera orientation format.")
    
    # Transform local light direction to global coordinates
    global_light_direction = np.dot(rotation_matrix, light_direction)
    global_light_direction /= np.linalg.norm(global_light_direction)  # Normalize

    # Normalize z-component to -1 for consistent direction representation
    if global_light_direction[2] > 0.0:
        global_light_direction[2] = -global_light_direction[2]

    global_light_direction *= abs(1.0/global_light_direction[2])
    
    return global_light_direction

def process_scene_light_direction(scene_dir, frame_idx=50):
    """
    Process a scene directory to estimate light direction for a specific frame.
    
    This is the main entry point for integrating light direction estimation
    into the SceneCrafter data preparation pipeline.
    
    Args:
        scene_dir (str): Path to the scene directory containing images and pose data
        frame_idx (int): Frame index to process (default: 50)
        
    Returns:
        bool: True if light direction estimation successful, False otherwise
    """
    try:
        # Construct paths for required files
        ref_image_path = os.path.join(scene_dir, f"images/000{str(frame_idx).zfill(3)}_0.png")
        pose_path = os.path.join(scene_dir, "car_info/ego_pose_ori.json")
        extrinsics_path = os.path.join(scene_dir, "extrinsics/0.txt")
        output_path = os.path.join(scene_dir, "light_direction.json")
        
        # Check if all required files exist
        if not os.path.exists(ref_image_path):
            print(f"Image not found: {ref_image_path}")
            return False
        if not os.path.exists(pose_path):
            print(f"Pose file not found: {pose_path}")
            return False
        if not os.path.exists(extrinsics_path):
            print(f"Extrinsics file not found: {extrinsics_path}")
            return False
            
        # Load pose data
        with open(pose_path, "r") as f:
            pose_data = json.load(f)
        
        ego_position = pose_data[frame_idx]["location"]
        ego_orientation = pose_data[frame_idx]["rotation"]
        
        # Build ego pose matrix
        ego_pose = np.eye(4)
        rotation_matrix = euler2mat(ego_orientation[0], ego_orientation[1], ego_orientation[2])
        ego_pose[:3, :3] = rotation_matrix
        ego_pose[:3, 3] = ego_position

        # Load camera extrinsics
        front_camera_extrinsics = np.loadtxt(extrinsics_path)
        front_camera_pose = ego_pose @ front_camera_extrinsics @ np.linalg.inv(opencv2camera)
        front_camera_position = front_camera_pose[:3, 3]
        front_camera_rotation = list(mat2euler(front_camera_pose[:3, :3]))
        
        # Estimate light direction
        local_light_direction = estimate_light_direction(ref_image_path)
        global_light_direction = transform_to_global(
            local_light_direction, 
            front_camera_position, 
            front_camera_rotation
        )
        
        # Save light direction to JSON file
        with open(output_path, "w") as f:
            json.dump(global_light_direction.tolist(), f)
        
        print(f"Light direction saved: {global_light_direction}")
        return True
        
    except Exception as e:
        print(f"Error processing light direction: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Estimate light direction from images')
    parser.add_argument('--scene_dir', type=str, required=True,
                       help='Scene directory containing images and pose data')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='Frame index to process')
    
    args = parser.parse_args()
    
    success = process_scene_light_direction(args.scene_dir, args.frame_idx)
    if success:
        print("Light direction estimation completed successfully")
    else:
        print("Light direction estimation failed")