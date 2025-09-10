"""
Waymo dataset utilities for camera parameter handling.
"""

import numpy as np


def load_extrinsics(file_path):
    """
    Load camera extrinsic parameters from text file.
    
    Reads a 4x4 transformation matrix from a text file and extracts the rotation
    matrix and translation vector. The matrix represents the transformation from
    world coordinates to camera coordinates.
    
    Args:
        file_path (str): Path to extrinsics file containing 4x4 transformation matrix
        
    Returns:
        tuple: (rotation_matrix, translation_vector) where
            - rotation_matrix (np.ndarray): 3x3 rotation matrix from world to camera
            - translation_vector (np.ndarray): 3D translation vector from world to camera
            
    Example:
        >>> rot, trans = load_extrinsics('camera_extrinsics.txt')
        >>> print(rot.shape)  # (3, 3)
        >>> print(trans.shape)  # (3,)
    """
    # Load the 4x4 transformation matrix from the text file
    # The matrix format is: [R | t] where R is 3x3 rotation and t is 3x1 translation
    matrix = np.loadtxt(file_path)
    
    # Validate matrix shape
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {matrix.shape}")
    
    # Extract the 3x3 rotation matrix (upper-left 3x3 submatrix)
    rotation_matrix = matrix[:3, :3]
    
    # Extract the 3D translation vector (first three elements of last column)
    translation_vector = matrix[:3, 3]
    
    return rotation_matrix, translation_vector


def load_intrinsics(file_path):
    """
    Load camera intrinsic parameters from text file.
    
    Reads camera intrinsic parameters from a text file and constructs a 3x3
    camera intrinsic matrix. The file should contain at least 4 values in order:
    [fx, fy, cx, cy] where fx and fy are focal lengths, and cx, cy are the
    principal point coordinates.
    
    Args:
        file_path (str): Path to intrinsics file containing camera parameters
        
    Returns:
        np.ndarray: 3x3 camera intrinsic matrix K in the format:
                   [[fx,  0, cx],
                    [ 0, fy, cy],
                    [ 0,  0,  1]]
                    
    Note:
        The intrinsic matrix follows the standard pinhole camera model:
        - fx, fy: focal lengths in pixels
        - cx, cy: principal point coordinates (usually image center)
        
    Example:
        >>> K = load_intrinsics('camera_intrinsics.txt')
        >>> print(K.shape)  # (3, 3)
        >>> print(K[0, 0], K[1, 1])  # fx, fy focal lengths
    """
    # Read camera parameters from text file
    # Expected format: [fx, fy, cx, cy, ...] (additional values ignored)
    intrinsic_params = np.loadtxt(file_path)
    
    # Ensure we have at least 4 parameters
    if len(intrinsic_params) < 4:
        raise ValueError("Intrinsic file must contain at least 4 parameters: [fx, fy, cx, cy]")
    
    # Extract the four essential parameters
    fx, fy, cx, cy = intrinsic_params[0], intrinsic_params[1], intrinsic_params[2], intrinsic_params[3]
    
    # Construct the 3x3 camera intrinsic matrix
    # This matrix projects 3D camera coordinates to 2D image coordinates
    intrinsic_matrix = np.array([
        [fx, 0, cx],   # fx: focal length in x, cx: principal point x-coordinate
        [0, fy, cy],   # fy: focal length in y, cy: principal point y-coordinate
        [0, 0, 1]      # Homogeneous coordinate normalization
    ])
    
    return intrinsic_matrix