"""
Calibration and Geometry Utilities for SceneCrafter
"""

import os
import numpy as np
import math
import tensorflow as tf

def calculate_distance(p1, p2):
    """
    Calculate the Euclidean distance between two 3D points.
    
    This function computes the straight-line distance between two points in 3D space
    using the standard Euclidean distance formula. It supports both NumPy arrays and
    lists/tuples as input.
    
    Args:
        p1 (np.ndarray or list): First point coordinates [x, y, z]
        p2 (np.ndarray or list): Second point coordinates [x, y, z]
        
    Returns:
        float: Euclidean distance between the two points
        
    Example:
        >>> p1 = np.array([1.0, 2.0, 3.0])
        >>> p2 = np.array([4.0, 6.0, 8.0])
        >>> distance = calculate_distance(p1, p2)
        >>> print(distance)  # Output: 7.071...
    """
    return np.linalg.norm(p1 - p2)

def rpy2R(rpy):
    """
    Convert roll-pitch-yaw Euler angles to a 3x3 rotation matrix.
    
    This function generates a rotation matrix from Euler angles following the
    aerospace convention (XYZ rotation sequence). The rotation is applied in the
    order: first roll (X-axis), then pitch (Y-axis), finally yaw (Z-axis).
    
    Args:
        rpy (np.ndarray or list): Euler angles in radians [roll, pitch, yaw]
                                - roll: Rotation around X-axis
                                - pitch: Rotation around Y-axis  
                                - yaw: Rotation around Z-axis
        
    Returns:
        np.ndarray: 3x3 rotation matrix representing the combined rotation
        
    Mathematical Basis:
        R = Rz(yaw) * Ry(pitch) * Rx(roll)
        
    Example:
        >>> rpy = [0.1, 0.2, 0.3]  # radians
        >>> rotation_matrix = rpy2R(rpy)
        >>> print(rotation_matrix.shape)  # Output: (3, 3)
    """
    # Extract individual rotation angles
    roll, pitch, yaw = rpy
    
    # Rotation matrix around X-axis (roll)
    rot_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    # Rotation matrix around Y-axis (pitch)
    rot_y = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    # Rotation matrix around Z-axis (yaw)
    rot_z = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations: Rz * Ry * Rx
    R = np.dot(rot_z, np.dot(rot_y, rot_x))
    return R

def bbox_to_corner3d(bbox):
    """
    Convert a 3D axis-aligned bounding box to its 8 corner points.
    
    This function takes a bounding box defined by minimum and maximum coordinates
    and returns all 8 corner vertices in 3D space. The corners are ordered in a
    consistent manner for easy indexing and visualization.
    
    Args:
        bbox (np.ndarray): Bounding box defined by [min_xyz, max_xyz] with shape [2, 3]
                          - bbox[0]: [min_x, min_y, min_z]
                          - bbox[1]: [max_x, max_y, max_z]
        
    Returns:
        np.ndarray: 8 corner points of the 3D bounding box with shape [8, 3]
                   The corners are ordered as follows:
                   0: [min_x, min_y, min_z] - bottom-back-left
                   1: [min_x, min_y, max_z] - top-back-left
                   2: [min_x, max_y, min_z] - bottom-front-left
                   3: [min_x, max_y, max_z] - top-front-left
                   4: [max_x, min_y, min_z] - bottom-back-right
                   5: [max_x, min_y, max_z] - top-back-right
                   6: [max_x, max_y, min_z] - bottom-front-right
                   7: [max_x, max_y, max_z] - top-front-right
        
    Example:
        >>> bbox = np.array([[0, 0, 0], [1, 2, 3]])
        >>> corners = bbox_to_corner3d(bbox)
        >>> print(corners.shape)  # Output: (8, 3)
    """
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]
    
    # Generate all 8 corner points by combining min/max coordinates
    corner3d = np.array([
        [min_x, min_y, min_z],  # 0: bottom-back-left
        [min_x, min_y, max_z],  # 1: top-back-left
        [min_x, max_y, min_z],  # 2: bottom-front-left
        [min_x, max_y, max_z],  # 3: top-front-left
        [max_x, min_y, min_z],  # 4: bottom-back-right
        [max_x, min_y, max_z],  # 5: top-back-right
        [max_x, max_y, min_z],  # 6: bottom-front-right
        [max_x, max_y, max_z]   # 7: top-front-right
    ])
    return corner3d

def disassemble_matrix(matrix):
    """
    Decompose a 4x4 homogeneous transformation matrix into rotation and translation components.
    
    This function extracts the rotation angles (Euler angles) and translation vector
    from a 4x4 transformation matrix. The rotation angles are extracted using the
    analytical solution for rotation matrix decomposition.
    
    Args:
        matrix (np.ndarray): 4x4 homogeneous transformation matrix in the format:
                           [R | t]
                           [0 | 1]
                           where R is 3x3 rotation matrix and t is 3x1 translation vector
        
    Returns:
        tuple: (theta_x, theta_y, theta_z, translation) where:
            - theta_x: Roll angle in radians (rotation around X-axis)
            - theta_y: Pitch angle in radians (rotation around Y-axis)
            - theta_z: Yaw angle in radians (rotation around Z-axis)
            - translation: 3D translation vector [tx, ty, tz]
        
    Mathematical Basis:
        The rotation angles are extracted using the following formulas:
        theta_x = atan2(R[2,1], R[2,2])
        theta_y = atan2(-R[2,0], sqrt(R[2,1]^2 + R[2,2]^2))
        theta_z = atan2(R[1,0], R[0,0])
        
    Example:
        >>> transform = np.eye(4)  # Identity transformation
        >>> roll, pitch, yaw, trans = disassemble_matrix(transform)
        >>> print(roll, pitch, yaw)  # Output: 0.0 0.0 0.0
        >>> print(trans)  # Output: [0. 0. 0.]
    """
    # Ensure input is a NumPy array
    matrix = np.asarray(matrix)
    
    # Validate matrix dimensions
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")
    
    # Extract the 3x3 rotation matrix (upper-left submatrix)
    rotation = matrix[:3, :3]
    
    # Compute Euler angles using analytical solution
    # Roll angle (rotation around X-axis)
    theta_x = np.arctan2(rotation[2, 1], rotation[2, 2])
    
    # Pitch angle (rotation around Y-axis) with singularity handling
    sin_pitch = -rotation[2, 0]
    cos_pitch = np.sqrt(rotation[2, 1]**2 + rotation[2, 2]**2)
    theta_y = np.arctan2(sin_pitch, cos_pitch)
    
    # Yaw angle (rotation around Z-axis)
    theta_z = np.arctan2(rotation[1, 0], rotation[0, 0])
    
    # Extract the translation vector (4th column, first 3 rows)
    translation = matrix[:3, 3]
    
    return theta_x, theta_y, theta_z, translation


def rotate(point, angle):
    """
    Rotate a 3D point around the Z-axis by the specified angle.
    
    Args:
        point (np.ndarray): 3D point coordinates [x, y, z]
        angle (float): Rotation angle in radians
        
    Returns:
        np.ndarray: Rotated 3D point coordinates
    """
    # Create 3D rotation matrix around Z-axis
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    return np.dot(rotation_matrix, point)


def get_translation_from_matrix(matrix):
    """
    Extract translation vector from a 4x4 transformation matrix.
    
    Args:
        matrix (np.ndarray): 4x4 transformation matrix
        
    Returns:
        list: Translation vector [x, y, z]
    """
    return [matrix[0, 3], matrix[1, 3], matrix[2, 3]]


def get_rotation_from_matrix(matrix):
    """
    Extract rotation angles (roll, pitch, yaw) from a 4x4 transformation matrix.
    
    Args:
        matrix (np.ndarray): 4x4 transformation matrix
        
    Returns:
        list: Rotation angles [roll, pitch, yaw] in radians
    """
    # Calculate Euler angles from rotation matrix
    return [np.arctan2(matrix[2, 1], matrix[2, 2]),
            np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2)),
            np.arctan2(matrix[1, 0], matrix[0, 0])]


def get_matrix_from_rotation_and_translation(rotation, translation):
    """
    Construct a 4x4 transformation matrix from rotation angles and translation.
    
    Args:
        rotation (list): Rotation angles [roll, pitch, yaw] in radians
        translation (list): Translation vector [x, y, z]
        
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    # Initialize identity matrix
    matrix = np.eye(4)
    
    # Convert Euler angles to rotation matrix using XYZ convention
    rotation_matrix = np.reshape(
        np.array(tf.transformations.euler_matrix(rotation[0], rotation[1], rotation[2], 'sxyz'))[:3, :3], 
        [3, 3]
    )
    
    # Set rotation and translation components
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation
    
    return matrix