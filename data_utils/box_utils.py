"""
Box utilities for 3D bounding box operations and 2D mask generation.
"""

import numpy as np
import cv2

def get_bound_2d_mask(corners_3d, K, pose, H, W):
    """
    Generate a 2D mask from 3D bounding box corners projected onto an image.
    
    This function takes 3D corner points of a bounding box, applies camera transformation
    and projection, then generates a 2D binary mask indicating where the projected box
    appears in the image.
    
    Args:
        corners_3d (np.ndarray): 3D coordinates of box corners, shape (8, 3)
        K (np.ndarray): Camera intrinsic matrix, shape (3, 3)
        pose (np.ndarray): Camera pose matrix (world to camera), shape (4, 4)
        H (int): Image height in pixels
        W (int): Image width in pixels
    
    Returns:
        np.ndarray: Binary mask of shape (H, W) where 1 indicates the projected box area
    """
    # Transform 3D corners from world to camera coordinate system
    corners_3d = np.dot(corners_3d, pose[:3, :3].T) + pose[:3, 3:].T
    
    # Prevent division by zero in projection by clipping z coordinates
    corners_3d[..., 2] = np.clip(corners_3d[..., 2], a_min=1e-3, a_max=None)
    
    # Project 3D points to 2D image coordinates using camera intrinsic matrix
    corners_3d = np.dot(corners_3d, K.T)
    corners_2d = corners_3d[:, :2] / corners_3d[:, 2:]
    corners_2d = np.round(corners_2d).astype(int)
    
    # Initialize empty mask
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Fill the 6 faces of the projected 3D box to create a solid 2D mask
    # Each fillPoly call corresponds to one face of the box
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)  # Back face
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)  # Front face
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)  # Left face
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)  # Right face
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)  # Bottom face
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)  # Top face
    
    return mask


def scale_to_corrner(scale):
    """
    Generate 3D box corners from a symmetric scale parameter.
    
    Creates a cube centered at the origin with dimensions 2*scale x 2*scale x 2*scale.
    The cube has 8 corners corresponding to all combinations of min/max coordinates.
    
    Args:
        scale (float): Half of the cube size (distance from center to any face)
    
    Returns:
        np.ndarray: 3D coordinates of 8 box corners, shape (8, 3)
    """
    min_x, min_y, min_z = -scale, -scale, -scale
    max_x, max_y, max_z = scale, scale, scale
    
    # Define all 8 corners of the cube using min/max coordinates
    corner3d = np.array([
        [min_x, min_y, min_z],  # Corner 0: (-s, -s, -s)
        [min_x, min_y, max_z],  # Corner 1: (-s, -s, s)
        [min_x, max_y, min_z],  # Corner 2: (-s, s, -s)
        [min_x, max_y, max_z],  # Corner 3: (-s, s, s)
        [max_x, min_y, min_z],  # Corner 4: (s, -s, -s)
        [max_x, min_y, max_z],  # Corner 5: (s, -s, s)
        [max_x, max_y, min_z],  # Corner 6: (s, s, -s)
        [max_x, max_y, max_z],  # Corner 7: (s, s, s)
    ])
    return corner3d

def bbox_to_corner3d(bbox):
    """
    Convert axis-aligned bounding box to 3D corner coordinates.
    
    Takes a bounding box defined by minimum and maximum coordinates and returns
    the 8 corner points of the box. The corners are ordered consistently with
    the scale_to_corrner function.
    
    Args:
        bbox (np.ndarray): Bounding box defined as [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    
    Returns:
        np.ndarray: 3D coordinates of 8 box corners, shape (8, 3)
    """
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]
    
    # Generate all 8 corners from the bounding box min/max values
    corner3d = np.array([
        [min_x, min_y, min_z],  # Corner 0: minimum coordinates
        [min_x, min_y, max_z],  # Corner 1: min x,y, max z
        [min_x, max_y, min_z],  # Corner 2: min x,z, max y
        [min_x, max_y, max_z],  # Corner 3: min x, max y,z
        [max_x, min_y, min_z],  # Corner 4: max x, min y,z
        [max_x, min_y, max_z],  # Corner 5: max x, min y, max z
        [max_x, max_y, min_z],  # Corner 6: max x,y, min z
        [max_x, max_y, max_z],  # Corner 7: maximum coordinates
    ])
    return corner3d

def points_to_bbox(points):
    """
    Compute axis-aligned bounding box from a set of 3D points.
    
    Finds the minimum and maximum coordinates along each axis to create
    the tightest-fitting axis-aligned bounding box that contains all points.
    
    Args:
        points (np.ndarray): 3D point cloud, shape (N, 3)
    
    Returns:
        np.ndarray: Bounding box as [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    """
    # Find minimum and maximum coordinates along each axis
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    
    # Combine into bounding box format
    bbox = np.array([min_xyz, max_xyz])
    return bbox

def inbbox_points(points, corner3d):
    """
    Test which 3D points lie inside a 3D bounding box.
    
    Determines for each point whether it lies within the bounds defined by
    the box corners. The box is assumed to be axis-aligned.
    
    Args:
        points (np.ndarray): 3D points to test, shape (N, 3)
        corner3d (np.ndarray): Box corners defining the bounding box, shape (8, 3)
    
    Returns:
        np.ndarray: Boolean array indicating which points are inside the box
    """
    # Extract minimum and maximum coordinates from corner points
    min_xyz = corner3d[0]  # First corner has minimum coordinates
    max_xyz = corner3d[-1]  # Last corner has maximum coordinates
    
    # Check if each point lies within the bounding box
    return np.logical_and(
        np.all(points >= min_xyz, axis=-1),  # All coordinates >= minimum
        np.all(points <= max_xyz, axis=-1)   # All coordinates <= maximum
    )