#!/usr/bin/env python3
"""
SceneController Agents Utilities

This module provides a comprehensive set of utility functions for the SceneController agents system.
It includes functionality for file I/O operations, coordinate system transformations, video generation,
color detection and classification, 3D bounding box processing, depth map scaling, HDR sky blending,
and various mathematical utilities for scene processing and rendering.
"""

import os
import yaml
import torch
import copy 
import cv2
import json
import collections
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from termcolor import colored
from tqdm import tqdm


class Struct:
    """
    A simple struct-like class for converting dictionaries to objects.
    
    This class allows dictionary keys to be accessed as object attributes,
    making it easier to work with configuration data loaded from JSON/YAML files.
    
    Usage:
        data = {'key1': 'value1', 'key2': 42}
        config = Struct(**data)
        print(config.key1)  # Outputs: value1
    """
    
    def __init__(self, **entries):
        """Initialize the struct with dictionary entries as attributes."""
        self.__dict__.update(entries)


def read_yaml(file_path):
    """
    Read and parse a YAML configuration file.
    
    Args:
        file_path (str): Path to the YAML file to read.
        
    Returns:
        dict: Parsed YAML data as a Python dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def dump_yaml(data, savepath):
    """
    Save configuration data to a YAML file.
    
    Args:
        data (dict): Configuration data to save.
        savepath (str): Directory path where the YAML file will be saved.
        
    Returns:
        None
    """
    with open(os.path.join(savepath, 'config.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def check_and_mkdirs(path):
    """
    Create directories if they don't exist.
    
    Args:
        path (str): Directory path to create.
        
    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def generate_video(scene, prompt):
    """
    Generate a video from rendered scene frames and save individual frames.
    
    This function creates an MP4 video from the final rendered frames stored in a scene object,
    and also saves individual PNG frames to a separate folder. Handles cleanup if specified.
    
    Args:
        scene: Scene object containing:
            - output_dir (str): Base directory for output
            - logging_name (str): Subdirectory name for this render
            - final_video_frames (list): List of image arrays (H, W, 3)
            - fps (int): Frames per second for the video
            - save_cache (bool): Whether to keep cache files after processing
    
    Returns:
        None
    """
    video_output_path = os.path.join(scene.output_dir, scene.logging_name)
    check_and_mkdirs(video_output_path)
    filename = prompt.replace(' ', '_')[:40]
    fps = scene.fps
    
    # Log start of video generation
    print(colored("[Compositing video]", 'blue', attrs=['bold']), "start...")

    # Create video writer
    writer = imageio.get_writer(os.path.join(video_output_path, f"{filename}.mp4"), 
                                fps=fps)
    
    # Write frames with progress bar
    for frame in tqdm(scene.final_video_frames):
        writer.append_data(frame)
    writer.close()
    
    # Save individual frames to folder
    check_and_mkdirs(os.path.join(video_output_path, f"{filename}"))
    for i, img in enumerate(scene.final_video_frames):
        imageio.imsave(os.path.join(video_output_path, f"{filename}/{i}.png"), img)
    
    # Clean cache if not saving
    if not scene.save_cache:
        scene.clean_cache()

    print(colored("[Compositing video]", 'blue', attrs=['bold']), "done.")


def transform_nerf2opencv_convention(extrinsic):
    """
    Transform NeRF convention extrinsic matrix to OpenCV convention.
    
    Converts from NeRF's Right-Up-Back (RUB) coordinate system to OpenCV's
    Right-Down-Forward (RDF) coordinate system. This involves flipping the
    Y and Z axes while maintaining the right-handed coordinate system.
    
    Args:
        extrinsic (np.ndarray): NeRF extrinsic matrix with shape [3, 4] in RUB convention.
    
    Returns:
        np.ndarray: OpenCV extrinsic matrix with shape [4, 4] in RDF convention.
    """
    all_ones = np.array([[0, 0, 0, 1]])
    extrinsic_opencv = np.concatenate((extrinsic, all_ones), axis=0)

    # Flip Y and Z axes to convert RUB to RDF
    extrinsic_opencv = np.concatenate(
        (
            extrinsic_opencv[:, 0:1],      # X remains the same
            -extrinsic_opencv[:, 1:2],     # Flip Y
            -extrinsic_opencv[:, 2:3],     # Flip Z
            extrinsic_opencv[:, 3:],       # Translation remains
        ),
        axis=1
    )

    return extrinsic_opencv


def rotate(point, angle):
    """
    Rotate a 3D point around the Z-axis by a given angle.
    
    Args:
        point (np.ndarray): 3D point coordinates [x, y, z]
        angle (float): Rotation angle in radians
        
    Returns:
        np.ndarray: Rotated 3D point coordinates
    """
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    return np.dot(rotation_matrix, point)


def generate_vertices(car):
    """
    Generate 3D vertices for a vehicle bounding box.
    
    Creates 8 corner vertices of a 3D bounding box based on vehicle dimensions
    and orientation. The box is centered at the vehicle's position and rotated
    according to its heading.
    
    Args:
        car (dict): Vehicle information containing:
            - cx, cy, cz: Center coordinates
            - length, width, height: Vehicle dimensions
            - heading: Orientation angle in radians
            
    Returns:
        np.ndarray: Array of 8 vertices, each as [x, y, z] coordinates
    """
    x = car["cx"]
    y = car["cy"]
    z = car["cz"]
    length = car["length"]
    width = car["width"]
    height = car["height"]
    heading = car["heading"]
    
    box_center = np.array([x, y, z])
    half_dims = np.array([length / 2, width / 2, height / 2])

    # Define relative positions of the 8 vertices from box center
    relative_positions = (
        np.array([
            [-1, -1, -1],  # Bottom-back-left
            [-1, -1, 1],   # Top-back-left
            [-1, 1, -1],   # Bottom-front-left
            [-1, 1, 1],    # Top-front-left
            [1, -1, -1],   # Bottom-back-right
            [1, -1, 1],    # Top-back-right
            [1, 1, -1],    # Bottom-front-right
            [1, 1, 1],     # Top-front-right
        ])
        * half_dims
    )

    # Rotate each vertex and add center position
    vertices = np.asarray([
        rotate(pos, heading) + box_center for pos in relative_positions
    ])
    return vertices


def get_outlines(corners, extrinsic, intrinsic, height, width):
    """
    Project 3D bounding box corners to 2D and create a mask.
    
    Projects 3D corner points to 2D image coordinates using camera parameters,
    then creates a bounding box mask in the image space.
    
    Args:
        corners (np.ndarray): 3D corner points [N, 3]
        extrinsic (np.ndarray): Camera extrinsic matrix [4, 4]
        intrinsic (np.ndarray): Camera intrinsic matrix [3, 3]
        height (int): Image height in pixels
        width (int): Image width in pixels
        
    Returns:
        tuple: (mask, bounds) where:
            - mask (np.ndarray): Binary mask [height, width] indicating bounding box region
            - bounds (list): [y_min, y_max, x_min, x_max] bounding box coordinates
    """
    def generate_convex_hull(points):
        """Generate convex hull from 2D points."""
        hull = ConvexHull(points)
        return points[hull.vertices]

    def polygon_to_mask(polygon, height, width):
        """Convert polygon points to binary mask."""
        img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(img).polygon([tuple(p) for p in polygon], outline=1, fill=1)
        mask = np.array(img)
        return mask

    # Convert to homogeneous coordinates
    all_one = np.ones((corners.shape[0], 1))
    points = np.concatenate((corners, all_one), axis=1).T
    
    # Transform to camera coordinates
    cam_points = (np.linalg.inv(extrinsic) @ points)[:3]
    cam_points = cam_points / cam_points[2:]  # Normalize by Z
    
    # Project to image coordinates
    points = (intrinsic @ cam_points).T[:, :2]
    points[:, 0] = np.clip(points[:, 0], 0, width)
    points[:, 1] = np.clip(points[:, 1], 0, height)

    # Create bounding box mask with padding
    mask = np.zeros((height, width))
    points = points.astype(int)
    y_min = max(points[:, 1].min() - 50, 0)
    y_max = min(points[:, 1].max() + 50, height)
    x_min = max(points[:, 0].min() - 50, 0)
    x_max = min(points[:, 0].max() + 50, width)

    mask[y_min:y_max, x_min:x_max] = 1
    return mask, [y_min, y_max, x_min, x_max]


def get_attributes_for_one_car(car, extrinsic, intrinsic):
    """
    Get 2D image coordinates and depth for a single vehicle.
    
    Projects a 3D vehicle position to 2D image coordinates and calculates
    the depth from the camera.
    
    Args:
        car (dict): Vehicle information with cx, cy, cz coordinates
        extrinsic (np.ndarray): Camera extrinsic matrix [4, 4]
        intrinsic (np.ndarray): Camera intrinsic matrix [3, 3]
        
    Returns:
        dict: Dictionary containing:
            - 'u': Horizontal pixel coordinate
            - 'v': Vertical pixel coordinate  
            - 'depth': Distance from camera
    """
    x = car["cx"]
    y = car["cy"]
    z = car["cz"]
    one_point = np.array([[x, y, z]])
    
    # Convert to homogeneous coordinates
    all_one = np.ones((one_point.shape[0], 1))
    points = np.concatenate((one_point, all_one), axis=1).T
    
    # Transform to camera coordinates
    cam_points = (np.linalg.inv(extrinsic) @ points)[:3]
    cam_points_without_norm = copy.copy(cam_points)
    cam_points = cam_points / cam_points[2:]  # Normalize by Z
    
    # Project to image coordinates
    points = (intrinsic @ cam_points).T[:, :2]
    
    return {
        "u": points[0, 0],
        "v": points[0, 1],
        "depth": cam_points_without_norm[-1, 0],
    }


def scale_dense_depth_map(dense_depth_map, sparse_depth_map, depth_map_mask):
    """
    Scale dense depth map to match sparse depth measurements.
    
    Uses least squares to find the optimal scaling factor that aligns
    dense depth predictions with sparse ground truth depth measurements.
    
    Args:
        dense_depth_map (np.ndarray): Dense depth predictions [H, W]
        sparse_depth_map (np.ndarray): Sparse depth ground truth [H, W]
        depth_map_mask (np.ndarray): Binary mask for valid sparse points [H, W]
        
    Returns:
        torch.Tensor: Scaled dense depth map with same shape as input
    """
    # Ensure all inputs are PyTorch tensors
    dense_depth_map = torch.tensor(dense_depth_map, dtype=torch.float32)
    sparse_depth_map = torch.tensor(sparse_depth_map, dtype=torch.float32)
    depth_map_mask = torch.tensor(depth_map_mask, dtype=torch.float32)
    
    # Extract valid points from both maps
    valid_dense_depths = torch.masked_select(dense_depth_map, depth_map_mask.bool())
    valid_sparse_depths = torch.masked_select(sparse_depth_map, depth_map_mask.bool())
    
    # Compute optimal scaling factor using least squares
    alpha_numerator = torch.sum(valid_dense_depths * valid_sparse_depths)
    alpha_denominator = torch.sum(valid_dense_depths ** 2)
    alpha = alpha_numerator / alpha_denominator
    print('Scaling factor:', alpha)
    
    # Apply scaling to dense depth map
    scaled_dense_depth_map = alpha * dense_depth_map
    
    return scaled_dense_depth_map


def srgb_gamma_correction(linear_image):
    """
    Apply sRGB gamma correction to linear RGB values.
    
    Converts linear RGB values (0-1 range) to sRGB gamma-corrected values
    using the standard sRGB transfer function.
    
    Args:
        linear_image (np.ndarray): Linear RGB image [H, W, 3] with values in [0, 1]
        
    Returns:
        np.ndarray: Gamma-corrected sRGB image [H, W, 3]
    """
    linear_image = np.clip(linear_image, 0, 1)  # Clamp to valid range
    gamma_corrected_image = np.where(
        linear_image <= 0.0031308,
        linear_image * 12.92,  # Linear segment
        1.055 * (linear_image ** (1 / 2.4)) - 0.055  # Gamma segment
    )
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 1)
    return gamma_corrected_image


def srgb_inv_gamma_correction(gamma_corrected_image):
    """
    Apply inverse sRGB gamma correction to convert sRGB to linear.
    
    Converts sRGB gamma-corrected values back to linear RGB values
    using the inverse sRGB transfer function.
    
    Args:
        gamma_corrected_image (np.ndarray): sRGB image [H, W, 3] with values in [0, 1]
        
    Returns:
        np.ndarray: Linear RGB image [H, W, 3]
    """
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 1)
    linear_image = np.where(
        gamma_corrected_image <= 0.04045,
        gamma_corrected_image / 12.92,  # Linear segment
        ((gamma_corrected_image + 0.055) / 1.055) ** 2.4  # Gamma segment
    )
    return linear_image


def parse_config(path_to_json):
    """
    Parse JSON configuration file into a Struct object.
    
    Args:
        path_to_json (str): Path to JSON configuration file
        
    Returns:
        Struct: Configuration data accessible via dot notation
    """
    with open(path_to_json) as f:
        data = json.load(f)
        args = Struct(**data)
    return args


def blending_hdr_sky(nerf_env_panorama, sky_dome_panorama, nerf_last_trans, sky_mask):
    """
    Blend HDR sky dome with NeRF environment panorama.
    
    Combines NeRF-rendered environment with HDR sky dome using transparency
    values and sky mask for seamless integration.
    
    Args:
        nerf_env_panorama (np.ndarray): NeRF environment [H1, W1, 3] in linear space
        sky_dome_panorama (np.ndarray): HDR sky dome [H2, W2, 3] in linear space
        nerf_last_trans (np.ndarray): Transparency values [H1, W1, 1] range (0-1)
        sky_mask (np.ndarray): Sky region mask
        
    Returns:
        np.ndarray: Blended HDR sky [max(H1,H2), max(W1,W2), 3]
    """
    H, W, _ = sky_dome_panorama.shape
    
    # Resize inputs to match sky dome dimensions
    sky_mask = cv2.resize(sky_mask, (W, H))[:, :, :1]
    nerf_env_panorama = cv2.resize(nerf_env_panorama, (W, H))
    nerf_last_trans = cv2.resize(nerf_last_trans, (W, H))[:, :, np.newaxis]

    # Ensure full sky transparency in sky regions
    nerf_last_trans[sky_mask > 255 * 0.5] = 1

    # Blend using transparency values
    final_hdr_sky = nerf_env_panorama + sky_dome_panorama * nerf_last_trans
    
    return final_hdr_sky


def skylatlong2world(u, v):
    """
    Convert lat-long map coordinates to 3D world direction vectors.
    
    Transforms 2D lat-long coordinates (u, v) to 3D direction vectors
    in world space for spherical environment mapping.
    
    Args:
        u (np.ndarray): Horizontal coordinate [0, 1]
        v (np.ndarray): Vertical coordinate [0, 1]
        
    Returns:
        np.ndarray: 3D direction vectors [N, 3] in world space
    """
    u = u * 2
    
    # Convert lat-long to spherical coordinates
    theta_latlong = np.pi * (u - 1)
    phi_latlong = np.pi * v / 2

    # Convert spherical to Cartesian coordinates
    x = np.sin(phi_latlong) * np.sin(theta_latlong)
    y = np.cos(phi_latlong)
    z = -np.sin(phi_latlong) * np.cos(theta_latlong)

    # Transform to world space (note coordinate system conversion)
    direction = np.concatenate((-z, -x, y), axis=1)
    return direction


def generate_rays(insert_x, insert_y, int_ext_path, nerf_exp_dir):
    """
    Generate rays for neural rendering from camera parameters.
    
    Creates ray origins and directions for rendering a scene from specified
    camera positions, using camera metadata from saved files.
    
    Args:
        insert_x (float): X-coordinate for ray origin
        insert_y (float): Y-coordinate for ray origin
        int_ext_path (str): Path to camera intrinsic/extrinsic parameters file
        nerf_exp_dir (str): Directory to save generated ray data
        
    Returns:
        None
    """
    # Define near and far bounds for ray marching
    near = 1e-2
    far = 1000.
    
    # Load camera parameters
    origin = np.array([insert_x, insert_y, 0.0])
    cam_meta = np.load(int_ext_path)
    extrinsic = cam_meta[:, :12].reshape(-1,3,4)
    
    # Normalize camera positions
    translation = extrinsic[:, :3, 3].copy()
    center = np.mean(translation, axis=0)
    bias = translation - center[None]
    radius = np.linalg.norm(bias, 2, -1, False).max()
    translation = (translation - center[None]) / radius
    extrinsic[:, :, 3] = translation
    origin = (origin - center) / radius

    # Generate ray directions for lat-long mapping
    dy = np.linspace(0, 1, 1280)
    dx = np.linspace(0, 1, 1280*4)
    u, v = np.meshgrid(dx, dy)
    u, v = u.ravel()[..., None], v.ravel()[..., None]

    rays_d = skylatlong2world(u, v)
    rays_o = np.tile(origin[None], (len(rays_d), 1))
    bounds = np.array([[near, far]]).repeat(len(rays_d), axis=0)

    # Save ray data
    np.save(os.path.join(nerf_exp_dir, 'rays_o.npy'), rays_o.astype(np.float32))
    np.save(os.path.join(nerf_exp_dir, 'rays_d.npy'), rays_d.astype(np.float32))
    np.save(os.path.join(nerf_exp_dir, 'bounds.npy'), bounds.astype(np.float32))


def getColorList():
    """
    Get predefined HSV color ranges for color detection.
    
    Returns a dictionary mapping color names to HSV range pairs for
    various colors including black, white, red, orange, yellow, green,
    cyan, blue, and purple.
    
    Returns:
        dict: Color name to [lower, upper] HSV range mapping
    """
    color_dict = collections.defaultdict(list)
    
    # Black color range
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    # White color range
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    # Red color ranges (wraps around HSV hue)
    lower_red1 = np.array([156, 43, 46])
    upper_red1 = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red1)
    color_list.append(upper_red1)
    dict['red'] = color_list

    lower_red2 = np.array([0, 43, 46])
    upper_red2 = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red2)
    color_list.append(upper_red2)
    dict['red2'] = color_list

    # Orange color range
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # Yellow color range
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # Green color range
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # Cyan color range
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # Blue color range
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # Purple color range
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list
 
    return color_dict


def get_color(frame):
    """
    Detect the dominant color in an RGB image using HSV color space.
    
    Analyzes the input image to determine the most prominent color from
    predefined color ranges. Uses HSV color space for robust color detection.
    
    Args:
        frame (np.ndarray): RGB image array [H, W, 3]
        
    Returns:
        str: Name of the dominant color, or None if no color detected
        
    Color detection includes: black, white, red, orange, yellow, green,
    cyan, blue, and purple.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    max_num = 0
    color = None
    color_dict = getColorList()
    
    # Check each color range
    for color_name, ranges in color_dict.items():
        mask = cv2.inRange(hsv, ranges[0], ranges[1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        
        # Count pixels in this color range
        mask_num = binary[binary == 255].shape[0]
        if mask_num > max_num:
            max_num = mask_num
            color = color_name
    
    return color


def interpolate_uniformly(track, num_points):
    """
    Uniformly interpolate a track to a specified number of points.
    
    Resamples a trajectory to have exactly num_points points distributed
    uniformly along the path length, using linear interpolation between
    existing points.
    
    Args:
        track (np.ndarray): Input trajectory [N, d] where N is points and d is dimensions
        num_points (int): Desired number of points in output
        
    Returns:
        np.ndarray: Uniformly interpolated track [num_points, d]
    """
    # Calculate cumulative distance along the track
    distances = np.cumsum(np.sqrt(np.sum(np.diff(track, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Include start point

    # Generate equally spaced distances
    max_distance = distances[-1]
    uniform_distances = np.linspace(0, max_distance, num_points)

    # Interpolate each dimension
    uniform_track = np.array([
        np.interp(uniform_distances, distances, track[:, dim])
        for dim in range(track.shape[1])
    ])

    return uniform_track.T