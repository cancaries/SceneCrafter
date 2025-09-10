"""
SceneCrafter PLY Shadow Generation Module

This module provides comprehensive functionality for generating shadow-enhanced PLY files
for 3D vehicle models. It creates multiple shadow variations by projecting vehicle points
onto shadow planes using various light directions, producing realistic shadow effects
for different lighting conditions in 3D scenes.

Copyright (c) 2024 SceneCrafter Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

import numpy as np
from plyfile import PlyData, PlyElement
import os
import copy

def generate_light_directions():
    """
    Generates a comprehensive dictionary of light directions for shadow generation.
    
    Creates 81 unique light directions by systematically varying x and y components
    while maintaining z as the primary direction. Each direction is normalized
    to unit length for consistent shadow calculations.
    
    Returns:
        Dictionary mapping string keys ("x_y") to normalized 3D light direction vectors
        
    Example:
        >>> light_dict = generate_light_directions()
        >>> light_dict["0.5_-0.25"]  # Returns normalized vector [0.5, -0.25, 1.0]
    """
    light_direction_dict = {}
    values = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]  # Systematic angle ranges

    for x in values:
        for y in values:
            z = 1.0
            direction = np.array([x, y, z])
            direction /= np.linalg.norm(direction)  # Normalize to unit vector
            light_direction_dict[f"{str(x)}_{str(y)}"] = direction

    return light_direction_dict

def downsample_xy(points, grid_size=0.1):
    """
    Downsamples points in the XY plane using grid-based sampling for efficiency.
    
    Reduces computational load by selecting representative points from grid cells
    in the XY plane, maintaining point cloud density while minimizing processing time.
    
    Args:
        points: Input point cloud as structured array with 'x', 'y', 'z' fields
        grid_size: Size of grid cells in XY plane (default: 0.1)
    
    Returns:
        List of indices for downsampled points
        
    Example:
        >>> indices = downsample_xy(points, grid_size=0.1)
        >>> efficient_points = points[indices]
    """
    grid_dict = {}
    downsampled_idx = []

    for i, point in enumerate(points):
        grid_x = int(point['x'] // grid_size)
        grid_y = int(point['y'] // grid_size)
        grid_key = (grid_x, grid_y)

        if grid_key not in grid_dict:
            grid_dict[grid_key] = True
            downsampled_idx.append(i)

    return downsampled_idx

def normalize_vector(v):
    """
    Normalizes a 3D vector to unit length.
    
    Args:
        v: Input vector as numpy array
    
    Returns:
        Normalized vector with unit length
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def add_shadow_points(original_path, output_path, light_direction=[0, 0, -1], shadow_plane_z=0.05, scale_factor=0.9):
    """
    Adds shadow points to a PLY file using ray-plane intersection projection.
    
    This function reads a PLY file containing 3D vehicle geometry, calculates shadow points
    by projecting existing vertices onto a shadow plane using the specified light direction,
    and creates a new PLY file containing both original and shadow vertices. Shadow points
    are positioned below the original geometry to create realistic shadow effects.
    
    Args:
        original_path: Path to the input PLY file containing vehicle geometry
        output_path: Path where the output PLY file with shadow points will be saved
        light_direction: 3D vector indicating the light source direction (default: [0, 0, -1])
        shadow_plane_z: Height of the shadow plane where shadows are cast (default: 0.05)
        scale_factor: Scaling factor applied to shadow point positions (default: 0.9)
    
    Returns:
        None. Creates a new PLY file at the specified output path.
    
    Example:
        >>> add_shadow_points("vehicle.ply", "vehicle_shadow.ply", 
        ...                   light_direction=[0.5, -0.5, -1], 
        ...                   shadow_plane_z=0.1, scale_factor=1.0)
    """
    # Read original PLY file
    plydata = PlyData.read(original_path)
    vertex = plydata['vertex']
    original_points = np.array([vertex['x'], vertex['y'], vertex['z']]).T
    bottom_points = vertex

    # Normalize light direction vector
    light_direction = normalize_vector(np.array(light_direction))

    # Calculate shadow point positions using ray-plane intersection
    shadow_points_loc = []
    for point in bottom_points:
        x, y, z = point['x'], point['y'], point['z']

        # Calculate intersection with shadow plane
        t = (shadow_plane_z - z) / light_direction[2]
        shadow_x = x + t * light_direction[0]
        shadow_y = y + t * light_direction[1]
        shadow_z = shadow_plane_z

        shadow_points_loc.append((shadow_x, shadow_y, shadow_z))

    # Create shadow points with scaled positions
    shadow_points = copy.deepcopy(bottom_points)
    shadow_points['x'] = np.array([loc[0]*scale_factor for loc in shadow_points_loc])
    shadow_points['y'] = np.array([loc[1]*scale_factor for loc in shadow_points_loc])
    shadow_points['z'] = np.array([loc[2] for loc in shadow_points_loc])
    
    # Downsample shadow points in XY plane for efficiency
    downsampled_idx = downsample_xy(shadow_points, grid_size=0.1)
    shadow_points = shadow_points[downsampled_idx]

    # Configure shadow point appearance parameters
    shadow_params = {
        'nx': 0.0,      # Normal vector x-component
        'ny': 0.0,      # Normal vector y-component  
        'nz': 1.0,      # Normal vector z-component (upward facing)
        'f_dc_0': -1.5, # Spherical harmonics coefficient 0 (dark)
        'f_dc_1': -1.5, # Spherical harmonics coefficient 1 (dark)
        'f_dc_2': -1.5, # Spherical harmonics coefficient 2 (dark)
        'opacity': -3.0, # Opacity value (semi-transparent)
        'scale_0': -2.5, # Scale parameter 0
        'scale_1': -2.5, # Scale parameter 1
        'scale_2': -2.5, # Scale parameter 2
        'rot_0': 1.0,   # Rotation quaternion w-component
        'rot_1': 0.0,   # Rotation quaternion x-component
        'rot_2': 0.0,   # Rotation quaternion y-component
        'rot_3': 0.0,   # Rotation quaternion z-component
    }

    # Initialize spherical harmonics coefficients
    for i in range(9):
        key = f'f_rest_{i}'
        shadow_params[key] = 0.0

    # Construct complete shadow point data
    shadow_data = []
    for i in range(len(shadow_points)):
        x, y, z = shadow_points['x'][i], shadow_points['y'][i], shadow_points['z'][i]
        row = (x, y, z, shadow_params['nx'], shadow_params['ny'], shadow_params['nz'], 
               shadow_params['f_dc_0'], shadow_params['f_dc_1'], shadow_params['f_dc_2'],
               shadow_params['f_rest_0'], shadow_params['f_rest_1'], shadow_params['f_rest_2'], 
               shadow_params['f_rest_3'], shadow_params['f_rest_4'], shadow_params['f_rest_5'], 
               shadow_params['f_rest_6'], shadow_params['f_rest_7'], shadow_params['f_rest_8'],
               shadow_params['opacity'], shadow_params['scale_0'], shadow_params['scale_1'], 
               shadow_params['scale_2'], shadow_params['rot_0'], shadow_params['rot_1'], 
               shadow_params['rot_2'], shadow_params['rot_3'])
        shadow_data.append(row)

    # Write new PLY file with shadow points
    new_elem = PlyElement.describe(np.array(shadow_data, dtype=vertex.dtype()), 'vertex')
    PlyData([new_elem], text=False).write(output_path)

if __name__ == "__main__":
    _original_model_path = 'path_to_your_models'  # Update with actual path
    output_path = 'path_to_your_models_w_shadow'
    
    # Get list of available 3D vehicle models
    gaussian_model_list = os.listdir(_original_model_path)
    model_name_list = [name.split('.')[0] for name in gaussian_model_list]
    
    # Generate comprehensive light direction variations
    light_direction_dict = generate_light_directions()
    
    # Process each vehicle model
    for model_name in model_name_list:
        # Copy original PLY file to output directory
        original_file = os.path.join(_original_model_path, f"{model_name}.ply")
        copied_file = os.path.join(output_path, f"{model_name}.ply")
        os.system(f'cp {original_file} {copied_file}')
        
        # Generate shadow variations for each light direction
        for light_key, light_vector in light_direction_dict.items():
            add_shadow_points(
                original_file,
                os.path.join(output_path, f"{model_name}_shadow_{light_key}.ply"),
                light_direction=light_vector,
                shadow_plane_z=0.05,
                scale_factor=1.0
            )