"""
Navigation utilities module providing geometric calculations and helper functions.

Copyright (c) 2024 SceneCrafter Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import time
import math
import numpy as np
from shapely.ops import unary_union
from SceneController.agents.navigation.waypoint import Transform,Waypoint
from shapely.geometry import Polygon, LineString, Point


def interpolate_transforms_by_steps(transforms, steps=5):
    """
    Interpolate a series of 4x4 transformation matrices by dividing each segment into equal steps.
    
    This function performs linear interpolation between consecutive transformation matrices,
    creating smooth intermediate poses. The interpolation is applied to both position (translation)
    and rotation components of the transforms.
    
    Args:
        transforms (list[np.ndarray]): List of 4x4 transformation matrices as numpy arrays.
            Each matrix represents a pose in homogeneous coordinates.
        steps (int, optional): Number of interpolation steps between each pair of transforms.
            Default is 5 steps per segment.
    
    Returns:
        list[np.ndarray]: List of interpolated 4x4 transformation matrices.
            The output includes all original transforms plus interpolated ones.
            
    Example:
        >>> transforms = [T1, T2, T3]  # Three 4x4 matrices
        >>> interpolated = interpolate_transforms_by_steps(transforms, steps=3)
        >>> # Returns 7 transforms: T1, T1.33, T1.66, T2, T2.33, T2.66, T3
    """
    interpolated_transforms = []
    interpolated_transforms.append(transforms[0])
    for i in range(len(transforms) - 1):
        T_i = transforms[i]
        T_next = transforms[i + 1]

        p_i = T_i[:3, 3]
        p_next = T_next[:3, 3]

        R_i = R.from_matrix(T_i[:3, :3])
        R_next = R.from_matrix(T_next[:3, :3])

        distance = np.linalg.norm(p_next - p_i)

        for i in range(steps):
            t = (i+1)/steps
            p_interpolated = (1 - t) * p_i + t * p_next
            R_interpolated = R_i.as_matrix() * (1 - t) + R_next.as_matrix() * t
            T_interpolated = np.eye(4)
            T_interpolated[:3, :3] = R_interpolated
            T_interpolated[:3, 3] = p_interpolated
            interpolated_transforms.append(T_interpolated)

    return interpolated_transforms

def interpolate_locations_by_steps(locations, steps=5):
    """
    Interpolate a series of 3D locations by dividing each segment into equal steps.
    
    This function performs linear interpolation between consecutive 3D points,
    creating smooth intermediate locations. Unlike interpolate_transforms_by_steps,
    this function only interpolates position vectors without rotation components.
    
    Args:
        locations (list[np.ndarray]): List of 3D position vectors as numpy arrays.
            Each element should be a 3-element array [x, y, z] or compatible.
        steps (int, optional): Number of interpolation steps between each pair of locations.
            Default is 5 steps per segment.
    
    Returns:
        list[np.ndarray]: List of interpolated 3D location vectors.
            The output includes all original locations plus interpolated ones.
    
    Example:
        >>> locations = [[0, 0, 0], [10, 0, 0], [10, 10, 0]]  # Three 3D points
        >>> interpolated = interpolate_locations_by_steps(locations, steps=2)
        >>> # Returns 5 points: [0,0,0], [5,0,0], [10,0,0], [10,5,0], [10,10,0]
    """
    interpolated_locations = []
    interpolated_locations.append(locations[0])
    for i in range(len(locations) - 1):
        p_i = locations[i]
        p_next = locations[i + 1]

        for j in range(steps):
            t = (j+1)/steps
            p_interpolated = (1 - t) * np.array(p_i) + t * np.array(p_next)
            interpolated_locations.append(p_interpolated)

    return interpolated_locations

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in 3D space.
    
    Args:
        point1 (array-like): First point coordinates [x, y, z] or compatible format.
        point2 (array-like): Second point coordinates [x, y, z] or compatible format.
    
    Returns:
        float: The Euclidean distance between the two points.
    
    Example:
        >>> calculate_distance([0, 0, 0], [3, 4, 0])
        5.0
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_rotation(point1, point2):
    """
    Calculate the 2D rotation (yaw angle) between two points in the XY plane.
    
    Computes the angle between the vector from point1 to point2 and the positive X-axis.
    The Z component is set to 0 as this is a 2D calculation.
    
    Args:
        point1 (array-like): Starting point coordinates [x, y, ...].
        point2 (array-like): Ending point coordinates [x, y, ...].
    
    Returns:
        list: Rotation vector [0, 0, yaw] where yaw is the angle in radians.
            Returns [0, 0, yaw] with yaw calculated as atan2(dy, dx).
    
    Example:
        >>> calculate_rotation([0, 0], [1, 1])
        [0, 0, 0.7853981633974483]  # 45 degrees in radians
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    yaw = math.atan2(dy, dx)
    return [0, 0, yaw]
    
def calculate_rotation_v2(point1, point2, default_rot):
    """
    Calculate the 2D rotation (yaw angle) between two points with threshold filtering.
    
    Similar to calculate_rotation but includes a minimum distance threshold to prevent
    noise-induced rotation changes when points are too close together.
    
    Args:
        point1 (array-like): Starting point coordinates [x, y, ...].
        point2 (array-like): Ending point coordinates [x, y, ...].
        default_rot (array-like): Default rotation vector [0, 0, yaw] to return when
            the distance between points is below threshold.
    
    Returns:
        list: Rotation vector [0, 0, yaw] where yaw is the angle in radians,
            or the default rotation if distance is below threshold (0.04 units).
    
    Example:
        >>> calculate_rotation_v2([0, 0], [0.01, 0.01], [0, 0, 1.57])
        [0, 0, 1.57]  # Returns default due to small distance
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    yaw = math.atan2(dy, dx)
    return [0, 0, yaw] if np.linalg.norm(np.array([dx,dy])) > 0.04 else default_rot



def calculate_local_global_angle(curr_pt, local_pt, global_gt):
    """
    Calculate the angle between local and global vectors from a common reference point.
    
    Computes the angle between the vector from current point to local point and
    the vector from current point to global ground truth point.
    
    Args:
        curr_pt (array-like): Current/reference point coordinates [x, y, ...].
        local_pt (array-like): Local point coordinates [x, y, ...].
        global_gt (array-like): Global ground truth point coordinates [x, y, ...].
    
    Returns:
        float: Angle in radians between the local and global vectors.
            Range: [0, π] radians.
    
    Example:
        >>> calculate_local_global_angle([0, 0], [1, 0], [0, 1])
        1.5707963267948966  # 90 degrees (π/2 radians)
    """
    vec_loc = np.array(local_pt) - np.array(curr_pt)
    vec_global = np.array(global_gt) - np.array(curr_pt)
    cos_angle = np.dot(vec_loc, vec_global) / (np.linalg.norm(vec_loc) * np.linalg.norm(vec_global))
    return np.arccos(cos_angle)

def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a specified distance ahead of a reference object.
    
    Determines whether a target is located within the forward-facing cone of a reference
    object and within the specified maximum distance. The check considers both distance
    and angular constraints (within 90 degrees of forward direction).
    
    Args:
        target_transform (Transform): Location and orientation of the target object.
            Must have .location.x and .location.y attributes.
        current_transform (Transform): Location and orientation of the reference object.
            Must have .location.x, .location.y, and .get_forward_vector() method.
        max_distance (float): Maximum allowed distance in meters.
    
    Returns:
        bool: True if target is within max_distance ahead of reference object,
            False otherwise.
    
    Algorithm:
        1. Calculate vector from reference to target
        2. Check if distance is within max_distance
        3. Check if angle between forward vector and target vector is < 90 degrees
    
    Example:
        >>> target = Transform(location=Location(x=5, y=0))
        >>> current = Transform(location=Location(x=0, y=0))
        >>> is_within_distance_ahead(target, current, 10.0)
        True  # Target is 5m ahead
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0

def is_within_distance(target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
    """
    Check if a target object is within specified distance and angular constraints from a reference object.
    
    Determines whether a target is located within a maximum distance and within a specified
    angular range relative to the reference object's orientation. This allows for flexible
    directional filtering (e.g., front cone, rear cone, side detection).
    
    Args:
        target_location (array-like): Target object coordinates [x, y] or compatible format.
        current_location (array-like): Reference object coordinates [x, y] or compatible format.
        orientation (float): Reference object orientation in radians (0 = forward along positive X-axis).
        max_distance (float): Maximum allowed distance in meters.
        d_angle_th_up (float): Upper angular threshold in degrees.
        d_angle_th_low (float, optional): Lower angular threshold in degrees. Default is 0.
    
    Returns:
        bool: True if target is within distance and angular constraints, False otherwise.
    
    Example:
        >>> # Check if target is within 10m and within 30 degrees of forward direction
        >>> is_within_distance([5, 0], [0, 0], 0, 10, 30, -30)
        True
        >>> # Check if target is behind within 45 degrees
        >>> is_within_distance([-5, 0], [0, 0], 0, 10, 225, 135)
        True
    """
    target_vector = np.array([target_location[0] - current_location[0], target_location[1] - current_location[1]])
    norm_target = np.linalg.norm(target_vector)

    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(orientation), math.sin(orientation)])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative distance and angle between target and reference locations.
    
    Calculates both the Euclidean distance and the angular offset between a target location
    and a reference location, considering the reference's orientation.
    
    Args:
        target_location (Location): Target object location with .x and .y attributes.
        current_location (Location): Reference object location with .x and .y attributes.
        orientation (float): Reference object orientation in radians (0 = forward along positive X-axis).
    
    Returns:
        tuple: (distance, angle) where:
            - distance (float): Euclidean distance between locations in meters
            - angle (float): Angular offset in degrees, range [0, 180]
                0 degrees = directly ahead, 90 degrees = perpendicular, 180 degrees = directly behind
    
    Example:
        >>> from carla import Location
        >>> target = Location(x=5, y=5)
        >>> current = Location(x=0, y=0)
        >>> compute_magnitude_angle(target, current, 0)
        (7.071, 45.0)  # Distance ~7.07m, 45 degrees to the right
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(orientation), math.sin(orientation)])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    """
    Calculate the 2D Euclidean distance from a waypoint to a vehicle.
    
    Computes the horizontal distance between a waypoint's location and a vehicle's
    transform location, ignoring the Z-axis (height) component.
    
    Args:
        waypoint (Waypoint): Navigation waypoint object with .transform.location attributes.
            Must have .transform.location.x and .transform.location.y attributes.
        vehicle_transform (Transform): Vehicle transform object with .location attributes.
            Must have .location.x and .location.y attributes.
    
    Returns:
        float: 2D Euclidean distance between waypoint and vehicle in meters.
    
    Example:
        >>> waypoint = Waypoint(transform=Transform(location=Location(x=10, y=5)))
        >>> vehicle = Transform(location=Location(x=0, y=0))
        >>> distance_vehicle(waypoint, vehicle)
        11.180339887498949  # ~11.18 meters

    """
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y

    return math.sqrt(x * x + y * y)

def distance_vehicle_by_loc(ego_loc, target_loc):
    """
    Calculate the 2D Euclidean distance between two location objects.
    
    Computes the horizontal distance between two location points, ignoring the Z-axis
    (height) component. This is a more direct version of distance_vehicle that works
    with location objects directly.
    
    Args:
        ego_loc (Location): First location object with .x and .y attributes.
        target_loc (Location): Second location object with .x and .y attributes.
    
    Returns:
        float: 2D Euclidean distance between the two locations in meters.
    
    Example:
        >>> from carla import Location
        >>> ego = Location(x=0, y=0)
        >>> target = Location(x=3, y=4)
        >>> distance_vehicle_by_loc(ego, target)
        5.0  # 3-4-5 triangle
    """
    x = ego_loc.x - target_loc.x
    y = ego_loc.y - target_loc.y

    return math.sqrt(x * x + y * y)

def vector(location_1, location_2):
    """
    Calculate the normalized unit vector between two 3D locations.
    
    Computes the direction vector from location_1 to location_2 and normalizes it
    to unit length. Handles edge cases with very small distances using machine epsilon.
    
    Args:
        location_1 (Location): Starting location with .x, .y, .z attributes.
        location_2 (Location): Ending location with .x, .y, .z attributes.
    
    Returns:
        list: Normalized 3D unit vector [x, y, z] from location_1 to location_2.
            Returns a zero vector if the distance is extremely small.
    
    Example:
        >>> from carla import Location
        >>> loc1 = Location(x=0, y=0, z=0)
        >>> loc2 = Location(x=3, y=0, z=4)
        >>> vector(loc1, loc2)
        [0.6, 0.0, 0.8]  # Normalized 3-4-5 vector
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def compute_distance(location_1, location_2):
    """
    Calculate the 3D Euclidean distance between two points with numerical stability.
    
    Computes the exact Euclidean distance between two 3D points using the standard
    distance formula. Includes machine epsilon to prevent numerical issues with very
    small distances.
    
    Args:
        location_1 (Location): First 3D point with .x, .y, .z attributes.
        location_2 (Location): Second 3D point with .x, .y, .z attributes.
    
    Returns:
        float: 3D Euclidean distance between the two points in meters.
            Returns a very small positive value (machine epsilon) if distance is zero.
    
    Formula:
        distance = sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²) + ε
    
    Example:
        >>> from carla import Location
        >>> loc1 = Location(x=1, y=2, z=3)
        >>> loc2 = Location(x=4, y=6, z=8)
        >>> compute_distance(loc1, loc2)
        7.0710678118654755  # sqrt(3² + 4² + 5²)
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def positive(num):
    """
    Return the input number if positive, otherwise return zero.
    
    A simple utility function that implements positive value clamping. Useful for
    ensuring non-negative values in control systems and distance calculations.
    
    Args:
        num (float or int): Numeric value to check and potentially clamp.
    
    Returns:
        float or int: The input value if positive (> 0), otherwise 0.
    
    Examples:
        >>> positive(5.5)
        5.5
        >>> positive(-3.2)
        0
        >>> positive(0)
        0
    """
    return num if num > 0.0 else 0.0

def get_bbox_corners(location, heading, size):
    """
    Calculate the 2D corner coordinates of a vehicle's bounding box.
    
    Computes the four corner points of a rectangular bounding box given the vehicle's
    center position, heading angle, and dimensions. The bounding box is oriented according
    to the vehicle's heading.
    
    Args:
        location (array-like): Vehicle center position [x, y] in world coordinates.
        heading (float): Vehicle heading/yaw angle in radians (0 = facing positive X-axis).
        size (array-like): Vehicle dimensions [length, width] in meters.
            Note: Order is [length, width] not [width, length] as indicated in parameter name.
    
    Returns:
        np.ndarray: Array of shape (4, 2) containing the four corner coordinates:
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] ordered as:
            [front-right, front-left, rear-left, rear-right] relative to vehicle heading.
    
    Coordinate System:
        - X-axis: Forward direction when heading=0
        - Y-axis: Left direction when heading=0
        - Origin: Vehicle center
        - Corners are calculated relative to vehicle center and then rotated/translated
    
    Example:
        >>> get_bbox_corners([0, 0], 0, [4.0, 2.0])  # 4m long, 2m wide
        array([[ 2. ,  1. ],   # front-right
               [-2. ,  1. ],   # front-left
               [-2. , -1. ],   # rear-left
               [ 2. , -1. ]])  # rear-right
    """
    x, y = location[:2]
    width, length = size[0], size[1]
    heading = -heading

    corners = np.array([
        [length/2, width/2],
        [-length/2, width/2],
        [-length/2, -width/2],
        [length/2, -width/2]
    ])

    rotation_matrix = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])
    
    rotated_corners = np.dot(corners, rotation_matrix)
    translated_corners = rotated_corners + np.array([x, y])
    
    return translated_corners
def calculate_relative_vector(v1, v2):
    """
    Calculate the relative position vector between two points.
    
    Computes the vector from v1 to v2, representing the displacement from the
    first point to the second point.
    
    Args:
        v1 (array-like): Starting point coordinates [x, y, ...] or compatible format.
        v2 (array-like): Ending point coordinates [x, y, ...] or compatible format.
    
    Returns:
        np.ndarray: Relative vector from v1 to v2 as a numpy array.
    
    Example:
        >>> calculate_relative_vector([1, 2], [4, 6])
        array([3, 4])  # Vector from [1,2] to [4,6]
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return v2 - v1

def calculate_movement_vector(current_position, previous_position):
    """
    Calculate the movement vector between two consecutive positions.
    
    Computes the displacement vector from a previous position to the current position,
    representing the movement that occurred between frames or time steps.
    
    Args:
        current_position (array-like): Current position coordinates [x, y, ...].
        previous_position (array-like): Previous position coordinates [x, y, ...].
    
    Returns:
        np.ndarray: Movement vector from previous to current position as numpy array.
            Positive values indicate movement in the positive coordinate direction.
    
    Example:
        >>> calculate_movement_vector([5, 3], [2, 1])
        array([3, 2])  # Moved 3 units in x, 2 units in y
    """
    current_position = np.array(current_position)
    previous_position = np.array(previous_position)
    movement_vector = current_position - previous_position
    return movement_vector

def calculate_angle_between_vectors(v1, v2):
    """
    Calculate the angle between two vectors in radians.
    
    Computes the smallest angle between two vectors using the dot product formula.
    Handles edge cases for zero vectors and ensures numerical stability.
    
    Args:
        v1 (array-like): First vector as array-like object [x, y, ...].
        v2 (array-like): Second vector as array-like object [x, y, ...].
    
    Returns:
        float: Angle between vectors in radians, range [0, π].
            Returns 0 if either vector has zero magnitude.
    
    Formula:
        angle = arccos((v1·v2) / (||v1|| * ||v2||))
    
    Example:
        >>> calculate_angle_between_vectors([1, 0], [0, 1])
        1.5707963267948966  # 90 degrees (π/2 radians)
        >>> calculate_angle_between_vectors([1, 0], [1, 0])
        0.0  # Same direction
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def calculate_angel_from_vector1_to_vector2(v1, v2):
    """
    Calculate the rotation angle from vector v1 to vector v2 in [0, 2π] range.
    
    Computes the directed angle from v1 to v2, determining both magnitude and direction
    of rotation. Uses cross product to determine rotation direction (clockwise/counter-clockwise).
    
    Args:
        v1 (array-like): First vector [x, y] or compatible 2D vector.
        v2 (array-like): Second vector [x, y] or compatible 2D vector.
    
    Returns:
        float: Rotation angle from v1 to v2 in radians, range [0, 2π].
            0 = same direction, π/2 = 90° counter-clockwise, π = opposite direction.
    
    Algorithm:
        1. Normalize both vectors to unit length
        2. Calculate dot product for base angle [0, π]
        3. Use cross product to determine rotation direction
        4. Adjust angle based on rotation direction
    
    Example:
        >>> calculate_angel_from_vector1_to_vector2([1, 0], [0, 1])
        1.5707963267948966  # 90 degrees (π/2 radians)
        >>> calculate_angel_from_vector1_to_vector2([1, 0], [-1, 0])
        3.141592653589793  # 180 degrees (π radians)
    
    Note:
        - Works with 2D vectors (x,y components)
        - Uses cross product sign to determine rotation direction
        - Returns angle in [0, 2π] range (not [-π, π])
        - Handles edge cases for zero vectors through normalization
    """
    # Normalize vectors to unit length for consistent angle calculation
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate dot product to get cosine of angle [0, π]
    dot_product = np.dot(v1_norm, v2_norm)
    
    # Clamp to prevent numerical errors from exceeding [-1, 1] range
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate base angle between vectors
    angle = np.arccos(dot_product)
    
    # Use cross product to determine rotation direction
    cross_product = np.cross(v1_norm, v2_norm)
    
    # Adjust angle based on rotation direction (positive z = counter-clockwise)
    if cross_product > 0:
        angle = 2 * np.pi - angle
    
    return angle
def is_collision(target_vehicle_loc,
                     target_vehicle_yaw,
                     target_vehicle_bbox,
                     reference_vehicle_loc,
                     reference_vehicle_yaw,
                     reference_vehicle_bbox):
        """
        Check if two vehicles are in collision based on their bounding boxes.
        
        Determines whether the bounding boxes of two vehicles intersect by calculating
        the 2D polygon corners for each vehicle and checking for geometric overlap.
        
        Args:
            target_vehicle_loc (tuple): Target vehicle location (x, y) coordinates.
            target_vehicle_yaw (float): Target vehicle yaw angle in degrees.
            target_vehicle_bbox (tuple): Target vehicle bounding box dimensions (length, width).
            reference_vehicle_loc (tuple): Reference vehicle location (x, y) coordinates.
            reference_vehicle_yaw (float): Reference vehicle yaw angle in degrees.
            reference_vehicle_bbox (tuple): Reference vehicle bounding box dimensions (length, width).
        
        Returns:
            bool: True if vehicle bounding boxes intersect (collision detected), False otherwise.
        
        Algorithm:
            1. Calculate 4 corner points for each vehicle's bounding box
            2. Create shapely Polygon objects for each vehicle
            3. Check if polygons intersect using shapely's intersection detection
        
        Example:
            >>> loc1, yaw1, bbox1 = (0, 0), 0, (4.5, 2.0)
            >>> loc2, yaw2, bbox2 = (2, 0), 0, (4.5, 2.0)
            >>> collision = is_collision(loc1, yaw1, bbox1, loc2, yaw2, bbox2)
            >>> print(f"Collision detected: {collision}")
        """
        # Calculate 4 corner points for each vehicle's bounding box
        corners1 = get_bbox_corners(target_vehicle_loc, target_vehicle_yaw, target_vehicle_bbox)
        corners2 = get_bbox_corners(reference_vehicle_loc, reference_vehicle_yaw, reference_vehicle_bbox)
        
        # Create shapely Polygon objects for collision detection
        polygon1 = Polygon(corners1)
        polygon2 = Polygon(corners2)
        
        return polygon1.intersects(polygon2)

def detect_route_interaction(test_path, reference_path, interaction_range_1=2.0, interaction_range_2=2.0):
    """
    Detect route interaction between two paths using shapely geometry library.
    
    Checks for spatial interaction between test and reference paths by creating
    buffer zones around paths and their endpoints. Considers both path buffers
    and endpoint buffers for comprehensive interaction detection.
    
    Args:
        test_path (list): Test path as list of (x, y) coordinate tuples.
        reference_path (list): Reference path as list of (x, y) coordinate tuples.
        interaction_range_1 (float, optional): Buffer radius for test path in meters. Defaults to 2.0.
        interaction_range_2 (float, optional): Buffer radius for reference path in meters. Defaults to 2.0.
    
    Returns:
        bool: True if paths interact (buffers intersect), False otherwise.
    
    Algorithm:
        1. Convert paths to LineString objects
        2. Create buffer zones around entire paths
        3. Create smaller buffer zones around start and end points
        4. Check all buffer combinations for intersection
    
    Example:
        >>> test = [(0, 0), (1, 1), (2, 2)]
        >>> ref = [(1, 0), (1, 1), (1, 2)]
        >>> interacts = detect_route_interaction(test, ref, 1.5, 1.5)
        >>> print(f"Paths interact: {interacts}")
    """
    if test_path is None or reference_path is None:
        return False
    if len(test_path) < 2 or len(reference_path) < 2:
        return False
    
    # Convert paths to LineString objects for geometric operations
    test_line = LineString(test_path)
    reference_line = LineString(reference_path)
    
    # Create buffer zones around entire paths
    test_buffer = test_line.buffer(interaction_range_1)
    reference_buffer = reference_line.buffer(interaction_range_2)
    
    # Create smaller buffer zones around start and end points
    test_start_buffer = Point(test_path[0]).buffer(interaction_range_1/2)
    test_end_buffer = Point(test_path[-1]).buffer(interaction_range_1/2)
    reference_start_buffer = Point(reference_path[0]).buffer(interaction_range_2/2)
    reference_end_buffer = Point(reference_path[-1]).buffer(interaction_range_2/2)
    
    # Check all buffer combinations for intersection
    if test_buffer.intersects(reference_buffer):
        return True
    if test_buffer.intersects(reference_start_buffer):
        return True
    if test_buffer.intersects(reference_end_buffer):
        return True
    if test_start_buffer.intersects(reference_buffer):
        return True
    if test_end_buffer.intersects(reference_buffer):
        return True
    if test_start_buffer.intersects(reference_start_buffer):
        return True
    if test_end_buffer.intersects(reference_end_buffer):
        return True
    
    return False

def build_transform_path_from_ego_pose_data(ego_pose_data):
    """
    Build a list of Transform objects from ego pose data.
    
    Converts ego vehicle pose data (typically from simulation logs or sensor data)
    into a list of Transform objects for navigation and path planning.
    
    Args:
        ego_pose_data (list): List of ego pose dictionaries, each containing:
            - 'location': dict with keys 'x', 'y', 'z' (float coordinates)
            - 'rotation': dict with keys 'pitch', 'yaw', 'roll' (float angles in degrees)
    
    Returns:
        list: List of Transform objects representing the ego vehicle's path.
            Each transform contains Location and Rotation for a pose along the path.
    
    Example:
        >>> ego_data = [
        ...     {'location': {'x': 0, 'y': 0, 'z': 0}, 'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0}},
        ...     {'location': {'x': 10, 'y': 0, 'z': 0}, 'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0}}
        ... ]
        >>> transforms = build_transform_path_from_ego_pose_data(ego_data)
        >>> len(transforms)
        2
    """
    transform_path = []
    for ego_pose in ego_pose_data:
        transform = Transform(ego_pose)
        transform_path.append(transform)
    return transform_path