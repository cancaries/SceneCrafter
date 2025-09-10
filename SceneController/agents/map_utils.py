#!/usr/bin/env python3
"""
Map utilities for path analysis and navigation.
"""

import math


def is_straight_path(polyline, tolerance=0.01):
    """
    Determine if a polyline represents a straight path.
    
    Checks if all intermediate points lie approximately on the line segment
    connecting the first and last points of the polyline.
    
    Args:
        polyline (list): List of 3D points [(x, y, z), ...] representing the path
        tolerance (float): Maximum perpendicular distance allowed for points from the line
        
    Returns:
        bool: True if all points lie within tolerance of the straight line, False otherwise
    """
    if len(polyline) < 3:
        return True  # With fewer than 3 points, consider it straight by default
    
    start = polyline[0]
    end = polyline[-1]
    
    # Check each intermediate point
    for point in polyline[1:-1]:
        if not is_point_on_line(start, end, point, tolerance):
            return False
    return True


def is_point_on_line(start, end, point, tolerance):
    """
    Check if a point lies approximately on a line segment defined by start and end points.
    
    Uses vector mathematics to determine if the perpendicular distance from the
    point to the line is within the specified tolerance.
    
    Args:
        start (tuple): Start point of the line segment (x, y, z)
        end (tuple): End point of the line segment (x, y, z)
        point (tuple): Point to test (x, y, z)
        tolerance (float): Maximum allowed perpendicular distance
    """
    x1, y1, _ = start
    x2, y2, _ = end
    x, y, _ = point
    
    # Handle vertical line case (infinite slope)
    if x2 - x1 == 0:
        return abs(x - x1) < tolerance
    
    # Calculate slope and intercept for the line equation: y = kx + b
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    
    # Calculate perpendicular distance from point to line
    return abs(y - (k * x + b)) < tolerance


def detect_turns(polyline, angle_threshold=10):
    """
    Detect turns in a polyline based on angle changes.
    
    Analyzes consecutive triplets of points to identify significant direction
    changes that indicate turns in the path.
    
    Args:
        polyline (list): List of 3D points [(x, y, z), ...] representing the path
        angle_threshold (float): Minimum angle in degrees to consider as a turn
        
    Returns:
        list: List of tuples [(index, direction), ...] where:
            - index: position in polyline where turn occurs
            - direction: 'left' or 'right' indicating turn direction
    """
    if len(polyline) < 3:
        return []
    
    turns = []
    
    # Analyze each triplet of consecutive points
    for i in range(1, len(polyline) - 1):
        p1 = polyline[i - 1]
        p2 = polyline[i]
        p3 = polyline[i + 1]
        
        # Calculate the angle between the two vectors
        angle = calculate_angle(p1, p2, p3)
        
        # Check if angle exceeds threshold for turn detection
        if angle > angle_threshold:
            direction = calculate_turn_direction(p1, p2, p3)
            turns.append((i, direction))
    
    return turns


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three consecutive points.
    
    Computes the angle formed by vectors p1->p2 and p2->p3 using the dot product formula.
    
    Args:
        p1 (tuple): First point (x, y, z)
        p2 (tuple): Middle point (x, y, z)
        p3 (tuple): Last point (x, y, z)
        
    Returns:
        float: Angle in degrees between the two vectors (0-180 degrees)
    """
    def vector(p1, p2):
        """Create vector from p1 to p2."""
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    def dot_product(v1, v2):
        """Calculate dot product of two 2D vectors."""
        return v1[0] * v2[0] + v1[1] * v2[1]
    
    def magnitude(v):
        """Calculate magnitude of a 2D vector."""
        return math.sqrt(v[0]**2 + v[1]**2)

    v1 = vector(p1, p2)
    v2 = vector(p2, p3)
    dot_prod = dot_product(v1, v2)
    mag_v1 = magnitude(v1)
    mag_v2 = magnitude(v2)
    
    # Handle edge case of zero-length vectors
    if mag_v1 == 0 or mag_v2 == 0:
        return 0
    
    # Calculate angle using dot product formula: cos(θ) = (v1·v2)/(|v1||v2|)
    cos_angle = dot_prod / (mag_v1 * mag_v2)
    
    # Clamp to handle floating point precision issues
    cos_angle = max(-1, min(1, cos_angle))
    
    angle = math.acos(cos_angle) * (180.0 / math.pi)  # Convert radians to degrees
    return angle


def calculate_turn_direction(p1, p2, p3):
    """
    Determine the direction of a turn between three consecutive points.
    
    Uses the cross product to determine if the turn is to the left or right.
    
    Args:
        p1 (tuple): First point (x, y, z)
        p2 (tuple): Middle point (x, y, z)
        p3 (tuple): Last point (x, y, z)
        
    Returns:
        str: Turn direction - 'left', 'right', or 'straight' if no turn
    """
    def vector(p1, p2):
        """Create vector from p1 to p2."""
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    v1 = vector(p1, p2)
    v2 = vector(p2, p3)
    
    # Use cross product to determine turn direction
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    
    if cross_product > 0:
        return "left"
    elif cross_product < 0:
        return "right"
    else:
        return "straight"


def is_complex_path(polyline, angle_threshold=10, segment_length_threshold=5):
    """
    Determine if a path is complex based on turns and segment lengths.
    
    A path is considered complex if it contains turns that create segments
    longer than the specified threshold.
    
    Args:
        polyline (list): List of 3D points [(x, y, z), ...] representing the path
        angle_threshold (float): Minimum angle in degrees to consider as a turn
        segment_length_threshold (int): Minimum segment length to consider significant
        
    Returns:
        bool: True if the path is complex, False otherwise
    """
    # Detect all turns in the polyline
    turns = detect_turns(polyline, angle_threshold)
    
    if not turns:
        return False
    
    # Split the polyline into segments based on turn locations
    segments = []
    current_segment = [polyline[0]]
    
    for i, direction in turns:
        current_segment.append(polyline[i])
        segments.append(current_segment)
        current_segment = [polyline[i]]
    
    current_segment.append(polyline[-1])
    segments.append(current_segment)
    
    # Check if any segment meets the length threshold
    for segment in segments:
        if len(segment) >= segment_length_threshold:
            return True
    
    return False


def calculate_direction(points):
    """
    Calculate the overall direction vector of a sequence of points.
    
    Computes the vector from the first point to the last point in the sequence.
    
    Args:
        points (list): List of 3D points [(x, y, z), ...]
        
    Returns:
        tuple: Direction vector (dx, dy) from first to last point
    """
    # Calculate direction vector from first to last point
    x1, y1, _ = points[0]
    x2, y2, _ = points[-1]
    return (x2 - x1, y2 - y1)


def calculate_turn_relation(end_direction, exit_direction):
    """
    Determine the relationship between two direction vectors.
    
    Analyzes the angle between two direction vectors to classify the turn
    relationship as straight, left turn, or right turn.
    
    Args:
        end_direction (tuple): Current direction vector (dx, dy)
        exit_direction (tuple): Next direction vector (dx, dy)
        
    Returns:
        str: Turn relationship classification:
            - 'LANEFOLLOW': Straight continuation (angle within ±45°)
            - 'TURNLEFT': Left turn (angle > 45°)
            - 'TURNRIGHT': Right turn (angle < -45°)
    """
    ex, ey = end_direction
    nx, ny = exit_direction
    
    # Calculate cross product and dot product for angle determination
    cross_product = ex * ny - ey * nx
    dot_product = ex * nx + ey * ny
    
    # Calculate signed angle using atan2 for full 360-degree range
    angle = math.atan2(cross_product, dot_product) * (180.0 / math.pi)
    
    # Classify the turn based on angle ranges
    if -45 <= angle <= 45:
        return 'LANEFOLLOW'  # Continue straight
    elif angle > 45:
        return 'TURNLEFT'    # Turn left
    else:
        return 'TURNRIGHT'   # Turn right