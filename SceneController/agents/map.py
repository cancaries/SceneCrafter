#!/usr/bin/env python3
"""
SceneCrafter Map Module

This module provides comprehensive map handling functionality for autonomous vehicle simulation.
It includes map feature management, path planning, waypoint generation, occupancy grid creation,
and graph-based routing for navigation in simulated environments.
"""
import enum
import json
import math
import numpy as np
import sys
import os

from sklearn import neighbors
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from map_utils import *
from scipy.spatial import KDTree
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy.spatial import cKDTree
from navigation.waypoint import Waypoint
from navigation.local_planner_behavior import RoadOption
from navigation.tools.misc import calculate_rotation, calculate_distance
import datetime
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from scipy.ndimage import rotate
from shapely.geometry import Polygon, Point
from PIL import Image, ImageDraw
from matplotlib.collections import LineCollection

def color_map(data, cmap):
    """
    Maps numerical values to colors for visualization purposes.
    
    This function creates a color mapping from numerical data values to a specified
    colormap, useful for visualizing continuous data like elevation, speed, or density.
    
    Args:
        data (np.ndarray): Input numerical data to be mapped to colors
        cmap (str): Name of the matplotlib colormap to use
        
    Returns:
        np.ndarray: Array of RGBA color values corresponding to input data
    """
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    cmo = plt.cm.get_cmap(cmap)
    cs, k = list(), 256/cmo.N
    
    for i in range(cmo.N):
        c = cmo(i)
        for j in range(int(i*k), int((i+1)*k)):
            cs.append(c)
    cs = np.array(cs)
    data = np.uint8(255*(data-dmin)/(dmax-dmin))
    
    return cs[data]


def convert_str2RoadOption(str):
    """
    Converts string representations of road options to RoadOption enum values.
    
    This utility function maps string descriptions of driving maneuvers to
    the corresponding RoadOption enum values used throughout the navigation system.
    
    Args:
        str (str): String representation of road option (e.g., 'LANEFOLLOW', 'TURNLEFT')
        
    Returns:
        RoadOption: Corresponding enum value for the road option
        
    Supported Options:
        - 'LANEFOLLOW': Continue following the current lane
        - 'TURNLEFT': Make a left turn
        - 'TURNRIGHT': Make a right turn
        - 'STRAIGHT': Continue straight
        - 'CHANGELANELEFT': Change to left lane
        - 'CHANGELANERIGHT': Change to right lane
        - 'INTURN': In the middle of a turn maneuver
        - Any other string: Returns RoadOption.VOID
    """
    if str == 'LANEFOLLOW':
        return RoadOption.LANEFOLLOW
    elif str == 'TURNLEFT':
        return RoadOption.TURNLEFT
    elif str == 'TURNRIGHT':
        return RoadOption.TURNRIGHT
    elif str == 'STRAIGHT':
        return RoadOption.STRAIGHT
    elif str == 'CHANGELANELEFT':
        return RoadOption.CHANGELANELEFT
    elif str == 'CHANGELANERIGHT':
        return RoadOption.CHANGELANERIGHT
    elif str == 'INTURN':
        return RoadOption.INTURN
    else:
        return RoadOption.VOID


def remove_close_points(points, threshold=2.5):
    """
    Removes closely spaced points from a list of 3D coordinates.
    
    This function uses a KD-tree to efficiently identify and remove points
    that are too close to each other, maintaining spatial diversity in point clouds.
    
    Args:
        points (list): List of 3D points as [x, y, z] coordinates
        threshold (float): Minimum distance threshold between points (default: 2.5m)
        
    Returns:
        list: Filtered list of points with minimum spacing enforced
        
    Algorithm:
        1. Build KD-tree for efficient nearest neighbor queries
        2. Iterate through points and keep only those with sufficient spacing
        3. Return filtered point list
    """
    points = np.array(points)
    tree = cKDTree(points)
    to_keep = []
    for i, point in enumerate(points):
        if all(tree.query(point, k=2)[0][1] >= threshold for j in to_keep):
            to_keep.append(i)
    return points[to_keep].tolist()


class MapFeature:
    """
    Represents a single map feature such as a lane, road edge, or crosswalk.
    
    This class encapsulates all properties and relationships for individual
    map features loaded from the Waymo Open Dataset format.
    
    Attributes:
        feature_id (str): Unique identifier for this feature
        feature_type (str): Type of feature ('lane', 'road_edge', 'crosswalk', etc.)
        polyline (list): List of 3D points defining the feature geometry
        is_junction (bool): Whether this feature is part of a junction/intersection
        road_edge_type (str): Type of road edge (if applicable)
        lane_type (int): Type classification for lanes
        speed_limit_mph (float): Speed limit in miles per hour
        entry_lanes (list): List of lane IDs that can enter this feature
        exit_lanes (list): List of lane IDs that can be exited to from this feature
        left_neighbors (list): List of left neighboring features with relationship details
        right_neighbors (list): List of right neighboring features with relationship details
        neighbor_relations (dict): Dictionary mapping road options to connected features
    """
    
    def __init__(self, feature_id, feature_data):
        """
        Initialize a MapFeature from feature data dictionary.
        
        Args:
            feature_id (str): Unique identifier for this feature
            feature_data (dict): Dictionary containing all feature properties from dataset
        """
        self.feature_id = feature_id
        self.feature_type = feature_data['feature_type']
        self.polyline = feature_data.get('polyline', [])
        self.is_junction = feature_data.get('interpolating', False)
        self.road_edge_type = feature_data.get('road_edge_type', None)
        self.lane_type = feature_data.get('lane_type', None)
        self.speed_limit_mph = feature_data.get('speed_limit_mph', None)
        self.entry_lanes = feature_data.get('entry_lanes', [])
        self.exit_lanes = feature_data.get('exit_lanes', [])
        self.left_neighbors = feature_data.get('left_neighbors', [])
        self.right_neighbors = feature_data.get('right_neighbors', [])
        self.is_intersection = feature_data.get('is_intersection', False)
        
        # Initialize neighbor relations for different driving maneuvers
        self.neighbor_relations = {
            "CHANGELANELEFT": [],
            "CHANGELANERIGHT": [],
            "LANEFOLLOW": [],
            "TURNLEFT": [],
            "TURNRIGHT": [],
        }


class Map:
    """
    Main map class providing comprehensive navigation and planning functionality.
    
    This class handles map loading, feature management, path planning, waypoint generation,
    and provides all necessary interfaces for autonomous vehicle navigation simulation.
    """
    
    def __init__(self, map_data_path, map_mode='original'):
        """
        Initialize the Map with data from the specified path.
        
        Args:
            map_data_path (str): Path to the map feature JSON file
            map_mode (str): Mode for map loading ('original' or 'model')
                          
        The initialization process:
        1. Load map data from JSON file
        2. Build feature objects and spatial indices
        3. Establish neighbor relationships
        4. Build graph for path planning
        5. Generate spawn points
        6. Create occupancy grid
        """
        # Handle different map modes
        if map_mode == 'original':
            pass
        elif map_mode == 'model':
            map_data_path = map_data_path.replace('map_feature', 'map_feature_w_model')
            
        self.map_data_path = map_data_path
        self.scene_path = os.path.dirname(self.map_data_path)
        
        # Load map data from JSON
        with open(self.map_data_path, 'r') as f:
            map_data = json.load(f)
            
        # Initialize core data structures
        self.graph = nx.DiGraph()
        self.features = {}
        self.lane_points = []
        self.lane_indices = []
        self.map_boundary = [1000000, 1000000, -1000000, -1000000]  # [min_x, min_y, max_x, max_y]
        
        # Process features and build spatial indices
        for feature_id, feature_data in map_data.items():
            # Skip empty feature dictionaries
            if not feature_data:
                continue
                
            feature = MapFeature(feature_id, feature_data)
            self.features[feature_id] = feature
            
            # Collect lane points for spatial indexing
            if feature.feature_type == 'lane':
                if feature.lane_type == 1 or feature.lane_type == 2:
                    for i, point in enumerate(feature.polyline):
                        # Update map boundary
                        self.map_boundary[0] = min(self.map_boundary[0], point[0])
                        self.map_boundary[1] = min(self.map_boundary[1], point[1])
                        self.map_boundary[2] = max(self.map_boundary[2], point[0])
                        self.map_boundary[3] = max(self.map_boundary[3], point[1])
                        
                        # Add to spatial index
                        self.lane_points.append(point)
                        self.lane_indices.append((feature_id, i))
        
        # Build spatial index for efficient queries
        self.kd_tree = KDTree(self.lane_points)
        self.spawn_points = None
        self.spawn_distance = 10
        
        # Initialize map relationships and structures
        self.refine_relation()  # Establish neighbor relationships
        self.build_graph()      # Build path planning graph
        self.generate_spawn_points()  # Generate vehicle spawn points
        self.refine_spawn_points()    # Refine spawn points for quality
        self.build_occ_grid()   # Create occupancy grid

    def generate_spawn_points(self):
        """
        Generate spawn points along all lanes in the map at regular intervals.
        
        This method traverses all lane features and generates spawn points at
        specified intervals along each lane. These points serve as potential
        starting locations for vehicles in the simulation.
        
        The spawn distance is controlled by self.spawn_distance (default: 10m)
        """
        spawn_points = []
        
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                last_spawn_point = None
                
                # Sample points along the lane at intervals of 10 indices
                for i in range(0, len(feature.polyline) - 1, 10):
                    cur_spawn_point = feature.polyline[i]
                    
                    if last_spawn_point is not None:
                        distance = calculate_distance(last_spawn_point, cur_spawn_point)
                        if distance > self.spawn_distance:
                            spawn_points.append(cur_spawn_point)
                            last_spawn_point = cur_spawn_point
                    else:
                        # First point in the lane
                        spawn_points.append(cur_spawn_point)
                        last_spawn_point = cur_spawn_point
        
        self.spawn_points = spawn_points

    def refine_spawn_points(self, min_plan_length=20, ego_init_point=None, distance_thre=3.0):
        """
        Refine spawn points by removing those with insufficient planning paths.
        
        This method filters out spawn points that cannot generate sufficiently
        long navigation paths, ensuring high-quality starting locations for vehicles.
        
        Args:
            min_plan_length (int): Minimum required path length in waypoints
            ego_init_point (list, optional): Ego vehicle initial position to avoid
            distance_thre (float): Minimum distance from ego vehicle
        """
        # Remove closely spaced spawn points
        spawn_points_new = remove_close_points(self.spawn_points.copy(), distance_thre)
        
        # Remove points near ego vehicle if specified
        if ego_init_point is not None:
            ego_init_point = np.array(ego_init_point)
            spawn_points_new = [point for point in spawn_points_new 
                              if calculate_distance(point, ego_init_point) > distance_thre]
        
        self.spawn_points = spawn_points_new
        spawn_points_new = self.spawn_points.copy()
        
        # Test path generation capability for each spawn point
        for spawn_point in self.spawn_points:
            plan_waypoints = self.generate_overall_plan_waypoints(
                spawn_point, driving_mode='Random', ignore_lanechange=True)
            
            if plan_waypoints is None or len(plan_waypoints) < min_plan_length:
                spawn_points_new.remove(spawn_point)
        
        self.spawn_points = spawn_points_new

    def refine_spawn_points_w_distance(self, distance_thre=5.0):
        """
        Refine spawn points based solely on distance criteria.
        
        Args:
            distance_thre (float): Minimum distance threshold between points
            
        This simplified refinement only removes closely spaced points without
        testing path generation capabilities.
        """
        spawn_points_new = remove_close_points(self.spawn_points.copy(), distance_thre)
        self.spawn_points = spawn_points_new

    def refine_spawn_points_w_location(self, location, distance_thre=2.0):
        """
        Refine spawn points by removing those near a specified location.
        
        Args:
            location (list): 3D coordinates [x, y, z] to avoid
            distance_thre (float): Minimum distance from the location
        """
        location = np.array(location)
        spawn_points_new = self.spawn_points.copy()
        spawn_points_new = [point for point in spawn_points_new 
                          if calculate_distance(point, location) > distance_thre]
        self.spawn_points = spawn_points_new

    def get_spawn_points(self):
        """
        Get the list of refined spawn points.
        
        Returns:
            list: List of 3D spawn point coordinates [x, y, z]
        """
        return self.spawn_points

    def get_road_ends(self):
        """
        Get road endpoints that can serve as spawn locations.
        
        Returns:
            list: List of 3D coordinates for road endpoints
            
        This method identifies endpoints of lanes (excluding junctions) that
        can serve as alternative spawn locations, particularly useful for
        generating traffic at road boundaries.
        """
        road_ends = []
        
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane' and not feature.is_junction:
                if len(feature.polyline) > 20:
                    # Take last 20 points as potential endpoints
                    for i in range(1, 20):
                        road_ends.append(feature.polyline[-i])
                else:
                    continue
        
        # Remove closely spaced endpoints
        road_ends_new = remove_close_points(road_ends.copy(), 1.0)
        return road_ends

    def generate_spawn_points_w_ego_path(self, ego_path, distance_thre=6.0):
        """
        Generate spawn points alongside an ego vehicle's path.
        
        This method creates spawn points on the left side of an ego vehicle's
        path, useful for generating traffic that maintains safe distances.
        
        Args:
            ego_path (list): List of waypoints representing the ego vehicle path
            distance_thre (float): Lateral distance from ego path (default: 6.0m)
            
        Returns:
            list: Generated spawn points alongside the ego path
        """
        spawn_points = []
        
        for i in range(0, len(ego_path)):
            # Skip junction areas
            is_junction = self.features[ego_path[i].lane_id].is_junction
            if is_junction:
                continue
                
            cur_location = ego_path[i].transform.location
            yaw = ego_path[i].transform.rotation[2] * 180 / np.pi
            
            # Calculate left-side point using perpendicular offset
            left_point = [
                cur_location[0] + distance_thre * math.cos(math.radians(yaw + 90)),
                cur_location[1] + distance_thre * math.sin(math.radians(yaw + 90)),
                cur_location[2]
            ]
            
            # Get correct elevation from nearest lane point
            left_point[2] = self.get_close_z(left_point)
            
            # Ensure minimum spacing
            if len(spawn_points) == 0:
                spawn_points.append(left_point)
            elif calculate_distance(left_point, spawn_points[-1]) >= 10.0:
                spawn_points.append(left_point)
        
        self.spawn_points = spawn_points
        return self.spawn_points

    def find_nearest_lane_point(self, query_point):
        """
        Find the nearest lane point to a given query location.
        
        Args:
            query_point (list): 3D coordinates [x, y, z] to query
            
        Returns:
            tuple: (lane_id, point_index, distance) for the nearest lane point
            
        Uses KD-tree for efficient nearest neighbor search across all lane points.
        """
        distance, index = self.kd_tree.query(query_point[:3])
        lane_id, point_index = self.lane_indices[index]
        return lane_id, point_index, distance

    def get_left_neighbor_info(self, cur_lane_id, left_lane_id):
        """
        Get detailed information about a left neighboring lane.
        
        Args:
            cur_lane_id (str): Current lane identifier
            left_lane_id (str): Left neighbor lane identifier
            
        Returns:
            dict: Neighbor relationship details including start/end indices
        """
        return [x for x in self.features[cur_lane_id].left_neighbors 
                if x['feature_id'] == left_lane_id][0]

    def get_right_neighbor_info(self, cur_lane_id, right_lane_id):
        """
        Get detailed information about a right neighboring lane.
        
        Args:
            cur_lane_id (str): Current lane identifier
            right_lane_id (str): Right neighbor lane identifier
            
        Returns:
            dict: Neighbor relationship details including start/end indices
        """
        return [x for x in self.features[cur_lane_id].right_neighbors 
                if x['feature_id'] == right_lane_id][0]

    def get_left_neighbor_lane_ids(self, lane_id):
        """
        Get all left neighbor lane IDs for a given lane.
        
        Args:
            lane_id (str): Lane identifier to query
            
        Returns:
            list: List of left neighbor lane IDs
        """
        return [x['feature_id'] for x in self.features[lane_id].left_neighbors]

    def get_right_neighbor_lane_ids(self, lane_id):
        """
        Get all right neighbor lane IDs for a given lane.
        
        Args:
            lane_id (str): Lane identifier to query
            
        Returns:
            list: List of right neighbor lane IDs
        """
        return [x['feature_id'] for x in self.features[lane_id].right_neighbors]

    def get_all_neighbor_lane_ids(self, lane_id):
        """
        Get all neighboring lane IDs for a given lane.
        
        Args:
            lane_id (str): Lane identifier to query
            
        Returns:
            list: List of all neighbor lane IDs (left, right, and connected)
        """
        neighbor_lane_ids = []
        
        # Add left neighbors
        for neighbor in self.features[lane_id].left_neighbors:
            neighbor_lane_ids.append(neighbor['feature_id'])
            
        # Add right neighbors
        for neighbor in self.features[lane_id].right_neighbors:
            neighbor_lane_ids.append(neighbor['feature_id'])
            
        # Add connected lanes via neighbor relations
        for neighbor_relation in self.features[lane_id].neighbor_relations:
            neighbor_lane_ids += self.features[lane_id].neighbor_relations[neighbor_relation]
            
        return neighbor_lane_ids

    def get_all_junction_neighbor_lane_ids(self, lane_id):
        """
        Get all neighboring junction lane IDs for a given lane.
        
        Args:
            lane_id (str): Lane identifier to query
            
        Returns:
            list: List of junction neighbor lane IDs
        """
        neighbor_lane_ids = []
        
        # Check left neighbors
        for neighbor in self.features[lane_id].left_neighbors:
            if self.features[neighbor['feature_id']].is_junction:
                neighbor_lane_ids += [self.get_all_neighbor_lane_ids(neighbor['feature_id'])]
                
        # Check right neighbors
        for neighbor in self.features[lane_id].right_neighbors:
            if self.features[neighbor['feature_id']].is_junction:
                neighbor_lane_ids += [self.get_all_neighbor_lane_ids(neighbor['feature_id'])]
                
        # Check connected lanes
        for neighbor_relation in self.features[lane_id].neighbor_relations:
            for neighbor_id in self.features[lane_id].neighbor_relations[neighbor_relation]:
                if self.features[neighbor_id].is_junction:
                    neighbor_lane_ids += [self.get_all_neighbor_lane_ids(neighbor['feature_id'])]
                    
        return neighbor_lane_ids

    def build_waypoint_config(self, location, rotation, road_option=None, lane_id=None, lane_point_idx=None):
        """
        Build a waypoint configuration dictionary with all necessary parameters.
        
        Args:
            location (list): 3D coordinates [x, y, z] for the waypoint
            rotation (list): 3D rotation [roll, pitch, yaw] in radians
            road_option (RoadOption, optional): Type of road maneuver
            lane_id (str, optional): Lane identifier (auto-detected if None)
            lane_point_idx (int, optional): Index within lane polyline (auto-detected if None)
            
        Returns:
            dict: Complete waypoint configuration ready for Waypoint instantiation
        """
        # Auto-detect lane if not provided
        if lane_id is None or lane_point_idx is None:
            cur_lane_id, cur_lane_point_idx, _ = self.find_nearest_lane_point(location)
            lane_id = cur_lane_id
            lane_point_idx = cur_lane_point_idx
            
        # Default road option
        if road_option is None:
            road_option = RoadOption.VOID
            
        # Get lane feature details
        lane_feature = self.features[lane_id]
        is_junction = lane_feature.is_junction
        
        # Find valid left lane change targets
        left_lane_ids = lane_feature.neighbor_relations['CHANGELANELEFT']
        left_lane_valid = []
        for left_lane_id in left_lane_ids:
            neighbor_info = self.get_left_neighbor_info(lane_id, left_lane_id)
            self_start_idx = neighbor_info['self_start_index']
            self_end_idx = neighbor_info['self_end_index']
            if lane_point_idx >= self_start_idx and lane_point_idx <= self_end_idx - 2:
                left_lane_valid.append(left_lane_id)
                
        # Find valid right lane change targets
        right_lane_ids = lane_feature.neighbor_relations['CHANGELANERIGHT']
        right_lane_valid = []
        for right_lane_id in right_lane_ids:
            neighbor_info = self.get_right_neighbor_info(lane_id, right_lane_id)
            self_start_idx = neighbor_info['self_start_index']
            self_end_idx = neighbor_info['self_end_index']
            if lane_point_idx >= self_start_idx and lane_point_idx <= self_end_idx - 2:
                right_lane_valid.append(right_lane_id)
                
        return {
            'transform': {
                'location': location,
                'rotation': rotation
            },
            'road_option': road_option,
            'lane_id': lane_id,
            'lane_point_idx': lane_point_idx,
            'is_junction': is_junction,
            'left_lane': left_lane_valid,
            'right_lane': right_lane_valid
        }

    def build_waypoint(self, waypoint_config):
        """
        Create a Waypoint object from a configuration dictionary.
        
        Args:
            waypoint_config (dict): Configuration dictionary from build_waypoint_config
            
        Returns:
            Waypoint: Instantiated waypoint object
        """
        return Waypoint(waypoint_config)

    def get_location_by_lane_point(self, lane_point):
        """
        Get 3D coordinates from a lane point specification.
        
        Args:
            lane_point (tuple): (lane_id, point_index) tuple
            
        Returns:
            list: 3D coordinates [x, y, z] of the specified lane point
        """
        lane_id, point_index = lane_point
        return self.features[lane_id].polyline[point_index]

    def check_lanechange(self, current_lane_id, current_point_index, roadoption):
        """
        Check if a lane change maneuver is possible at the current position.
        
        Args:
            current_lane_id (str): Current lane identifier
            current_point_index (int): Current position index within the lane
            roadoption (str): Type of lane change ('CHANGELANELEFT' or 'CHANGELANERIGHT')
            
        Returns:
            bool: True if lane change is possible, False otherwise
        """
        if roadoption == 'CHANGELANELEFT':
            if self.features[current_lane_id].neighbor_relations['CHANGELANELEFT']:
                for target_lane_id in self.features[current_lane_id].neighbor_relations['CHANGELANELEFT']:
                    self_start_index = [x['self_start_index'] 
                                      for x in self.features[current_lane_id].left_neighbors 
                                      if x['feature_id'] == target_lane_id][0]
                    self_end_index = [x['self_end_index'] 
                                    for x in self.features[current_lane_id].left_neighbors 
                                    if x['feature_id'] == target_lane_id][0]
                    return current_point_index < self_end_index
            return False
            
        elif roadoption == 'CHANGELANERIGHT':
            if self.features[current_lane_id].neighbor_relations['CHANGELANERIGHT']:
                for target_lane_id in self.features[current_lane_id].neighbor_relations['CHANGELANERIGHT']:
                    self_start_index = [x['self_start_index'] 
                                      for x in self.features[current_lane_id].right_neighbors 
                                      if x['feature_id'] == target_lane_id][0]
                    self_end_index = [x['self_end_index'] 
                                    for x in self.features[current_lane_id].right_neighbors 
                                    if x['feature_id'] == target_lane_id][0]
                    return current_point_index < self_end_index
            return False
            
        return False

    def refine_plan_waypoints(self, plan_waypoints, min_distance=4.5):
        """
        Refine a waypoint path by removing redundant close points.
        
        Args:
            plan_waypoints (list): List of waypoints to refine
            min_distance (float): Minimum distance between consecutive waypoints
            
        Returns:
            list: Refined list of waypoints with close points removed
        """
        plan_waypoints_new = []
        lane_change_point_flag = False
        
        if plan_waypoints is None:
            return plan_waypoints
            
        for i, waypoint in enumerate(plan_waypoints):
            # Preserve lane change sequences
            if lane_change_point_flag:
                plan_waypoints_new.append(waypoint)
                lane_change_point_flag = False
                continue
                
            # Always keep the first waypoint
            if i == 0:
                plan_waypoints_new.append(waypoint)
            else:
                # Handle special maneuvers
                if waypoint.road_option != RoadOption.STRAIGHT:
                    if waypoint.road_option in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]:
                        plan_waypoints_new.append(waypoint)
                        lane_change_point_flag = True
                    else:
                        plan_waypoints_new.append(waypoint)
                else:
                    # Check distance for straight segments
                    distance = calculate_distance(
                        plan_waypoints_new[-1].transform.get_location(),
                        waypoint.transform.get_location())
                    if distance >= min_distance:
                        plan_waypoints_new.append(waypoint)
                        
        return plan_waypoints_new

    def get_plan_waypoints(self, current_location, driving_mode='Random', ignore_lanechange=True):
        """
        Generate planned waypoints from current location based on driving mode.
        
        Args:
            current_location (list): 3D coordinates [x, y, z] as starting point
            driving_mode (str): Driving mode ('LANEFOLLOW', 'TURNLEFT', 'TURNRIGHT', 
                             'CHANGELANELEFT', 'CHANGELANERIGHT', 'STRAIGHT', 'Random')
            ignore_lanechange (bool): Whether to ignore lane change maneuvers
            
        Returns:
            tuple: (driving_mode, plan_waypoints) where plan_waypoints is a list
                   of Waypoint objects for the planned path
        """
        current_lane_id, current_point_index, _ = self.find_nearest_lane_point(current_location)
        current_lane = self.features[current_lane_id]
        
        # Handle STRAIGHT mode
        if driving_mode == 'STRAIGHT':
            if current_point_index == len(current_lane.polyline) - 1:
                # Already at lane end
                return 'STRAIGHT', None
            else:
                plan_waypoints = []
                
                # Add remaining points in current lane
                for i in range(current_point_index, len(current_lane.polyline) - 1):
                    rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i + 1])
                    waypoint_config = self.build_waypoint_config(
                        current_lane.polyline[i], rotation, RoadOption.STRAIGHT,
                        current_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                
                # Add final point
                rotation = calculate_rotation(current_lane.polyline[-2], current_lane.polyline[-1])
                waypoint_config = self.build_waypoint_config(
                    current_lane.polyline[-1], rotation, RoadOption.STRAIGHT,
                    current_lane_id, len(current_lane.polyline) - 1)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                
                return 'STRAIGHT', plan_waypoints
                
        # Handle LANEFOLLOW mode
        elif driving_mode == 'LANEFOLLOW':
            if current_lane.neighbor_relations['LANEFOLLOW']:
                next_lane_id = current_lane.neighbor_relations['LANEFOLLOW'][0]
                next_lane = self.features[next_lane_id]
                plan_waypoints = []
                
                # Add remaining points in current lane
                for i in range(current_point_index, len(current_lane.polyline) - 1):
                    rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i + 1])
                    waypoint_config = self.build_waypoint_config(
                        current_lane.polyline[i], rotation, RoadOption.STRAIGHT,
                        current_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                
                if len(current_lane.polyline) < 2:
                    return 'LANEFOLLOW', None
                    
                # Add final point with LANEFOLLOW option
                rotation = calculate_rotation(current_lane.polyline[-2], current_lane.polyline[-1])
                waypoint_config = self.build_waypoint_config(
                    current_lane.polyline[-1], rotation, RoadOption.LANEFOLLOW,
                    current_lane_id, len(current_lane.polyline) - 1)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                
                # Add first point in next lane
                rotation = calculate_rotation(current_lane.polyline[-1], next_lane.polyline[0])
                waypoint_config = self.build_waypoint_config(
                    next_lane.polyline[0], rotation, RoadOption.STRAIGHT,
                    next_lane_id, 0)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                
                # Add remaining points in next lane
                for i in range(1, len(next_lane.polyline)):
                    rotation = calculate_rotation(next_lane.polyline[i - 1], next_lane.polyline[i])
                    waypoint_config = self.build_waypoint_config(
                        next_lane.polyline[i], rotation, RoadOption.STRAIGHT,
                        next_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                
                return 'LANEFOLLOW', plan_waypoints
            else:
                return 'LANEFOLLOW', None
                
        # Handle TURNLEFT mode
        elif driving_mode == 'TURNLEFT':
            if current_lane.neighbor_relations['TURNLEFT']:
                next_lane_id = current_lane.neighbor_relations['TURNLEFT'][0]
                next_lane = self.features[next_lane_id]
                plan_waypoints = []
                
                # Add remaining points in current lane
                for i in range(current_point_index, len(current_lane.polyline) - 1):
                    rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i + 1])
                    waypoint_config = self.build_waypoint_config(
                        current_lane.polyline[i], rotation, RoadOption.STRAIGHT,
                        current_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                
                # Add final point with TURNLEFT option
                rotation = calculate_rotation(current_lane.polyline[-2], current_lane.polyline[-1])
                waypoint_config = self.build_waypoint_config(
                    current_lane.polyline[-1], rotation, RoadOption.TURNLEFT,
                    current_lane_id, len(current_lane.polyline) - 1)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                
                # Add first point in next lane with INTURN option
                rotation = calculate_rotation(current_lane.polyline[-1], next_lane.polyline[0])
                waypoint_config = self.build_waypoint_config(
                    next_lane.polyline[0], rotation, RoadOption.INTURN,
                    next_lane_id, 0)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                
                # Add remaining points in next lane
                for i in range(1, len(next_lane.polyline)):
                    rotation = calculate_rotation(next_lane.polyline[i - 1], next_lane.polyline[i])
                    waypoint_config = self.build_waypoint_config(
                        next_lane.polyline[i], rotation, RoadOption.INTURN,
                        next_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                
                return 'TURNLEFT', plan_waypoints
            else:
                return 'TURNLEFT', None
                
        # Handle TURNRIGHT mode (similar to TURNLEFT)
        elif driving_mode == 'TURNRIGHT':
            if current_lane.neighbor_relations['TURNRIGHT']:
                next_lane_id = current_lane.neighbor_relations['TURNRIGHT'][0]
                next_lane = self.features[next_lane_id]
                plan_waypoints = []
                
                # Add remaining points in current lane
                for i in range(current_point_index, len(current_lane.polyline) - 1):
                    rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i + 1])
                    waypoint_config = self.build_waypoint_config(
                        current_lane.polyline[i], rotation, RoadOption.STRAIGHT,
                        current_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                
                # Add final point with TURNRIGHT option
                rotation = calculate_rotation(current_lane.polyline[-2], current_lane.polyline[-1])
                waypoint_config = self.build_waypoint_config(
                    current_lane.polyline[-1], rotation, RoadOption.TURNRIGHT,
                    current_lane_id, len(current_lane.polyline) - 1)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                
                # Add first point in next lane with INTURN option
                rotation = calculate_rotation(current_lane.polyline[-1], next_lane.polyline[0])
                waypoint_config = self.build_waypoint_config(
                    next_lane.polyline[0], rotation, RoadOption.INTURN,
                    next_lane_id, 0)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                
                # Add remaining points in next lane
                for i in range(1, len(next_lane.polyline)):
                    rotation = calculate_rotation(next_lane.polyline[i - 1], next_lane.polyline[i])
                    waypoint_config = self.build_waypoint_config(
                        next_lane.polyline[i], rotation, RoadOption.INTURN,
                        next_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                
                return 'TURNRIGHT', plan_waypoints
            else:
                return 'TURNRIGHT', None
                
        # Handle CHANGELANELEFT mode
        elif driving_mode == 'CHANGELANELEFT':
            if current_lane.neighbor_relations['CHANGELANELEFT']:
                next_lane_id = current_lane.neighbor_relations['CHANGELANELEFT'][0]
                next_lane = self.features[next_lane_id]
                
                # Get neighbor lane relationship details
                neighbor_info = self.get_left_neighbor_info(current_lane_id, next_lane_id)
                neighbor_start_idx = neighbor_info['neighbor_start_index']
                neighbor_end_idx = neighbor_info['neighbor_end_index']
                self_start_idx = neighbor_info['self_start_index']
                self_end_idx = neighbor_info['self_end_index']
                
                # Check if beyond valid change region
                if current_point_index > self_end_idx:
                    return 'CHANGELANELEFT', None
                    
                plan_waypoints = []
                
                # Handle pre-change region
                if current_point_index < self_start_idx:
                    # Add points up to change region
                    for i in range(current_point_index, self_start_idx):
                        rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i + 1])
                        waypoint_config = self.build_waypoint_config(
                            current_lane.polyline[i], rotation, RoadOption.STRAIGHT,
                            current_lane_id, i)
                        waypoint = Waypoint(waypoint_config)
                        plan_waypoints.append(waypoint)
                    
                    # Add change point
                    rotation = calculate_rotation(current_lane.polyline[self_start_idx - 1], 
                                                current_lane.polyline[self_start_idx])
                    waypoint_config = self.build_waypoint_config(
                        current_lane.polyline[self_start_idx], rotation, RoadOption.CHANGELANELEFT,
                        current_lane_id, self_start_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                    
                    # Add first point in target lane
                    rotation = calculate_rotation(current_lane.polyline[self_start_idx], 
                                                next_lane.polyline[neighbor_start_idx])
                    waypoint_config = self.build_waypoint_config(
                        next_lane.polyline[neighbor_start_idx], rotation, RoadOption.STRAIGHT,
                        next_lane_id, neighbor_start_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                    
                # Handle within change region
                elif current_point_index >= self_start_idx and current_point_index < self_end_idx:
                    idx_offset = current_point_index - self_start_idx
                    lanechange_idx = neighbor_start_idx + idx_offset
                    
                    # Ensure valid target index
                    if lanechange_idx + 2 <= neighbor_end_idx:
                        lanechange_idx += 2
                    
                    # Add transition point
                    rotation = calculate_rotation(current_lane.polyline[current_point_index], 
                                                next_lane.polyline[lanechange_idx - 1])
                    waypoint_config = self.build_waypoint_config(
                        next_lane.polyline[lanechange_idx - 1], rotation, RoadOption.CHANGELANELEFT,
                        next_lane_id, lanechange_idx - 1)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                    
                    # Add target lane point
                    rotation = calculate_rotation(current_lane.polyline[current_point_index], 
                                                next_lane.polyline[lanechange_idx])
                    waypoint_config = self.build_waypoint_config(
                        next_lane.polyline[lanechange_idx], rotation, RoadOption.STRAIGHT,
                        next_lane_id, lanechange_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                
                return 'CHANGELANELEFT', plan_waypoints
            else:
                return 'CHANGELANELEFT', None
                
        # Handle CHANGELANERIGHT mode (similar to CHANGELANELEFT)
        elif driving_mode == 'CHANGELANERIGHT':
            if current_lane.neighbor_relations['CHANGELANERIGHT']:
                next_lane_id = current_lane.neighbor_relations['CHANGELANERIGHT'][0]
                next_lane = self.features[next_lane_id]
                
                neighbor_info = self.get_right_neighbor_info(current_lane_id, next_lane_id)
                neighbor_start_idx = neighbor_info['neighbor_start_index']
                neighbor_end_idx = neighbor_info['neighbor_end_index']
                self_start_idx = neighbor_info['self_start_index']
                self_end_idx = neighbor_info['self_end_index']
                
                if current_point_index > self_end_idx:
                    return 'CHANGELANERIGHT', None
                    
                plan_waypoints = []
                
                # Handle pre-change region
                if current_point_index < self_start_idx:
                    for i in range(current_point_index, self_start_idx):
                        rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i + 1])
                        waypoint_config = self.build_waypoint_config(
                            current_lane.polyline[i], rotation, RoadOption.STRAIGHT,
                            current_lane_id, i)
                        waypoint = Waypoint(waypoint_config)
                        plan_waypoints.append(waypoint)
                    
                    # Add change point
                    rotation = calculate_rotation(current_lane.polyline[self_start_idx - 1], 
                                                current_lane.polyline[self_start_idx])
                    waypoint_config = self.build_waypoint_config(
                        current_lane.polyline[self_start_idx], rotation, RoadOption.CHANGELANERIGHT,
                        current_lane_id, self_start_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                    
                    # Add first point in target lane
                    rotation = calculate_rotation(current_lane.polyline[self_start_idx], 
                                                next_lane.polyline[neighbor_start_idx])
                    waypoint_config = self.build_waypoint_config(
                        next_lane.polyline[neighbor_start_idx], rotation, RoadOption.STRAIGHT,
                        next_lane_id, neighbor_start_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                
                # Handle within change region
                elif current_point_index >= self_start_idx and current_point_index < self_end_idx:
                    idx_offset = current_point_index - self_start_idx
                    lanechange_idx = neighbor_start_idx + idx_offset
                    
                    if lanechange_idx + 2 <= neighbor_end_idx:
                        lanechange_idx += 2
                    
                    # Add transition point
                    rotation = calculate_rotation(current_lane.polyline[current_point_index], 
                                                next_lane.polyline[lanechange_idx - 1])
                    waypoint_config = self.build_waypoint_config(
                        next_lane.polyline[lanechange_idx - 1], rotation, RoadOption.CHANGELANERIGHT,
                        next_lane_id, lanechange_idx - 1)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                    
                    # Add target lane point
                    rotation = calculate_rotation(current_lane.polyline[current_point_index], 
                                                next_lane.polyline[lanechange_idx])
                    waypoint_config = self.build_waypoint_config(
                        next_lane.polyline[lanechange_idx], rotation, RoadOption.STRAIGHT,
                        next_lane_id, lanechange_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                
                return 'CHANGELANERIGHT', plan_waypoints
            else:
                return 'CHANGELANERIGHT', None
                
        else:  # Random mode
            # Collect valid driving modes
            valid_driving_modes = []
            
            # Check lane change capabilities
            lane_change_left_flag = self.check_lanechange(current_lane_id, current_point_index, 'CHANGELANELEFT')
            lane_change_right_flag = self.check_lanechange(current_lane_id, current_point_index, 'CHANGELANERIGHT')
            
            # Basic driving modes
            basic_driving_modes = ['LANEFOLLOW', 'TURNLEFT', 'TURNRIGHT']
            
            if ignore_lanechange:
                # Only consider basic modes
                for mode in basic_driving_modes:
                    if self.features[current_lane_id].neighbor_relations[mode]:
                        valid_driving_modes.append(mode)
            else:
                # Include lane changes if possible
                allowed_driving_modes = basic_driving_modes.copy()
                if lane_change_left_flag:
                    allowed_driving_modes.append('CHANGELANELEFT')
                if lane_change_right_flag:
                    allowed_driving_modes.append('CHANGELANERIGHT')
                    
                for mode in allowed_driving_modes:
                    if self.features[current_lane_id].neighbor_relations[mode]:
                        valid_driving_modes.append(mode)
            
            # Select random valid mode or fallback to straight
            if valid_driving_modes:
                driving_mode = np.random.choice(valid_driving_modes)
                return self.get_plan_waypoints(current_location, driving_mode)
            else:
                driving_mode, plan_waypoints = self.get_plan_waypoints(current_location, 'STRAIGHT')
                if plan_waypoints is not None:
                    return 'STRAIGHT', plan_waypoints
                else:
                    return driving_mode, None

    def get_plan_waypoints_w_refine(self, current_location, driving_mode='Random', ignore_lanechange=True):
        """
        Generate planned waypoints with refinement for optimal spacing.
        
        Args:
            current_location (list): 3D coordinates [x, y, z] as starting point
            driving_mode (str): Driving mode for path generation
            ignore_lanechange (bool): Whether to ignore lane changes
            
        Returns:
            tuple: (driving_mode, refined_waypoints) with optimized waypoint spacing
        """
        driving_mode, plan_waypoints = self.get_plan_waypoints(
            current_location, driving_mode, ignore_lanechange)
            
        if plan_waypoints is not None:
            plan_waypoints = self.refine_plan_waypoints(plan_waypoints)
        else:
            return 'INVALID', None
            
        return driving_mode, plan_waypoints

    def generate_overall_plan_waypoints(self, current_location, driving_mode='Random', 
                                      ignore_lanechange=True, max_plan_length=5000, min_plan_length=200):
        """
        Generate extended path waypoints through iterative planning.
        
        This method continuously generates waypoints by selecting appropriate
        driving modes when the current path segment ends, creating long
        navigation paths for comprehensive route planning.
        
        Args:
            current_location (list): 3D coordinates [x, y, z] as starting point
            driving_mode (str): Initial driving mode
            ignore_lanechange (bool): Whether to ignore lane changes
            max_plan_length (int): Maximum number of waypoints to generate
            min_plan_length (int): Minimum required path length
            
        Returns:
            list: Extended list of waypoints forming a complete navigation path
        """
        plan_waypoints = None
        flag_first_plan = True
        
        while True:
            if flag_first_plan:
                # Initial path generation
                driving_mode, new_plan = self.get_plan_waypoints(
                    current_location, driving_mode, ignore_lanechange=ignore_lanechange)
                flag_first_plan = False
                plan_waypoints = new_plan
                
                # Handle lane change failures
                if (driving_mode in ['CHANGELANELEFT', 'CHANGELANERIGHT'] and 
                    plan_waypoints is None):
                    driving_mode, new_plan = self.get_plan_waypoints(
                        current_location, 'LANEFOLLOW', ignore_lanechange=ignore_lanechange)
                    plan_waypoints = new_plan
                
                if plan_waypoints is None:
                    break
            else:
                # Extend existing path
                driving_mode, new_plan = self.get_plan_waypoints(
                    plan_waypoints[-1].transform.get_location(),
                    driving_mode='Random', ignore_lanechange=ignore_lanechange)
                
                if new_plan is None:
                    break
                
                # Handle lane transitions
                last_point_in_current_plan = plan_waypoints[-1]
                first_point_in_new_plan = new_plan[0]
                
                if last_point_in_current_plan.lane_id != first_point_in_new_plan.lane_id:
                    plan_waypoints[-1].road_option = driving_mode
                
                # Extend path
                plan_waypoints += new_plan
            
            # Check termination conditions
            if plan_waypoints is None or len(plan_waypoints) > max_plan_length:
                break
        
        return plan_waypoints

    def extend_plan_waypoints(self, current_plan, driving_mode='STRAIGHT'):
        """
        Extend an existing plan with additional waypoints.
        
        Args:
            current_plan (list): Existing list of waypoints to extend
            driving_mode (str): Driving mode for extension
            
        Returns:
            list: Extended plan with additional waypoints
        """
        print('current plan:', len(current_plan))
        extend_plan = current_plan.copy()
        last_point = current_plan[-1]
        
        # Generate next segment
        driving_mode, new_plan = self.get_plan_waypoints(
            last_point.transform.get_location(), driving_mode)
        
        if new_plan is not None:
            extend_plan += new_plan
            
        # Generate additional random segment
        driving_mode, new_plan = self.get_plan_waypoints(
            new_plan[-1].transform.get_location(), 'Random')
        
        if new_plan is not None:
            extend_plan += new_plan
            
        return extend_plan

    def redefine_ego_road_option(self, current_plan, angle_thresh=0.02):
        """
        Redefine road options based on actual path curvature.
        
        Args:
            current_plan (list): List of waypoints to analyze
            angle_thresh (float): Angle change threshold for detecting turns
            
        This method analyzes the actual path curvature and updates road options
        to better reflect the real driving maneuvers required.
        """
        init_rot = current_plan[0].transform.rotation[-1]
        
        for wpt in current_plan[1:]:
            rot = wpt.transform.rotation[-1]
            if abs(rot - init_rot) > angle_thresh:
                wpt.road_option = RoadOption.INTURN
            init_rot = rot

    def generate_overall_plan_waypoints_w_refine(self, current_location, driving_mode='Random',
                                               ignore_lanechange=True, max_plan_length=5000, min_plan_length=200):
        """
        Generate refined extended path waypoints.
        
        Combines extended path generation with waypoint refinement for
        optimal path quality and waypoint spacing.
        
        Args:
            current_location (list): 3D coordinates [x, y, z] as starting point
            driving_mode (str): Initial driving mode
            ignore_lanechange (bool): Whether to ignore lane changes
            max_plan_length (int): Maximum path length in waypoints
            min_plan_length (int): Minimum required path length
            
        Returns:
            list: Refined extended path waypoints
        """
        plan_waypoints = None
        flag_first_plan = True
        
        while True:
            if flag_first_plan:
                driving_mode, new_plan = self.get_plan_waypoints(
                    current_location, driving_mode, ignore_lanechange=ignore_lanechange)
                flag_first_plan = False
                plan_waypoints = new_plan
                
                # Handle lane change failures
                if (driving_mode in ['CHANGELANELEFT', 'CHANGELANERIGHT'] and 
                    plan_waypoints is None):
                    driving_mode, new_plan = self.get_plan_waypoints(
                        current_location, 'LANEFOLLOW', ignore_lanechange=ignore_lanechange)
                    plan_waypoints = new_plan
                
                if plan_waypoints is None:
                    break
            else:
                # Continue extending path
                driving_mode, new_plan = self.get_plan_waypoints(
                    plan_waypoints[-1].transform.get_location(),
                    driving_mode='Random', ignore_lanechange=ignore_lanechange)
                
                if new_plan is None:
                    break
                
                # Handle lane transitions
                last_point_in_current_plan = plan_waypoints[-1]
                first_point_in_new_plan = new_plan[0]
                
                if last_point_in_current_plan.lane_id != first_point_in_new_plan.lane_id:
                    plan_waypoints[-1].road_option = driving_mode
                
                plan_waypoints += new_plan
            
            # Check termination conditions
            if plan_waypoints is None or len(plan_waypoints) > max_plan_length:
                break
        
        # Apply refinement
        if plan_waypoints is not None:
            plan_waypoints = self.refine_plan_waypoints(plan_waypoints)
        
        return plan_waypoints

    def build_graph(self):
        """
        Build a NetworkX graph representation of the map for path planning.
        
        This method constructs a directed graph where nodes are (lane_id, point_index)
        tuples and edges represent valid transitions between points. The graph
        includes:
        - Intra-lane connections (sequential points)
        - Inter-lane connections (lane transitions)
        - Lane change connections (left/right neighbors)
        - Turn connections (LANEFOLLOW, TURNLEFT, TURNRIGHT)
        
        Edge weights are based on Euclidean distance between points.
        """
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                polyline = feature.polyline
                
                # Add intra-lane edges (sequential points)
                for i in range(0, len(polyline) - 1):
                    start = polyline[i]
                    end = polyline[i + 1]
                    distance = calculate_distance(start, end)
                    self.graph.add_edge((feature_id, i), (feature_id, i + 1), weight=distance)
                
                # Add exit lane connections
                for exit_lane_id in feature.exit_lanes:
                    if exit_lane_id not in self.features.keys():
                        print(f"Exit lane {exit_lane_id} not found")
                        continue
                        
                    distance = calculate_distance(polyline[-1], 
                                                self.features[exit_lane_id].polyline[0])
                    
                    # Handle exact overlap
                    if distance < 0.01:
                        idx_offset = 1
                    else:
                        idx_offset = 0
                    
                    # Skip distant connections
                    if distance > 10:
                        continue
                    
                    self.graph.add_edge((feature_id, len(polyline) - 1 - idx_offset),
                                      (exit_lane_id, 0), weight=distance)
                
                # Add entry lane connections
                for entry_lane_id in feature.entry_lanes:
                    if entry_lane_id not in self.features.keys():
                        print(f"Entry lane {entry_lane_id} not found")
                        continue
                        
                    distance = calculate_distance(polyline[0], 
                                                self.features[entry_lane_id].polyline[-1])
                    
                    if distance < 0.01:
                        idx_offset = 1
                    else:
                        idx_offset = 0
                    
                    if distance > 10:
                        continue
                    
                    self.graph.add_edge((entry_lane_id, 
                                       len(self.features[entry_lane_id].polyline) - 1 - idx_offset),
                                      (feature_id, 0), weight=distance)
                
                # Add left lane change connections
                for neighbor_id in feature.neighbor_relations['CHANGELANELEFT']:
                    if neighbor_id not in self.features.keys():
                        print(f"Left neighbor {neighbor_id} not found")
                        continue
                        
                    neighbor = self.features[neighbor_id]
                    neighbor_info = [x for x in feature.left_neighbors 
                                   if x['feature_id'] == neighbor_id][0]
                    
                    neighbor_start_idx = neighbor_info['neighbor_start_index']
                    neighbor_end_idx = neighbor_info['neighbor_end_index']
                    self_start_idx = neighbor_info['self_start_index']
                    self_end_idx = neighbor_info['self_end_index']
                    
                    # Calculate lane offset for distance thresholds
                    lane_offset = calculate_distance(polyline[self_start_idx],
                                                   self.features[neighbor_id].polyline[neighbor_start_idx])
                    
                    # Define distance thresholds based on lane offset
                    low_threshold = math.sqrt(1 * 1 + lane_offset * lane_offset)
                    high_threshold = math.sqrt(4 * 4 + lane_offset * lane_offset)
                    
                    # Add connections within valid regions
                    for i in range(self_start_idx, self_end_idx):
                        for j in range(neighbor_start_idx, neighbor_end_idx):
                            distance = calculate_distance(polyline[i], neighbor.polyline[j])
                            
                            # Skip invalid distances
                            if distance < low_threshold or distance > high_threshold:
                                continue
                            
                            self.graph.add_edge((feature_id, i), (neighbor_id, j), weight=distance)
                            break  # Only add closest valid connection
                
                # Add right lane change connections (similar to left)
                for neighbor_id in feature.neighbor_relations['CHANGELANERIGHT']:
                    if neighbor_id not in self.features.keys():
                        print(f"Right neighbor {neighbor_id} not found")
                        continue
                        
                    neighbor = self.features[neighbor_id]
                    neighbor_info = [x for x in feature.right_neighbors 
                                   if x['feature_id'] == neighbor_id][0]
                    
                    neighbor_start_idx = neighbor_info['neighbor_start_index']
                    neighbor_end_idx = neighbor_info['neighbor_end_index']
                    self_start_idx = neighbor_info['self_start_index']
                    self_end_idx = neighbor_info['self_end_index']
                    
                    lane_offset = calculate_distance(polyline[self_start_idx],
                                                   self.features[neighbor_id].polyline[neighbor_start_idx])
                    
                    low_threshold = math.sqrt(1 * 1 + lane_offset * lane_offset)
                    high_threshold = math.sqrt(4 * 4 + lane_offset * lane_offset)
                    
                    for i in range(self_start_idx, self_end_idx):
                        for j in range(neighbor_start_idx, neighbor_end_idx):
                            distance = calculate_distance(polyline[i], neighbor.polyline[j])
                            
                            if distance < low_threshold or distance > high_threshold:
                                continue
                            
                            self.graph.add_edge((feature_id, i), (neighbor_id, j), weight=distance)
                            break
        
        # Remove self-loops from the graph
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))

    def build_occ_grid(self):
        """
        Build a 2D occupancy grid for collision detection and navigation.
        
        Creates a binary occupancy grid where 1 represents occupied space
        (road edges, obstacles) and 0 represents free space. The grid uses
        0.25m resolution for fine-grained collision detection.
        
        The grid covers the entire map area defined by map_boundary and
        includes road edges as occupied cells.
        
        Returns:
            np.ndarray: 2D occupancy grid array
            
        Visualization:
        The method also saves a visualization of the occupancy grid to:
        '/GPFS/public/junhaoge/workspace/SceneCrafter/data/occ/occ_grid.png'
        """
        # Grid resolution: 0.25m per cell
        min_x, min_y = self.map_boundary[0], self.map_boundary[1]
        max_x, max_y = self.map_boundary[2], self.map_boundary[3]
        
        # Load map features for occupancy calculation
        map_feature_path = self.map_data_path
        with open(map_feature_path, 'r') as f:
            map_feature = json.load(f)
        
        # Calculate grid dimensions
        occ_grid_width = int((max_x - min_x) / 0.25)
        occ_grid_height = int((max_y - min_y) / 0.25)
        
        # Initialize empty grid
        self.occ_grid = np.zeros((occ_grid_width, occ_grid_height))
        point_num = 0
        
        # Process road edges for occupancy
        for lane_id, lane_info in map_feature.items():
            if 'feature_type' not in lane_info.keys():
                continue
                
            if lane_info['feature_type'] == 'road_edge':
                polyline = lane_info['polyline']
                
                # Rasterize road edges into grid
                for i in range(0, len(polyline) - 1):
                    start = polyline[i]
                    end = polyline[i + 1]
                    distance = calculate_distance(start, end)
                    num_points = max(1, int(distance / 0.25))
                    
                    # Handle single point case
                    if num_points == 0:
                        x = int((start[0] - min_x) / 0.25)
                        y = int((start[1] - min_y) / 0.25)
                        try:
                            self.occ_grid[x, y] = 1
                            point_num += 1
                        except IndexError:
                            pass
                    
                    # Interpolate points along segment
                    for j in range(num_points):
                        x = int((start[0] + (end[0] - start[0]) * j / num_points - min_x) / 0.25)
                        y = int((start[1] + (end[1] - start[1]) * j / num_points - min_y) / 0.25)
                        try:
                            self.occ_grid[x, y] = 1
                            point_num += 1
                        except IndexError:
                            pass
        
        # Visualize occupancy grid
        # if self.occ_grid is not None:
        #     cmap = ListedColormap(['white', 'black'])
        #     plt.imshow(self.occ_grid, cmap=cmap)
        #     plt.title('Occupancy Grid')
        #     plt.xlabel('X (m)')
        #     plt.ylabel('Y (m)')
        #     plt.savefig('data/occ/occ_grid.png')
        #     plt.close()
        
        return self.occ_grid

    def get_occupancy_grid(self):
        """
        Get the current occupancy grid.
        
        Returns:
            np.ndarray: 2D occupancy grid where 1=occupied, 0=free
        """
        return self.occ_grid

    def get_occupancy_grid_w_loc(self, location, yaw, bbox):
        """
        Generate an occupancy grid centered on a vehicle's position and orientation.
        
        Creates a vehicle-centric occupancy grid that includes the map's static
        obstacles and the vehicle's own footprint. Useful for local planning.
        
        Args:
            location (list): Vehicle 2D position [x, y]
            yaw (float): Vehicle orientation in radians
            bbox (list): Vehicle bounding box dimensions [width, length]
            
        Returns:
            np.ndarray: Vehicle-centric occupancy grid
        """
        # Create empty grid matching map dimensions
        occ_grid = np.zeros_like(self.occ_grid)
        
        # Calculate vehicle center in grid coordinates
        x = int((location[0] - self.map_boundary[0]) / 0.25)
        y = int((location[1] - self.map_boundary[1]) / 0.25)
        x = min(max(x, 0), occ_grid.shape[1] - 1)
        y = min(max(y, 0), occ_grid.shape[0] - 1)
        car_bbox = np.array(bbox) / 2
        car_bbox = np.array([[car_bbox[0],car_bbox[1]],[-car_bbox[0],car_bbox[1]],[-car_bbox[0],-car_bbox[1]],[car_bbox[0],-car_bbox[1]],[car_bbox[0],car_bbox[1]]])
        car_bbox = np.dot(car_bbox, np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]))
        car_bbox += np.array([location[0], location[1]])
        car_bbox = np.array(car_bbox)

        img_size = (occ_grid.shape[1], occ_grid.shape[0])
        image = Image.new("L", img_size, 0)
        draw = ImageDraw.Draw(image)
        
        # Convert polygon coordinates to grid indices
        polygon_coords = [
            (int((p[1] - self.map_boundary[1]) / 0.25),
             int((p[0] - self.map_boundary[0]) / 0.25))
            for p in car_bbox
        ]
        draw.polygon(polygon_coords, fill=1)
        
        # Convert back to numpy array
        occ_grid = np.array(image)
        
        return occ_grid
    
    def plan_path(self, start_point, end_point):
        """
        Plan a path between two points using the map graph.
        
        Args:
            start_point (list): Starting position [x, y, z]
            end_point (list): Target position [x, y, z]
            
        Returns:
            list: Path as list of (lane_id, point_index) tuples
        """
        # Ensure 3D coordinates
        if len(start_point) == 2:
            start_point.append(0)
        if len(end_point) == 2:
            end_point.append(0)
            
        # Find nearest lane points
        start_lane_id, start_idx, _ = self.find_nearest_lane_point(start_point)
        end_lane_id, end_idx, _ = self.find_nearest_lane_point(end_point)
        
        try:
            # Use NetworkX shortest path algorithm
            path = nx.shortest_path(
                self.graph,
                (start_lane_id, start_idx),
                (end_lane_id, end_idx),
                weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            print("No path found between start and end points")
            return []
    
    def plan_waypoint_path(self, start_point, end_point):
        """
        Plan a waypoint-based path between two points.
        
        Args:
            start_point (list): Starting position [x, y, z]
            end_point (list): Target position [x, y, z]
            
        Returns:
            list: List of Waypoint objects representing the planned path
        """
        # Ensure 3D coordinates
        if len(start_point) == 2:
            start_point.append(0)
        if len(end_point) == 2:
            end_point.append(0)
            
        # Find nearest lane points
        start_lane_id, start_idx, _ = self.find_nearest_lane_point(start_point)
        end_lane_id, end_idx, _ = self.find_nearest_lane_point(end_point)
        
        try:
            # Get graph path
            path = nx.shortest_path(
                self.graph,
                (start_lane_id, start_idx),
                (end_lane_id, end_idx),
                weight='weight'
            )
            
            # Convert to waypoints
            waypoint_path = []
            for idx, node in enumerate(path):
                lane_id, point_index = node
                location = self.features[lane_id].polyline[point_index]
                
                # Calculate rotation based on path direction
                if point_index > 0:
                    prev_point = self.features[lane_id].polyline[point_index-1]
                    rotation = calculate_rotation(prev_point, location)
                else:
                    rotation = 0.0
                    
                waypoint_config = self.build_waypoint_config(
                    location, rotation, RoadOption.STRAIGHT, lane_id, point_index
                )
                waypoint = Waypoint(waypoint_config)
                waypoint_path.append(waypoint)
                
            return waypoint_path
            
        except nx.NetworkXNoPath:
            print("No path found between start and end points")
            return []
    
    def judge_roadoption(self, current_waypoint, next_waypoint):
        """
        Determine the navigation action between two consecutive waypoints.
        
        Analyzes the relationship between current and next waypoints to determine
        the appropriate driving maneuver (straight, turn, lane change, etc.).
        
        Args:
            current_waypoint (Waypoint): Current position and context
            next_waypoint (Waypoint): Next target position
            
        Returns:
            RoadOption: Navigation instruction for the transition
        """
        current_lane_id = current_waypoint.lane_id
        current_point_index = current_waypoint.lane_point_idx
        next_lane_id = next_waypoint.lane_id
        next_point_index = next_waypoint.lane_point_idx
        current_lane = self.features[current_lane_id]
        next_lane = self.features[next_lane_id]
        if current_lane_id == next_lane_id:
            return RoadOption.STRAIGHT
            
        # Check lane change options
        current_lane = self.features[current_lane_id]
        
        if next_lane_id in current_lane.neighbor_relations['CHANGELANELEFT']:
            return RoadOption.CHANGELANELEFT
        elif next_lane_id in current_lane.neighbor_relations['CHANGELANERIGHT']:
            return RoadOption.CHANGELANERIGHT
        elif next_lane_id in current_lane.neighbor_relations['LANEFOLLOW']:
            return RoadOption.LANEFOLLOW
        elif next_lane_id in current_lane.neighbor_relations['TURNLEFT']:
            return RoadOption.TURNLEFT
        elif next_lane_id in current_lane.neighbor_relations['TURNRIGHT']:
            return RoadOption.TURNRIGHT
            
        return RoadOption.STRAIGHT
    
    def judge_roadoption_w_lane_id(self, current_lane_id, next_lane_id):
        """
        Determine navigation action between lanes using lane IDs.
        
        Similar to judge_roadoption but operates on lane identifiers directly
        rather than waypoint objects.
        
        Args:
            current_lane_id (str): Current lane identifier
            next_lane_id (str): Target lane identifier
            
        Returns:
            RoadOption: Navigation instruction for the transition
        """
        # Same lane indicates straight movement
        if current_lane_id == next_lane_id:
            return RoadOption.STRAIGHT
            
        # Check lane relationships
        current_lane = self.features[current_lane_id]
        
        if next_lane_id in current_lane.neighbor_relations['CHANGELANELEFT']:
            return RoadOption.CHANGELANELEFT
        elif next_lane_id in current_lane.neighbor_relations['CHANGELANERIGHT']:
            return RoadOption.CHANGELANERIGHT
        elif next_lane_id in current_lane.neighbor_relations['LANEFOLLOW']:
            return RoadOption.LANEFOLLOW
        elif next_lane_id in current_lane.neighbor_relations['TURNLEFT']:
            return RoadOption.TURNLEFT
        elif next_lane_id in current_lane.neighbor_relations['TURNRIGHT']:
            return RoadOption.TURNRIGHT
            
        return RoadOption.STRAIGHT
    
    def plan_path_w_waypoints(self, start_waypoint, end_waypoint):
        """
        Plan a detailed waypoint path between two waypoints.
        
        Provides enhanced path planning with proper road option assignment
        for each segment of the path.
        
        Args:
            start_waypoint (Waypoint): Starting waypoint
            end_waypoint (Waypoint): Target waypoint
            
        Returns:
            list: List of Waypoint objects with appropriate road options
        """
        start_lane_id = start_waypoint.lane_id
        start_idx = start_waypoint.lane_point_idx
        end_lane_id = end_waypoint.lane_id
        end_idx = end_waypoint.lane_point_idx
        
        try:
            # Get graph path
            path = nx.shortest_path(
                self.graph,
                (start_lane_id, start_idx),
                (end_lane_id, end_idx),
                weight='weight'
            )
            
            # Convert to detailed waypoints
            plan_waypoints = []
            for idx, node in enumerate(path):
                lane_id, point_index = node
                location = self.features[lane_id].polyline[point_index]
                
                # Calculate rotation
                if point_index > 0:
                    prev_point = self.features[lane_id].polyline[point_index-1]
                    rotation = calculate_rotation(prev_point, location)
                else:
                    rotation = 0.0
                    
                waypoint_config = self.build_waypoint_config(
                    location, rotation, RoadOption.STRAIGHT, lane_id, point_index
                )
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
            
            # Assign appropriate road options to each waypoint
            for i in range(len(plan_waypoints) - 1):
                plan_waypoints[i].road_option = self.judge_roadoption(
                    plan_waypoints[i], plan_waypoints[i + 1]
                )
                
            # Refine the waypoint sequence
            if plan_waypoints:
                plan_waypoints = self.refine_plan_waypoints(plan_waypoints)
                
            return plan_waypoints
            
        except nx.NetworkXNoPath:
            print("No path found between waypoints")
            return []
    
    def refine_waypoint_path(self, waypoint_path):
        """
        Refine a waypoint path by simplifying redundant waypoints.
        
        Processes the waypoint sequence to remove unnecessary points while
        maintaining path validity. Also handles turn waypoint marking.
        
        Args:
            waypoint_path (list): Original waypoint sequence
            
        Returns:
            list: Refined waypoint sequence
        """
        refined_waypoint_path = []
        
        for idx, waypoint in enumerate(waypoint_path):
            # Handle turn waypoints
            if waypoint.road_option in [RoadOption.TURNLEFT, RoadOption.TURNRIGHT]:
                refined_waypoint_path.append(waypoint)
                
                # Mark subsequent straight segments as in-turn
                for i in range(idx + 1, len(waypoint_path)):
                    if waypoint_path[i].road_option == RoadOption.STRAIGHT:
                        waypoint_path[i].road_option = RoadOption.INTURN
                    else:
                        break
                        
            refined_waypoint_path.append(waypoint)
            
        return refined_waypoint_path
    
    def generate_waypoint_path_from_transform_path(self, transform_path):
        """
        Convert a transform-based path to waypoint-based path.
        
        Takes a sequence of transforms (position + rotation) and converts
        them into properly annotated waypoints with road options.
        
        Args:
            transform_path (list): List of transform objects with get_location() and get_rotation() methods
            
        Returns:
            list: List of Waypoint objects
        """
        waypoint_path = []
        
        for idx, transform in enumerate(transform_path):
            # Skip last point as there's no next point to calculate direction
            if idx == len(transform_path) - 1:
                break
                
            # Get current position and rotation
            location = transform.get_location()
            rotation = transform.get_rotation()
            
            # Find corresponding lane points
            cur_lane_id, cur_point_index, _ = self.find_nearest_lane_point(location)
            next_lane_id, next_point_index, _ = self.find_nearest_lane_point(
                transform_path[idx + 1].get_location()
            )
            
            # Determine road option for this segment
            road_option = self.judge_roadoption_w_lane_id(cur_lane_id, next_lane_id)
            
            # Create waypoint
            waypoint_config = self.build_waypoint_config(
                location, rotation, road_option, cur_lane_id, cur_point_index
            )
            waypoint = Waypoint(waypoint_config)
            waypoint_path.append(waypoint)
        
        # Refine the waypoint sequence
        refined_waypoint_path = self.refine_waypoint_path(waypoint_path)
        return refined_waypoint_path
    
    def generate_waypoint_path_from_two_points(self, cur_waypoint, next_waypoint, direction='CHANGELANERIGHT'):
        """
        Generate a waypoint path between two specific waypoints.
        
        Creates a simple path connecting two waypoints with specified direction.
        Useful for lane changes and specific maneuvers.
        
        Args:
            cur_waypoint (Waypoint): Current waypoint
            next_waypoint (Waypoint): Target waypoint
            direction (str): Direction of movement ('CHANGELANERIGHT', 'CHANGELANELEFT', etc.)
            
        Returns:
            tuple: (direction_string, waypoint_path) where waypoint_path is list of Waypoint objects
        """
        cur_lane_id = cur_waypoint.lane_id
        cur_point_index = cur_waypoint.lane_point_idx
        next_lane_id = next_waypoint.lane_id
        next_point_index = next_waypoint.lane_point_idx
        
        # Convert direction string to RoadOption
        roadoption = convert_str2RoadOption(direction)
        
        # Calculate initial rotation
        first_rotation = calculate_rotation(
            cur_waypoint.transform.get_location(),
            self.features[next_lane_id].polyline[next_point_index]
        )
        
        # Create waypoints
        plan_waypoints = []
        
        # First waypoint (current position with calculated rotation)
        first_waypoint_config = self.build_waypoint_config(
            cur_waypoint.transform.get_location(),
            first_rotation,
            roadoption,
            cur_lane_id,
            cur_point_index
        )
        first_waypoint = Waypoint(first_waypoint_config)
        plan_waypoints.append(first_waypoint)
        
        # Second waypoint (target position)
        next_waypoint_config = self.build_waypoint_config(
            next_waypoint.transform.get_location(),
            next_waypoint.transform.get_rotation(),
            RoadOption.STRAIGHT,
            next_lane_id,
            next_point_index
        )
        next_waypoint = Waypoint(next_waypoint_config)
        plan_waypoints.append(next_waypoint)
        
        return direction, plan_waypoints
    
    def get_waypoint_w_offset(self, waypoint, offset=1.0, direction=None):
        """
        Calculate a new waypoint offset from an existing waypoint.
        
        Args:
            waypoint (Waypoint): Reference waypoint
            offset (float): Lateral distance to offset (meters)
            direction (str, optional): Direction of offset ('left' or 'right')
                                     If None, determined from lane neighbors
            
        Returns:
            Waypoint: New waypoint at the offset position
        """
        location = waypoint.transform.get_location()
        yaw = waypoint.transform.get_rotation()[2]  # Extract yaw from rotation
        lane_id = waypoint.lane_id
        
        # Determine offset direction if not specified
        if direction is None:
            if self.features[lane_id].neighbor_relations['CHANGELANELEFT']:
                direction = 'left'
            elif self.features[lane_id].neighbor_relations['CHANGELANERIGHT']:
                direction = 'right'
        
        # Calculate offset angle
        if direction == 'left':
            yaw += np.pi / 2
        else:
            yaw -= np.pi / 2
            
        # Calculate new position
        x = location[0] + offset * math.cos(yaw)
        y = location[1] + offset * math.sin(yaw)
        
        # Create new waypoint
        waypoint_new = Waypoint(
            self.build_waypoint_config(
                [x, y, location[2]],
                waypoint.transform.get_rotation(),
                waypoint.road_option,
                waypoint.lane_id,
                waypoint.lane_point_idx
            )
        )
        return waypoint_new
    
    def get_close_z(self, location):
        """
        Find the closest valid Z-coordinate for a given 2D position.

        Args:
            location (list): 2D or 3D position [x, y] or [x, y, z]
            
        Returns:
            float: Appropriate Z-coordinate for the position
        """
        lane_id, point_index, _ = self.find_nearest_lane_point(location)
        return self.features[lane_id].polyline[point_index][2]
    
    def generate_overtake_path_from_reference_path(self, reference_path, direction='left', overtake_offset=1.5):
        """
        Generate an overtaking path parallel to a reference path.
        
        Creates a path that runs parallel to the reference path for overtaking
        maneuvers. The path follows the longest available straight segment
        before any turn.
        
        Args:
            reference_path (list): Original path as list of Waypoint objects
            direction (str): Direction of overtake ('left' or 'right')
            overtake_offset (float): Lateral distance for overtake path (meters)
            
        Returns:
            tuple: (status_string, overtake_path) where overtake_path is list of Waypoint objects
                   Returns ('INVALIDPATH', None) if no valid path can be generated
        """
        def calculate_point_w_offset(location, yaw, offset, direction='left'):
            """Helper function to calculate offset point."""
            if direction == 'left':
                yaw = yaw + np.pi / 2
            else:
                yaw = yaw - np.pi / 2
            x = location[0] + offset * math.cos(yaw)
            y = location[1] + offset * math.sin(yaw)
            return [x, y, location[2]]
        
        # Validate starting conditions
        if not reference_path or reference_path[0].is_junction:
            return 'INVALIDPATH', None
            
        # Find straight segment for overtaking
        reference_points = []
        for idx, waypoint in enumerate(reference_path):
            if idx == len(reference_path) - 1:
                break
            if waypoint.road_option != RoadOption.STRAIGHT:
                break
            reference_points.append(waypoint)
        
        # Check if segment is long enough
        if len(reference_points) <= 5:
            return 'INVALIDPATH', None
        
        # Generate overtake path
        overtake_path = []
        
        # First waypoint (transition to overtake lane)
        first_waypoint = reference_points[0]
        next_waypoint = reference_points[1]
        next_location_w_offset = calculate_point_w_offset(
            next_waypoint.transform.get_location(),
            next_waypoint.transform.get_rotation()[2],
            overtake_offset,
            direction
        )
        
        location = first_waypoint.transform.get_location()
        rotation = calculate_rotation(location, next_location_w_offset)
        
        first_waypoint_config = self.build_waypoint_config(
            location, rotation, RoadOption.CHANGELANELEFT,
            first_waypoint.lane_id, first_waypoint.lane_point_idx
        )
        first_waypoint = Waypoint(first_waypoint_config)
        overtake_path.append(first_waypoint)
        
        # Generate parallel waypoints along straight segment
        for idx, cur_waypoint in enumerate(reference_points):
            if idx == len(reference_points) - 2 or idx == 0:
                continue
                
            cur_location_w_offset = calculate_point_w_offset(
                cur_waypoint.transform.get_location(),
                cur_waypoint.transform.get_rotation()[2],
                overtake_offset,
                direction
            )
            
            next_location_w_offset = calculate_point_w_offset(
                reference_points[idx + 1].transform.get_location(),
                reference_points[idx + 1].transform.get_rotation()[2],
                overtake_offset,
                direction
            )
            
            cur_rotation = calculate_rotation(cur_location_w_offset, next_location_w_offset)
            cur_waypoint_config = self.build_waypoint_config(
                cur_location_w_offset, cur_rotation, RoadOption.STRAIGHT,
                cur_waypoint.lane_id, cur_waypoint.lane_point_idx
            )
            waypoint_offset = Waypoint(cur_waypoint_config)
            overtake_path.append(waypoint_offset)
        
        # Final transition back to original lane
        last_2_waypoint = reference_points[-2]
        last_waypoint = reference_points[-1]
        
        last_2_location_w_offset = calculate_point_w_offset(
            last_2_waypoint.transform.get_location(),
            last_2_waypoint.transform.get_rotation()[2],
            overtake_offset,
            direction
        )
        
        last_2_rotation = calculate_rotation(
            last_2_location_w_offset,
            last_waypoint.transform.get_location()
        )
        
        waypoint_config = self.build_waypoint_config(
            last_2_location_w_offset, last_2_rotation, RoadOption.CHANGELANERIGHT,
            last_2_waypoint.lane_id, last_2_waypoint.lane_point_idx
        )
        waypoint = Waypoint(waypoint_config)
        overtake_path.append(waypoint)
        overtake_path.append(reference_points[-1])
        
        return 'OVERTAKE', overtake_path
    
    def generate_turn_around_path(self, current_waypoint, turn_around_offset=2.5):
        """
        Generate a U-turn path from current position.
        
        Creates a path that performs a U-turn by crossing to the opposite
        direction lane. Useful for route reversal maneuvers.
        
        Args:
            current_waypoint (Waypoint): Current vehicle position
            turn_around_offset (float): Lateral distance for U-turn (meters)
            
        Returns:
            tuple: (status_string, turn_around_path) where turn_around_path is list of Waypoint objects
                   Returns ('INVALIDPATH', None) if no valid path can be generated
        """
        def calculate_point_w_offset(location, yaw, offset, direction='left'):
            """Helper function to calculate offset point."""
            if direction == 'left':
                yaw = yaw + np.pi / 2
            else:
                yaw = yaw - np.pi / 2
            x = location[0] + offset * math.cos(yaw)
            y = location[1] + offset * math.sin(yaw)
            return [x, y, location[2]]
        
        turn_around_path = []
        location = current_waypoint.transform.get_location()
        rotation = current_waypoint.transform.get_rotation()
        
        # Check if U-turn is possible
        current_lane_id = current_waypoint.lane_id
        current_lane = self.features[current_lane_id]
        
        # Cannot perform U-turn if left neighbors exist
        if current_lane.left_neighbors:
            return 'INVALIDPATH', None
        
        # Find opposite direction lane
        next_lane_id = None
        next_location_w_offset = location
        
        # Search for lane in opposite direction
        while True:
            next_location_w_offset = calculate_point_w_offset(
                next_location_w_offset,
                rotation[2],
                turn_around_offset,
                'left'
            )
            next_lane_id, next_point_index, _ = self.find_nearest_lane_point(
                next_location_w_offset
            )
            if next_lane_id != current_lane_id:
                break
        
        # Create transition waypoint
        transition_rotation = calculate_rotation(location, next_location_w_offset)
        waypoint_config = self.build_waypoint_config(
            location, transition_rotation, RoadOption.CHANGELANELEFT,
            current_waypoint.lane_id, current_waypoint.lane_point_idx
        )
        waypoint = Waypoint(waypoint_config)
        turn_around_path.append(waypoint)
        
        # Generate new route from opposite lane
        new_route = self.generate_overall_plan_waypoints_w_refine(next_location_w_offset)
        turn_around_path.extend(new_route)
        
        return 'TURNAROUND', turn_around_path
    
    def visualize_graph(self):
        """Generate a visual representation of the map graph."""
        pos = {}
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                for i, point in enumerate(feature.polyline):
                    pos[(feature_id, i)] = (point[0], point[1])
        
        plt.figure(figsize=(24, 16), dpi=200)
        nx.draw(self.graph, pos, with_labels=False, node_size=10, font_size=8)
        plt.savefig('graph.png')
        plt.close()
    
    def refine_relation(self):
        """
        Establish lane connectivity relationships for navigation.
        """
        for feature_id, feature in self.features.items():
            if feature.feature_type != 'lane':
                continue
                
            # Process exit lanes for turn relationships
            self_lane_end_direction = calculate_direction(feature.polyline[-4:])
            
            for exit_lane_id in feature.exit_lanes:
                if exit_lane_id not in self.features:
                    print(f"Exit lane {exit_lane_id} not found")
                    continue
                    
                exit_feature = self.features[exit_lane_id]
                
                # Check if lanes are connected
                distance = calculate_distance(
                    feature.polyline[-1],
                    exit_feature.polyline[0]
                )
                if distance > 10:  # Connection threshold
                    continue
                    
                # Determine turn relationship
                exit_direction = calculate_direction(exit_feature.polyline[-4:])
                relation = calculate_turn_relation(
                    self_lane_end_direction,
                    exit_direction
                )
                feature.neighbor_relations[relation].append(exit_lane_id)
            
            # Process left neighbors for lane change relationships
            for neighbor in feature.left_neighbors:
                lane_id = neighbor['feature_id']
                if lane_id not in self.features:
                    print(f"Left neighbor {lane_id} not found")
                    continue
                    
                neighbor_feature = self.features[lane_id]
                
                # Check neighbor validity
                if neighbor_feature.feature_type != 'lane':
                    continue
                    
                # Calculate neighbor length
                neighbor_length = neighbor['neighbor_end_index'] - neighbor['neighbor_start_index']
                if neighbor_length >= 20:  # Minimum length for lane change
                    feature.neighbor_relations['CHANGELANELEFT'].append(lane_id)
            
            # Process right neighbors for lane change relationships
            for neighbor in feature.right_neighbors:
                lane_id = neighbor['feature_id']
                if lane_id not in self.features:
                    print(f"Right neighbor {lane_id} not found")
                    continue
                    
                neighbor_feature = self.features[lane_id]
                
                # Check neighbor validity
                if neighbor_feature.feature_type != 'lane':
                    continue
                    
                # Calculate neighbor length
                neighbor_length = neighbor['neighbor_end_index'] - neighbor['neighbor_start_index']
                if neighbor_length >= 20:  # Minimum length for lane change
                    feature.neighbor_relations['CHANGELANERIGHT'].append(lane_id)
    
    def refine_junction(self):
        """
        Refine junction information across the map.
        
        Ensures consistent junction marking across connected lanes.
        If a lane is marked as a junction, ensures its connected neighbors
        are also appropriately marked.
        """
        for feature_id, feature in self.features.items():
            if feature.feature_type != 'lane' or not feature.is_junction:
                continue
                
            # Check if this is truly a junction area
            exact_junction = False
            neighbours = feature.left_neighbors + feature.right_neighbors
            
            for neighbor in neighbours:
                lane_id = neighbor['feature_id']
                if lane_id not in self.features:
                    continue
                    
                neighbor_feature = self.features[lane_id]
                if neighbor_feature.is_junction:
                    exact_junction = True
                    break
            
            if exact_junction:
                # Ensure connected non-junction lanes are marked as junction
                for neighbor in neighbours:
                    lane_id = neighbor['feature_id']
                    if lane_id not in self.features:
                        continue
                        
                    neighbor_feature = self.features[lane_id]
                    if not neighbor_feature.is_junction:
                        self.features[lane_id].is_junction = True
    
    def save_map_convertion(self, save_path):
        """
        Save map data in a simplified format for external use.
        
        Converts the internal map representation into a simplified JSON format
        suitable for external applications. Filters data based on ego vehicle
        trajectory to ensure relevance.
        
        Args:
            save_path (str): Directory path for saving converted map data
        """
        # Initialize data containers
        road_edges = []
        crosswalk = []
        road_line = []
        lane = []
        
        # Load ego pose data for filtering
        ego_pose_path = os.path.join(self.scene_path, 'ego_pose.json')
        
        with open(self.map_data_path, 'r') as f:
            map_feature = json.load(f)
        with open(ego_pose_path, 'r') as f:
            ego_pose = json.load(f)
        
        # Calculate ego trajectory bounds
        ego_z = [pose['location'][2] for pose in ego_pose]
        ego_xy = [pose['location'][:2] for pose in ego_pose]
        
        min_ego_z = min(ego_z) - 5.0
        max_ego_z = max(ego_z) + 5.0
        
        # Filter and process map features
        for lane_id, lane_info in map_feature.items():
            if 'road_edge_type' not in lane_info:
                continue
                
            # Process road edges
            if lane_info['feature_type'] == 'road_edge':
                lane_new = []
                for point in lane_info['polyline']:
                    if min_ego_z <= point[2] <= max_ego_z:
                        lane_new.append(point)
                if lane_new:
                    road_edges.append(lane_new)
            
            # Process crosswalks
            if lane_info.get('feature_type') == 'crosswalk':
                lane_new = []
                for point in lane_info['polyline']:
                    lane_new.append(point)
                if lane_new:
                    lane_new.append(lane_info['polyline'][0])  # Close polygon
                    crosswalk.append(lane_new)
            
            # Process road lines
            if lane_info.get('feature_type') == 'road_line':
                lane_new = []
                for point in lane_info['polyline']:
                    if min_ego_z <= point[2] <= max_ego_z:
                        lane_new.append(point)
                if lane_new:
                    road_line.append(lane_new)
            
            # Process lanes
            if lane_info.get('feature_type') == 'lane':
                lane_new = []
                for point in lane_info['polyline']:
                    if min_ego_z <= point[2] <= max_ego_z:
                        lane_new.append(point)
                if lane_new:
                    lane.append(lane_new)
        
        # Save converted data
        save_file_name = os.path.join(save_path, 'map_feature.json')
        map_feature_new = {
            'road_edges': road_edges,
            'crosswalk': crosswalk,
            'road_line': road_line,
            'lane': lane
        }
        
        with open(save_file_name, 'w') as f:
            json.dump(map_feature_new, f, indent=2)
    
    def draw_map(self):
        """Generate a basic map visualization."""
        plt.figure(figsize=(24, 16), dpi=200)
        
        road_edges = []
        lanes = []
        
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                lanes.append(feature.polyline)
            else:
                road_edges.append(feature.polyline)
        
        # Plot road edges in red
        for edge in road_edges:
            edge = np.array(edge)
            plt.plot(edge[:, 0], edge[:, 1], c='red')
        
        # Plot lanes in green
        for lane in lanes:
            lane = np.array(lane)
            plt.plot(lane[:, 0], lane[:, 1], c='green')
            
        plt.savefig('data/traffic_flow/map.png')
        plt.close()
    
    def draw_map_w_spawn_points(self):
        """Generate map visualization with spawn points marked."""
        plt.figure(figsize=(24, 16), dpi=200)
        
        road_edges = []
        lanes = []
        
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                lanes.append(feature.polyline)
            else:
                road_edges.append(feature.polyline)
        
        # Plot road edges in red
        for edge in road_edges:
            edge = np.array(edge)
            plt.plot(edge[:, 0], edge[:, 1], c='red')
        
        # Plot lanes in green
        for lane in lanes:
            lane = np.array(lane)
            plt.plot(lane[:, 0], lane[:, 1], c='green')
        
        # Mark spawn points in blue
        spawn_points = self.get_spawn_points()
        for point in spawn_points:
            plt.scatter(point[0], point[1], c='blue', s=10)
            
        plt.savefig('data/draw_pic/map_w_spawn_points.png')
        plt.close()
    
    def draw_map_w_traffic_flow(self, car_dict, text=''):
        """
        Generate map visualization with traffic flow.
        
        Creates a map image showing both the road network and current
        vehicle positions with their bounding boxes.
        
        Args:
            car_dict (dict): Dictionary of vehicle information
            text (str): Additional text for filename
        """
        save_folder = '/home/ubuntu/junhaoge/ChatSim/data/end2end_map_data/traffic_flow' + text
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        plt.figure(figsize=(24, 16), dpi=200)
        
        road_edges = []
        lanes = []
        
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                lanes.append(feature.polyline)
            else:
                road_edges.append(feature.polyline)
        
        # Plot road network
        for edge in road_edges:
            edge = np.array(edge)
            plt.plot(edge[:, 0], edge[:, 1], c='red')
        
        for lane in lanes:
            lane = np.array(lane)
            plt.plot(lane[:, 0], lane[:, 1], c='green')
        
        # Plot vehicle bounding boxes
        for car_name, car_info in car_dict.items():
            car_centre_loc = car_info['loc']  # [x, y, z]
            car_rot = car_info['rot'][2]  # yaw
            car_bbox = np.array(car_info['bbox'])  # [w, l]
            
            # Swap width and length for correct orientation
            car_bbox = np.array([car_bbox[1], car_bbox[0]]) / 2
            
            # Create bounding box polygon
            car_bbox_polygon = np.array([
                [car_bbox[0], car_bbox[1]],
                [-car_bbox[0], car_bbox[1]],
                [-car_bbox[0], -car_bbox[1]],
                [car_bbox[0], -car_bbox[1]],
                [car_bbox[0], car_bbox[1]]
            ])
            
            # Apply rotation and translation
            rotation_matrix = np.array([
                [np.cos(car_rot), -np.sin(car_rot)],
                [np.sin(car_rot), np.cos(car_rot)]
            ])
            car_bbox_polygon = np.dot(car_bbox_polygon, rotation_matrix.T) + car_centre_loc[:2]
            
            plt.plot(car_bbox_polygon[:, 0], car_bbox_polygon[:, 1], c='blue')
        
        timestamp = time.time()
        plt.savefig(f'{save_folder}/{timestamp}.png')
        plt.close()
    
    def draw_map_w_traffic_flow_sequence_in_one_frame(self, car_dict_sequence, save_root='/GPFS/public/junhaoge/workspace/SceneCrafter/data/traffic_flow/', text='', with_id=False, skip_frames=1, only_ego=False, extra_map=False, ego_plan_trajs=None, is_save=False):
        """
        Generate ego-centric visualization of traffic flow sequence.
        
        Creates a series of images showing traffic flow from the perspective
        of the ego vehicle, with proper coordinate transformations.
        
        Args:
            car_dict_sequence (list): Sequence of vehicle states over time
            save_root (str): Root directory for saving images
            text (str): Additional text for filename
            with_id (bool): Whether to display vehicle IDs
            skip_frames (int): Frame skipping for animation
            only_ego (bool): Show only ego vehicle
            extra_map (bool): Include additional map details
            ego_plan_trajs: Ego vehicle planned trajectories
            is_save (bool): Whether to save the visualization
        """
        save_folder = save_root + text
        print(f"Saving traffic flow to {save_folder}")
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Save vehicle sequence data
        with open(f'{save_folder}/car_dict_sequence.json', 'w') as f:
            json.dump(car_dict_sequence, f, indent=2)
        
        # Use first frame as reference
        ref_car_dict = car_dict_sequence[0]
        ego_info = ref_car_dict['ego_vehicle']
        ego_loc = np.array(ego_info['loc'][:2])  # Ego position (x, y)
        ego_yaw = ego_info['rot'][2] - np.pi / 2  # Ego yaw angle
        
        # Create figure with ego-centric view
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        
        # Transformation matrix for ego-centric coordinates
        rotation_matrix = np.array([
            [np.cos(-ego_yaw), -np.sin(-ego_yaw)],
            [np.sin(-ego_yaw), np.cos(-ego_yaw)]
        ])
        
        def transform_to_ego(coord):
            """Transform global coordinates to ego-centric coordinates."""
            relative_coord = np.array(coord) - ego_loc
            return np.dot(rotation_matrix, relative_coord.T).T
        
        # Draw road network in ego-centric coordinates
        for edge in self.features.values():
            if edge.feature_type == 'road_edge':
                edge_coords = np.array(edge.polyline)[:, :2]
                edge_coords_transformed = transform_to_ego(edge_coords)
                ax.plot(edge_coords_transformed[:, 0], edge_coords_transformed[:, 1], c='red', linewidth=3)
        
        for lane in self.features.values():
            if lane.feature_type == 'lane':
                lane_coords = np.array(lane.polyline)[:, :2]
                lane_coords_transformed = transform_to_ego(lane_coords)
                ax.plot(lane_coords_transformed[:, 0], lane_coords_transformed[:, 1], c='green', linewidth=3)
        
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Process vehicle trajectories
        fut_trajs_dict = {}
        all_colors = ['Blues', 'Greys', 'Oranges', 'Purples','Blues', 'Greys', 'Oranges', 'Purples','Blues', 'Greys', 'Oranges', 'Purples','Blues', 'Greys', 'Oranges', 'Purples']
        color_points = ['blue', 'Grey', 'orange', 'purple','blue', 'Grey', 'orange', 'purple','blue', 'Grey', 'orange', 'purple','blue', 'Grey', 'orange', 'purple']
        
        # Plot vehicles and their trajectories
        for car_name, car_info in ref_car_dict.items():
            car_loc = np.array(car_info['loc'][:2])
            car_rot = car_info['rot'][2]
            car_bbox = np.array(car_info['bbox']) / 2
            
            # Create vehicle bounding box
            w, l = car_bbox[0], car_bbox[1]
            car_bbox = np.array([[l, w], [-l, w], [-l, -w], [l, -w], [l, w]]) / 2
            
            # Apply rotation and translation
            rotation_matrix = np.array([
                [np.cos(car_rot), -np.sin(car_rot)],
                [np.sin(car_rot), np.cos(car_rot)]
            ])
            car_bbox = np.dot(car_bbox, rotation_matrix.T) + car_loc
            car_bbox_transformed = transform_to_ego(car_bbox)
            new_car_center = transform_to_ego(car_loc)
            
            # Skip vehicles outside 30m range
            if np.linalg.norm(new_car_center) > 30:
                continue
            
            # Determine vehicle color
            if car_name == 'ego_vehicle':
                color = 'red'
            elif car_info.get('if_static', False):
                color = 'black'
            else:
                color = color_points[cnt%4]
                cnt+=1
            
            # Plot vehicle
            ax.plot(car_bbox_transformed[:, 0], car_bbox_transformed[:, 1], c=color,linewidth=5,label=car_name)
            ax.plot(np.array([new_car_center[0],(car_bbox_transformed[3, 0]+car_bbox_transformed[0, 0])/2]), [new_car_center[1], (car_bbox_transformed[3, 1]+car_bbox_transformed[0, 1])/2],\
                    c=color)
            
            # Generate trajectory
            fut_trajs = [new_car_center]
            if not car_info.get('if_static', False):
                for car_dict in car_dict_sequence[1:]:
                    if car_name in car_dict:
                        next_car_info = car_dict[car_name]
                        next_car_loc = np.array(next_car_info['loc'][:2])
                        new_next_car_center = transform_to_ego(next_car_loc)
                        fut_trajs.append(new_next_car_center)
                
                fut_trajs_dict[car_name] = np.array(fut_trajs)
                
                # Plot trajectory
                if len(fut_trajs) > 1:
                    ax.plot(fut_trajs[:, 0], fut_trajs[:, 1], c=color, alpha=0.7, linewidth=2)
        
        cnt = 0
        for i,(car_name, plan_traj) in enumerate(fut_trajs_dict.items()):
            if car_name == 'ego_vehicle':
               color = 'Reds'
               color_point = 'red'
            else:
                color = all_colors[cnt%4]
                color_point = color_points[cnt%4]
            cnt+=1
            plan_vecs = None
            for i in range(plan_traj.shape[0]-1):
                plan_vec_i = plan_traj[i:i+2,:]
                x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
                y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
                xy = np.stack((x_linspace, y_linspace), axis=1)
                xy = np.stack((xy[:-1], xy[1:]), axis=1)
                if plan_vecs is None:
                    plan_vecs = xy
                else:
                    plan_vecs = np.concatenate((plan_vecs, xy), axis=0)
            cmap = color
            y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
            colors = color_map(y[:-1], cmap)
            line_segments = LineCollection(plan_vecs, colors=colors, linewidths=6, linestyles='solid', cmap=cmap)
            ax.add_collection(line_segments)

            ax.scatter(plan_traj[1:,0], plan_traj[1:,1], marker='*', color=color_point, s=300, label='Points with Start Symbol')
        ax.set_xlim(-30, 30)
        ax.set_ylim(-10, 30)
        #ax.legend()
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)
        plt.axis('off')  # Close the axes
        plt.gca().set_aspect('equal', adjustable='box')
        if is_save:
            plt.savefig(f'{save_folder}/ego_centered_1figure.png', bbox_inches='tight', pad_inches=0)
            plt.close()

        return ax

    def draw_map_w_traffic_flow_sequence(self,car_dict_sequence,
                                         save_path='/GPFS/public/junhaoge/workspace/SceneCrafter/data/traffic_flow/',
                                         text='',with_id=False,skip_frames=1,only_ego=False,extra_map=False,
                                         ego_plan_trajs = None
                                         ):
        save_folder = save_path
        print(f"saving traffic flow to {save_folder}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # save car_dict_sequence in json
        with open(f'{save_folder}/car_dict_sequence.json','w') as f:
            json.dump(car_dict_sequence,f,indent=2)
        # skip frames 
        if not only_ego:
            for idx,car_dict in tqdm(enumerate(car_dict_sequence)):
                if idx % skip_frames != 0:
                    continue
                plt.figure(figsize=(24, 16), dpi=100)
                # plt.axis("equal")
                road_edges = []
                lanes = []
                timestamp = time.time()
                for feature_id, feature in self.features.items():
                    if feature.feature_type == 'lane':
                        lanes.append(feature.polyline)
                        # pass
                    else:
                        road_edges.append(feature.polyline)

                for edge in road_edges:
                    edge = np.array(edge)
                    plt.plot(edge[:,0],edge[:,1],c='red')

                for lane in lanes:
                    lane = np.array(lane)
                    plt.plot(lane[:,0],lane[:,1],c='green')
                if extra_map:
                    plt.savefig(f'{save_folder}/{idx}_map.png')
                # draw cars with their 2D bbox
                for car_name, car_info in car_dict.items():
                    if car_name == 'ego_vehicle':
                        color = 'red'
                        car_id = 'ego'
                    else:
                        car_id = car_name[0] + car_name.split('_')[-1]
                        if car_info['if_overtake']:
                            color = 'brown'
                        elif car_info['if_tailgate']:
                            color = 'orange'
                        elif car_info['if_static']:
                            color = 'black'
                        else:
                            color = 'blue'
                    car_centre_loc = car_info['loc'] # [x,y,z]
                    car_rot = car_info['rot'][2] # yaw
                    car_bbox = car_info['bbox'] # [w,l]
                    car_bbox = np.array(car_bbox)
                    w = car_bbox[0]
                    l = car_bbox[1]
                    car_bbox[0] = l
                    car_bbox[1] = w
                    car_bbox = car_bbox / 2
                    car_bbox = np.array([[car_bbox[0],car_bbox[1]],[-car_bbox[0],car_bbox[1]],[-car_bbox[0],-car_bbox[1]],[car_bbox[0],-car_bbox[1]],[car_bbox[0],car_bbox[1]]])
                    car_bbox = np.dot(np.array([[np.cos(car_rot),-np.sin(car_rot)],[np.sin(car_rot),np.cos(car_rot)]]),car_bbox.T).T
                    car_bbox = car_bbox + np.array(car_centre_loc[:2])
                    plt.plot(car_bbox[:,0],car_bbox[:,1],c=color)
                    if with_id:
                        plt.text(car_centre_loc[0],car_centre_loc[1],car_id,fontsize=10)
                    

                plt.savefig(f'{save_folder}/{idx}.png')
                plt.close()
        if only_ego:
            for idx, car_dict in tqdm(enumerate(car_dict_sequence)):
                if idx % skip_frames != 0:
                    continue
                if 'ego_vehicle' not in car_dict:
                    print(f"Skipping frame {idx}: no ego_vehicle found.")
                    continue

                ego_info = car_dict['ego_vehicle']
                ego_loc = np.array(ego_info['loc'][:2])  # Ego position (x, y)
                ego_yaw = ego_info['rot'][2] - np.pi / 2  # Ego yaw angle (in radians)

                fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

                # Transformation matrix for ego-centric view (negative yaw to counteract vehicle rotation)
                rotation_matrix = np.array([
                    [np.cos(-ego_yaw), -np.sin(-ego_yaw)],
                    [np.sin(-ego_yaw), np.cos(-ego_yaw)]
                ])

                def transform_to_ego(coord):
                    """Transform a global coordinate to the ego-vehicle-centered coordinate system."""
                    relative_coord = np.array(coord) - ego_loc
                    return np.dot(rotation_matrix, relative_coord.T).T

                # Draw road edges
                for edge in self.features.values():
                    if edge.feature_type == 'road_edge':
                        edge_coords = np.array(edge.polyline)[:, :2]
                        edge_coords_transformed = transform_to_ego(edge_coords)
                        ax.plot(edge_coords_transformed[:, 0], edge_coords_transformed[:, 1], c='red')

                # Draw lanes
                for lane in self.features.values():
                    if lane.feature_type == 'lane':
                        lane_coords = np.array(lane.polyline)[:, :2]
                        lane_coords_transformed = transform_to_ego(lane_coords)
                        ax.plot(lane_coords_transformed[:, 0], lane_coords_transformed[:, 1], c='green')
                ax.set_xlim(-30, 30)
                ax.set_ylim(-30, 30)
                plt.axis('off')  # Close the axes
                plt.gca().set_aspect('equal', adjustable='box')
                if extra_map:
                    plt.savefig(f'{save_folder}/{idx}_map.png', bbox_inches='tight', pad_inches=0)
                # Draw cars
                for car_name, car_info in car_dict.items():
                    car_loc = np.array(car_info['loc'][:2])
                    car_rot = car_info['rot'][2]
                    car_bbox = np.array(car_info['bbox']) / 2
                    w = car_bbox[0]
                    l = car_bbox[1]
                    car_bbox[0] = l
                    car_bbox[1] = w
                    car_bbox = np.array([[car_bbox[0], car_bbox[1]],
                                        [-car_bbox[0], car_bbox[1]],
                                        [-car_bbox[0], -car_bbox[1]],
                                        [car_bbox[0], -car_bbox[1]],
                                        [car_bbox[0], car_bbox[1]]])
                    car_bbox = np.dot(np.array([[np.cos(car_rot), -np.sin(car_rot)],
                                                [np.sin(car_rot), np.cos(car_rot)]]), car_bbox.T).T + car_loc
                    car_bbox_transformed = transform_to_ego(car_bbox)
                    new_car_center = transform_to_ego(car_loc)

                    # Skip if car is outside the 30m range
                    if np.linalg.norm(transform_to_ego(car_loc)) > 30:
                        continue

                    color = 'blue'  # Default color
                    if car_name == 'ego_vehicle':
                        color = 'red'
                    elif car_info['if_overtake']:
                        color = 'brown'
                    elif car_info['if_tailgate']:
                        color = 'orange'
                    elif car_info['if_static']:
                        color = 'black'

                    ax.plot(car_bbox_transformed[:, 0], car_bbox_transformed[:, 1], c=color)
                    ax.plot(np.array([new_car_center[0],(car_bbox_transformed[3, 0]+car_bbox_transformed[0, 0])/2]), [new_car_center[1], (car_bbox_transformed[3, 1]+car_bbox_transformed[0, 1])/2],\
                         c=color,linewidth=2)
                    
                    if with_id:
                        car_id = 'ego' if car_name == 'ego_vehicle' else car_name[0] + car_name.split('_')[-1]
                        ax.text(car_bbox_transformed[0, 0], car_bbox_transformed[0, 1], car_id, fontsize=10)

                if ego_plan_trajs is not None:
                    ego_plan_traj = ego_plan_trajs[idx]
                    ego_plan_traj[abs(ego_plan_traj) < 0.01] = 0.0
                    ego_plan_traj = ego_plan_traj.cumsum(axis=0)
                    ego_plan_traj = np.concatenate((np.zeros((1, ego_plan_traj.shape[1])), ego_plan_traj), axis=0)
                    ego_plan_traj = np.stack((ego_plan_traj[:-1], ego_plan_traj[1:]), axis=1)

                    plan_vecs = None
                    for i in range(ego_plan_traj.shape[0]):
                        plan_vec_i = ego_plan_traj[i]
                        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
                        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
                        xy = np.stack((x_linspace, y_linspace), axis=1)
                        xy = np.stack((xy[:-1], xy[1:]), axis=1)
                        if plan_vecs is None:
                            plan_vecs = xy
                        else:
                            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)
                    cmap = 'winter'
                    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
                    colors = color_map(y[:-1], cmap)
                    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=6, linestyles='solid', cmap=cmap)
                    ax.add_collection(line_segments)
                
                # Set plot limits for 30m range
                ax.set_xlim(-30, 30)
                ax.set_ylim(-30, 30)
                plt.axis('off')  # Close the axes
                plt.gca().set_aspect('equal', adjustable='box')

                plt.savefig(f'{save_folder}/{idx}_ego_centered.png', bbox_inches='tight', pad_inches=0)
                plt.close()

    def draw_map_w_traffic_flow_sequence_end2end(self,car_dict_sequence,save_path,with_id=False,skip_frames=1):
        save_folder = save_path
        print(f"saving traffic flow to {save_folder}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # save car_dict_sequence in json
        with open(f'{save_folder}/car_dict_sequence.json','w') as f:
            json.dump(car_dict_sequence,f,indent=2)
        # skip frames 

        for idx,car_dict in tqdm(enumerate(car_dict_sequence)):
            if idx % skip_frames != 0:
                continue
            plt.figure(figsize=(24, 16), dpi=100)
            road_edges = []
            lanes = []
            timestamp = time.time()
            for feature_id, feature in self.features.items():
                if feature.feature_type == 'lane':
                    lanes.append(feature.polyline)
                    # pass
                else:
                    road_edges.append(feature.polyline)

            for edge in road_edges:
                edge = np.array(edge)
                plt.plot(edge[:,0],edge[:,1],c='red')

            for lane in lanes:
                lane = np.array(lane)
                plt.plot(lane[:,0],lane[:,1],c='green')

            # draw cars with their 2D bbox
            for car_name, car_info in car_dict.items():
                if car_name == 'ego_vehicle':
                    color = 'red'
                    car_id = 'ego'
                else:
                    car_id = car_name[0] + car_name.split('_')[-1]
                    if car_info['if_overtake']:
                        color = 'brown'
                    elif car_info['if_tailgate']:
                        color = 'orange'
                    elif car_info['if_static']:
                        color = 'black'
                    else:
                        color = 'blue'
                car_centre_loc = car_info['loc'] # [x,y,z]
                car_rot = car_info['rot'][2] # yaw
                car_bbox = car_info['bbox'] # [w,l]
                car_bbox = np.array(car_bbox)
                w = car_bbox[0]
                l = car_bbox[1]
                car_bbox[0] = l
                car_bbox[1] = w
                car_bbox = car_bbox / 2
                car_bbox = np.array([[car_bbox[0],car_bbox[1]],[-car_bbox[0],car_bbox[1]],[-car_bbox[0],-car_bbox[1]],[car_bbox[0],-car_bbox[1]],[car_bbox[0],car_bbox[1]]])
                car_bbox = np.dot(np.array([[np.cos(car_rot),-np.sin(car_rot)],[np.sin(car_rot),np.cos(car_rot)]]),car_bbox.T).T
                car_bbox = car_bbox + np.array(car_centre_loc[:2])
                plt.plot(car_bbox[:,0],car_bbox[:,1],c=color)
                if with_id:
                    plt.text(car_centre_loc[0],car_centre_loc[1],car_id,fontsize=10)
                

            plt.savefig(f'{save_folder}/{idx}.png')
            plt.close()

    def draw_map_w_waypoints(self, waypoints, file_path=None, spawn_point_num=None):
        plt.figure(figsize=(24, 16), dpi=200)
        road_edges = []
        lanes = []
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                lanes.append(feature.polyline)
            else:
                road_edges.append(feature.polyline)

        for edge in road_edges:
            edge = np.array(edge)
            plt.plot(edge[:,0],edge[:,1],c='red')

        for lane in lanes:
            lane = np.array(lane)
            plt.plot(lane[:,0],lane[:,1],c='green') 
        plt.scatter(waypoints[0].transform.get_location()[0], waypoints[0].transform.get_location()[1], c='green', s=100)
        for waypoint in waypoints:
            rotation = waypoint.transform.get_rotation()
            rotation = rotation[2]
            x = waypoint.transform.get_location()[0]
            y = waypoint.transform.get_location()[1]
            dx = 0.5 * math.cos(rotation)
            dy = 0.5 * math.sin(rotation)
            plt.arrow(x, y, dx, dy, head_width=2, head_length=2, fc='k', ec='k')
        if file_path is None:
            time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            plt.savefig(f'data/draw_pic/map_w_waypoints_{time}.png')
        else:
            plt.savefig(file_path)
        plt.close()

    def draw_map_w_key_waypoints(self, waypoints, spawn_point_num=None):
        plt.figure(figsize=(24, 16), dpi=200)
        road_edges = []
        lanes = []
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                lanes.append(feature.polyline)
            else:
                road_edges.append(feature.polyline)

        for edge in road_edges:
            edge = np.array(edge)
            plt.plot(edge[:,0],edge[:,1],c='red')

        for lane in lanes:
            lane = np.array(lane)
            plt.plot(lane[:,0],lane[:,1],c='green') 

        plt.scatter(waypoints[0].transform.get_location()[0], waypoints[0].transform.get_location()[1], c='green', s=100)
        plt.scatter(waypoints[-1].transform.get_location()[0], waypoints[-1].transform.get_location()[1], c='red', s=100)
        plt.text(waypoints[0].transform.get_location()[0], waypoints[0].transform.get_location()[1], f"Spawn Point Num: {spawn_point_num}", fontsize=20)
        for waypoint in waypoints:
            waypoint_loc = np.array(waypoint.transform.get_location())
            if waypoint.road_option != RoadOption.STRAIGHT:
                plt.scatter(waypoint_loc[0], waypoint_loc[1], c='red', s=100)
            else:
                plt.scatter(waypoint_loc[0], waypoint_loc[1], c='blue', s=10)
        if not os.path.exists('data/draw_pic/spawn_points_w_waypoints'):
            os.makedirs('data/draw_pic/spawn_points_w_waypoints')
        if spawn_point_num is not None:
            plt.savefig(f'data/draw_pic/spawn_points_w_waypoints/map_w_waypoints_{spawn_point_num}.png')
        else: 
            plt.savefig('data/draw_pic/spawn_points_w_waypoints/map_w_waypoints.png')
        plt.close()


if __name__ == '__main__':
    map_json_path = 'data/end2end_map_data/test_map.json'
    with open(map_json_path, 'r') as file:
        map_data = json.load(file)

    map_obj = Map(map_data)

    lane_id = '97'
    map_obj.draw_map_w_spawn_points()
    spawn_points = map_obj.get_spawn_points()
    for idx, spawn_point in enumerate(spawn_points):
        print(f"Draw Waypoints of Spawn Point {idx}")
        waypoints = map_obj.generate_overall_plan_waypoints(spawn_point, driving_mode='Random', ignore_lanechange=False)
        # map_obj.draw_map_w_waypoints(waypoints, idx)
        map_obj.draw_map_w_key_waypoints(waypoints, idx)
    
    