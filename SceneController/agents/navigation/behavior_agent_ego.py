#!/usr/bin/env python3
"""
Behavior Agent Ego - Advanced Autonomous Vehicle Navigation System
"""
import re
import random
import numpy as np
from sklearn import neighbors
from navigation.controller import Control
from navigation.local_planner_behavior import LocalPlanner, RoadOption
from navigation.types_behavior import (
    Cautious, Aggressive, Normal, ExtremeAggressive,
    Cautious_fast, Aggressive_fast, Normal_fast, ExtremeAggressive_fast,
    Cautious_highway, Aggressive_highway, Normal_highway, ExtremeAggressive_highway
)
from SceneController.scene import Scene
from .waypoint import Waypoint
from .tools.misc import (
    is_within_distance, calculate_distance, positive, 
    calculate_rotation, calculate_rotation_v2, interpolate_locations_by_steps
)
import copy
from collections import deque

def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi] range for consistent angular calculations.
    
    Args:
        angle (float): Input angle in radians
        
    Returns:
        float: Normalized angle within [-pi, pi] range
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def rotation_matrix(rot):
    """
    Create a 3D rotation matrix from Euler angles for coordinate transformations.
    
    Args:
        rot (list): Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    t1, t2, t3 = rot
    
    # Rotation matrix around X-axis (roll)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(t1), -np.sin(t1)],
        [0, np.sin(t1), np.cos(t1)]
    ])
    
    # Rotation matrix around Y-axis (pitch)
    R_y = np.array([
        [np.cos(t2), 0, np.sin(t2)],
        [0, 1, 0],
        [-np.sin(t2), 0, np.cos(t2)]
    ])
    
    # Rotation matrix around Z-axis (yaw)
    R_z = np.array([
        [np.cos(t3), -np.sin(t3), 0],
        [np.sin(t3), np.cos(t3), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: R = Rz * Ry * Rx
    R = R_z @ R_y @ R_x
    return R


class BehaviorAgent:
    """
    Advanced behavior-based autonomous vehicle agent for realistic traffic simulation.
    
    This agent implements sophisticated navigation behaviors including path planning,
    collision avoidance, lane changing, overtaking, and traffic rule compliance. It supports
    multiple driving personalities and can adapt its behavior based on traffic conditions.
    
    Key Capabilities:
    - Dynamic route planning with waypoint following
    - Multi-level behavior system (cautious to aggressive)
    - Real-time collision detection and avoidance
    - Lane changing and overtaking maneuvers
    - Speed adaptation based on traffic and road conditions
    - Future trajectory prediction and planning
    - Integration with traffic simulation environment
    
    The agent maintains a comprehensive state including position, velocity, acceleration,
    and behavioral parameters to make intelligent driving decisions.
    """

    def __init__(self, config):
        """
        Initialize the behavior agent with comprehensive configuration.
        
        Args:
            config (dict): Configuration dictionary containing:
                - name (str): Unique vehicle identifier
                - initial_path (list): Starting waypoints for initial route
                - vehicle_type (str): Vehicle classification type
                - vehicle_bbox (list): Vehicle dimensions [length, width, height]
                - behavior (str): Driving behavior type ('cautious', 'normal', 'aggressive', 'extreme_aggressive')
                - speed_type (str): Speed profile ('normal', 'fast', 'highway')
                - max_speed (float, optional): Maximum speed limit override
                - f_len (float): Front axle to center distance
                - r_len (float): Rear axle to center distance
                - control_angel (float): Control angle parameters for steering
        """
        # Vehicle identification and state
        self.vehicle_name = config['name']
        self.look_ahead_steps = 0
        self.end_route_flag = False

        # Current vehicle state
        self.cur_waypoint = config['initial_path'][0]  # Current position waypoint
        self.last_waypoint = config['initial_path'][0]  # Previous position for tracking
        self.speed = 0  # Current speed in km/h
        self.prev_speed = 0  # Previous speed for acceleration calculation
        self.velocity_xy = [0, 0]  # 2D velocity vector in km/h
        self.acceleration = 0  # Linear acceleration in m/s²
        self.prev_velocity_xy = [0, 0]  # Previous velocity for derivative calculation
        self.acceleration_xy = [0, 0]  # 2D acceleration vector in m/s²
        self.omega = 0  # Angular velocity in rad/s
        self.steer_value = 0  # Current steering angle
        
        # Navigation and planning
        self.speed_limit = 0  # Current speed limit
        self.direction = None  # Current driving direction
        self.incoming_direction = None  # Next planned direction
        self.incoming_waypoint = None  # Next target waypoint
        self.start_waypoint = None  # Route start waypoint
        self.end_waypoint = None  # Route destination waypoint
        self.plan_path = config['initial_path']  # Complete planned route
        self.min_speed = 5  # Minimum operational speed in km/h
        
        # Vehicle specifications
        self.vehicle_type = config['vehicle_type']
        self.bounding_box = config['vehicle_bbox']  # [length, width, height] in meters
        self.f_len = config['f_len']  # Front axle to CG distance
        self.r_len = config['r_len']  # Rear axle to CG distance
        self.control_angel = config['control_angel']  # Steering control parameters
        self.speed_type = config['speed_type']  # Speed profile classification
        
        # Control and planning parameters
        self._sampling_resolution = 4.5  # Waypoint sampling distance in meters
        self.cur_control = Control()  # Current control commands
        
        # Overtaking and maneuvering state
        self.if_overtake = False  # Active overtaking flag
        self.overtake_direction = None  # Direction of overtaking maneuver
        self.if_tailgate = False  # Tailgating detection flag
        self.overtake_end_waypoint = None  # End waypoint for overtaking
        self.vehicle_name_to_overtake = None  # Target vehicle for overtaking
        self.if_static = False  # Static obstacle detection
        self.last_target_waypoint = None  # Last successfully reached waypoint

        # Behavior configuration based on driving style and speed profile
        behavior = config['behavior']
        if behavior == 'cautious':
            if self.speed_type == 'fast':
                self.behavior = Cautious_fast()
            elif self.speed_type == 'highway':
                self.behavior = Cautious_highway()
            else:
                self.behavior = Cautious()

        elif behavior == 'normal':
            if self.speed_type == 'fast':
                self.behavior = Normal_fast()
            elif self.speed_type == 'highway':
                self.behavior = Normal_highway()
            else:
                self.behavior = Normal()

        elif behavior == 'aggressive':
            if self.speed_type == 'fast':
                self.behavior = Aggressive_fast()
            elif self.speed_type == 'highway':
                self.behavior = Aggressive_highway()
            else:
                self.behavior = Aggressive()

        elif behavior == 'extreme_aggressive':
            if self.speed_type == 'fast':
                self.behavior = ExtremeAggressive_fast()
            elif self.speed_type == 'highway':
                self.behavior = ExtremeAggressive_highway()
            else:
                self.behavior = ExtremeAggressive()

        # Apply custom speed limits if provided
        if 'max_speed' in config.keys():
            self.behavior.max_speed = config['max_speed']

        # Add realistic speed variation based on vehicle type
        if 'ego' in self.vehicle_name:
            # Ego vehicle gets smaller speed variation for more stable behavior
            self.behavior.max_speed *= (1 + random.uniform(-0.15, 0.15))
        else:
            # NPC vehicles get larger speed variation for more diverse traffic
            self.behavior.max_speed *= (1 + random.uniform(-0.25, 0.25))

        # Calculate safety parameters based on vehicle speed and dimensions
        self.original_braking_distance = self.behavior.braking_distance
        if self.behavior.max_speed < 50:
            # Low speed: linear scaling with vehicle size consideration
            self.behavior.min_proximity_threshold += (
                self.behavior.max_speed / 3.6 / 1.5 + 
                max(self.bounding_box[0], self.bounding_box[1])
            )
        else:
            # High speed: quadratic scaling for increased safety margins
            self.behavior.min_proximity_threshold *= 2.0 * (self.behavior.max_speed/100)
            self.behavior.min_proximity_threshold += (
                self.behavior.max_speed / 3.6 + 
                max(self.bounding_box[0], self.bounding_box[1])
            )

        # Initialize dynamic control parameters
        self.cur_speed_limit = self.behavior.max_speed
        self.update_braking_distance()
        
        # Maximum acceleration and braking limits based on vehicle capabilities
        self.max_acceleration = max(self.behavior.max_speed / 3.6 / 3.0, 6.0)
        self.max_braking = max(self.behavior.max_speed / 3.6 / 2.0, 8.0)

        # Initialize local planner for path following
        self.scenario_trigger = self.behavior.scenario_trigger
        self._local_planner = LocalPlanner(self, min_distance=3)
        self.speed_limit = self.behavior.max_speed
        self._local_planner.set_speed(self.speed_limit)
        self._local_planner.set_global_plan(self.plan_path, clean=True)
        
        # Set up initial trajectory tracking
        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)
        
        # Initialize future trajectory buffer for prediction (30 steps)
        self.future_trajectory = deque(maxlen=30)
        self.future_trajectory.append(self.cur_waypoint)
        self.future_trajectory.append(self.incoming_waypoint)

    def update_braking_distance(self):
        """
        Dynamically update braking distance based on current speed and conditions.
        
        This method adjusts the braking distance parameter based on the vehicle's
        current speed limit and physical characteristics to ensure safe stopping
        distances under different operating conditions.
        """
        if self.behavior.max_speed < 50.0:
            # Low speed scenario: conservative braking with linear scaling
            self.behavior.braking_distance = (
                self.original_braking_distance * (self.behavior.max_speed/50) +
                self.cur_speed_limit / 3.6 / 6.0 +
                max(self.bounding_box[0], self.bounding_box[1])
            )
        else:
            # High speed scenario: extended braking distance with quadratic scaling
            self.behavior.braking_distance = (
                self.original_braking_distance * 2.0 * (self.behavior.max_speed/100) +
                self.cur_speed_limit / 3.6 / 4.0 +
                max(self.bounding_box[0], self.bounding_box[1])
            )

    def get_trajectory(self, step_num=10):
        """
        Retrieve the planned trajectory for the specified number of future steps.
        
        Args:
            step_num (int): Number of future steps to include in trajectory
            
        Returns:
            list: List of future waypoints representing the planned trajectory
        """
        return self._local_planner.get_trajectory(step_num)

    def update_information(self, update_transform):
        """
        Update vehicle state information based on current world state and sensor data.
        
        This method synchronizes the agent's internal state with the actual vehicle
        position and orientation from the simulation environment, handling lane
        changes and ensuring accurate waypoint tracking.
        
        Args:
            update_transform: Current vehicle transform containing location and rotation
        """
        # Update current driving direction from local planner
        self.direction = self._local_planner.target_road_option
        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW
            
        # Adjust lookahead steps based on current speed for appropriate planning horizon
        self.look_ahead_steps = int(self.speed_limit / 15)
        
        # Extract current position and orientation
        location = update_transform.location
        rotation = update_transform.rotation
        
        # Handle lane changes during driving - re-identify current lane
        # During driving, vehicle behavior might cause lane changes, so we need to
        # re-acquire the current lane's lane_id and lane_point_idx
        neighbour_ids = Scene._map.get_all_neighbor_lane_ids(self.cur_waypoint.lane_id)
        cur_lane_id, cur_point_index, _ = Scene._map.find_nearest_lane_point(location)
        
        # If current lane is not in neighboring lanes, maintain current lane assignment
        # The cur_point_index is not critical for subsequent processes as we mainly use
        # lane_id to represent neighboring lane relationships
        if cur_lane_id not in neighbour_ids:
            cur_lane_id = self.cur_waypoint.lane_id
            cur_point_index = self.cur_waypoint.lane_point_idx
            
        # Build updated waypoint configuration
        cur_waypoint_config = Scene._map.build_waypoint_config(
            location, rotation, self.direction, cur_lane_id, cur_point_index
        )
        self.cur_waypoint = Waypoint(cur_waypoint_config)
        
        # Get next waypoint and direction for path following
        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)
        
        # Handle edge cases where no valid incoming waypoint is found
        if self.incoming_waypoint is None:
            self.incoming_waypoint = self.cur_waypoint
        if self.incoming_direction is None:
            self.incoming_direction = RoadOption.LANEFOLLOW

    def get_control_cmd(self):
        """
        Determine steering control command based on relative position to next waypoint.
        
        This method calculates the appropriate steering direction by analyzing the
        relative position of the next waypoint in the vehicle's coordinate frame.
        
        Returns:
            tuple: (control_cmd, incoming_loc) where:
                - control_cmd: 0 for left turn, 1 for right turn, 2 for straight
                - incoming_loc: Absolute location coordinates of the next waypoint
        """
        # Get current vehicle state
        cur_loc = self.cur_waypoint.transform.location
        cur_rot = self.cur_waypoint.transform.rotation
        
        # Get immediate next waypoint
        closest_incoming_waypoint, closest_incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(steps=1)
        
        # Fallback to last known waypoint if none found
        if closest_incoming_waypoint is None:
            closest_incoming_waypoint = self.last_target_waypoint
            
        incoming_loc = closest_incoming_waypoint.transform.location
        incoming_rot = closest_incoming_waypoint.transform.rotation
        
        # Transform next waypoint to vehicle coordinate system
        # Create transformation matrices for coordinate conversion
        cur_pose = np.eye(4)
        cur_pose[:3, :3] = rotation_matrix(cur_rot)
        cur_pose[:3, 3] = np.array([cur_loc[0], cur_loc[1], cur_loc[2]])
        
        incoming_pose = np.eye(4)
        incoming_pose[:3, :3] = rotation_matrix(incoming_rot)
        incoming_pose[:3, 3] = np.array([incoming_loc[0], incoming_loc[1], incoming_loc[2]])
        
        # Calculate relative position in vehicle frame
        relative_pose = np.linalg.inv(cur_pose) @ incoming_pose
        
        # Convert to LiDAR coordinate system for consistent reference
        l2e_matrix = np.eye(4)
        l2e_matrix[:3, :3] = rotation_matrix([0, 0, -np.pi/2])
        l2e_matrix[:3, 3] = np.array([0.943713, 0.0, 1.84023])
        relative_pose_lidar = np.linalg.inv(l2e_matrix) @ relative_pose
        
        relative_loc = relative_pose_lidar[:3, 3]
        
        # Determine steering command based on lateral offset
        # If next waypoint is >2m left of current heading: turn left (0)
        # If next waypoint is >2m right of current heading: turn right (1)
        # Otherwise: go straight (2)
        if relative_loc[0] > 2:
            control_cmd = 0  # Left turn
        elif relative_loc[0] < -2:
            control_cmd = 1  # Right turn
        else:
            control_cmd = 2  # Go straight
            
        self.last_target_waypoint = copy.deepcopy(closest_incoming_waypoint)
        return control_cmd, incoming_loc

    def set_last_waypoint(self, last_waypoint):
        """Set reference waypoint for tracking purposes."""
        self.last_waypoint = copy.deepcopy(last_waypoint)

    def get_speed(self):
        """
        Get current vehicle speed.
        
        Returns:
            float: Current speed in km/h
        """
        return self.speed

    def set_speed(self, speed):
        """
        Update vehicle speed and recalculate velocity vectors.
        
        Args:
            speed (float): New speed in km/h
        """
        self.prev_velocity_xy = self.velocity_xy
        self.prev_speed = self.speed
        self.speed = speed
        
        # Update velocity vector based on current heading
        current_yaw = self.cur_waypoint.get_rotation()[2]
        self.velocity_xy = [
            speed * np.cos(current_yaw),
            speed * np.sin(current_yaw)
        ]

    def set_omega(self, omega):
        """Set angular velocity for vehicle rotation control."""
        self.omega = omega

    def set_steer_value(self, steer_value):
        """Set steering angle for vehicle control."""
        self.steer_value = steer_value

    def set_acceleration(self, acceleration):
        """
        Update vehicle acceleration and calculate acceleration vectors.
        
        Args:
            acceleration (float): Linear acceleration value in m/s²
        """
        # Calculate acceleration from velocity change over 0.5s time step
        self.acceleration_xy = [
            (self.velocity_xy[0] - self.prev_velocity_xy[0]) / 0.5,
            (self.velocity_xy[1] - self.prev_velocity_xy[1]) / 0.5
        ]

    def rearrange_route(self, additional_route):
        """
        Insert additional route segment into current plan while maintaining connectivity.
        
        This method allows dynamic route modification by inserting new waypoints
        while preserving the connection to the original destination.
        
        Args:
            additional_route (list): New route segment to insert
        """
        if additional_route is None or len(additional_route) == 0:
            return
            
        # Get endpoints for route connection
        end_point_of_additional_route = copy.deepcopy(additional_route[-1])
        end_point_of_plan_path = self.plan_path[-1]
        
        # Calculate route from additional segment to original destination
        new_route = self._trace_route(end_point_of_additional_route, end_point_of_plan_path)
        
        if new_route:
            # Combine routes, avoiding duplicate waypoints
            route_combined = additional_route + new_route[1:]
            route_combined = route_combined[1:]  # Remove current position
            
            # Update planner with new route
            self._local_planner.set_global_plan(route_combined, clean=True, clean_global=True)
            self.plan_path = copy.deepcopy(route_combined)
        else:
            print("Route Invalid. Keeping the current plan.")

    def reroute(self, cur_loc):
        """
        Generate new route when approaching destination for continuous operation.
        
        This method is called when the vehicle is nearing its destination to ensure
        continuous operation by generating a new random destination.
        
        Args:
            cur_loc: Current vehicle location
        """
        print("Target almost reached, setting new destination...")
        new_route = Scene._map.generate_overall_plan_waypoints(cur_loc)
        self._local_planner.set_global_plan(new_route, clean=True, clean_global=True)

    def reroute_all(self, new_plan):
        """
        Completely replace current route with new plan.
        
        Args:
            new_plan (list): Complete new route plan as list of waypoints
        """
        # Refine waypoint spacing for smooth navigation
        new_plan = Scene._map.refine_plan_waypoints(new_plan, 2.0)
        self.plan_path = copy.deepcopy(new_plan)
        self._local_planner.set_global_plan(new_plan, clean=True, clean_global=True)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        Compute optimal route between two waypoints using global routing.
        
        Args:
            start_waypoint: Starting waypoint
            end_waypoint: Destination waypoint
            
        Returns:
            list: Optimal route as list of waypoints
        """
        return Scene.generate_route_w_waypoints(start_waypoint, end_waypoint)

    def set_current_plan(self, plan):
        """
        Set immediate navigation plan with waypoint generation.
        
        Converts coordinate list to waypoint objects with proper orientation
        and lane assignment.
        
        Args:
            plan (list): List of [x,y,z] coordinate tuples representing path
        """
        new_waypoint_plan = []
        last_elem = None
        
        for elem in plan:
            # Ensure valid direction
            self.direction = self._local_planner.target_road_option
            if self.direction is None:
                self.direction = RoadOption.LANEFOLLOW
                
            location = elem
            
            # Calculate orientation for smooth path following
            if last_elem is not None:
                rotation = calculate_rotation(last_elem, elem)
            else:
                rotation = self.cur_waypoint.get_rotation()
                
            # Handle potential lane changes during route following
            neighbour_ids = Scene._map.get_all_neighbor_lane_ids(self.cur_waypoint.lane_id)
            cur_lane_id, cur_point_index, _ = Scene._map.find_nearest_lane_point(location)
            
            # Maintain current lane if not in neighboring lanes
            if cur_lane_id not in neighbour_ids:
                cur_lane_id = self.cur_waypoint.lane_id
                cur_point_index = self.cur_waypoint.lane_point_idx
                
            # Create waypoint with proper configuration
            cur_waypoint_config = Scene._map.build_waypoint_config(
                location, rotation, self.direction, cur_lane_id, cur_point_index
            )
            new_waypoint_plan.append(Waypoint(cur_waypoint_config))
            last_elem = elem
            
        self._local_planner.set_current_plan(new_waypoint_plan)

    def set_future_trajectory(self, trajectory, u=0.1):
        """
        Set predictive future trajectory with interpolation for smooth motion.
        
        This method creates a detailed future trajectory by interpolating between
        provided waypoints and ensuring smooth vehicle motion.
        
        Args:
            trajectory (np.ndarray): Array of future positions [[x,y,z], ...]
            u (float): Interpolation parameter for smoothness (default: 0.1)
        """
        # Insert current position as trajectory starting point
        trajectory = np.insert(trajectory, 0, self.cur_waypoint.transform.location, axis=0)
        
        # Generate intermediate waypoints for smooth motion (5 steps between each)
        interpolate_locations = interpolate_locations_by_steps(trajectory, 5)
        self.future_trajectory.clear()
        
        new_waypoint_plan = []
        last_elem = None
        offset = 1
        
        for idx, elem in enumerate(interpolate_locations):
            # Skip first point (already current position)
            if idx == 0:
                continue
                
            # Adjust offset for end of trajectory
            if idx >= len(interpolate_locations) - 5:
                offset = 1
                
            # Ensure valid direction
            self.direction = self._local_planner.target_road_option
            if self.direction is None:
                self.direction = RoadOption.LANEFOLLOW
                
            location = elem
            
            # Calculate appropriate orientation for smooth path following
            if idx == len(interpolate_locations) - 1:
                # Last point: use previous segment for orientation
                rotation = calculate_rotation(interpolate_locations[idx-1], elem)
            else:
                # Intermediate points: use smooth interpolation
                if idx == 1:
                    rotation_prev = calculate_rotation_v2(
                        interpolate_locations[idx-offset], elem,
                        self.cur_waypoint.transform.rotation
                    )
                else:
                    rotation_prev = calculate_rotation_v2(
                        interpolate_locations[idx-offset], elem,
                        self.future_trajectory[-1].transform.rotation
                    )
                rotation_next = calculate_rotation_v2(
                    elem, interpolate_locations[idx+offset], rotation_prev
                )
                
                # Handle sharp turns with yaw rate limitation
                yr = rotation_next[-1] - rotation_prev[-1]
                if np.abs(yr) > 5:
                    yr = yr-2*np.pi*np.sign(yr)
                rotation = [0,0,(u*yr+rotation_prev[-1])]
            neighbour_ids = Scene._map.get_all_neighbor_lane_ids(self.cur_waypoint.lane_id)
            cur_lane_id, cur_point_index, _ = Scene._map.find_nearest_lane_point(location)
            if cur_lane_id not in neighbour_ids:
                cur_lane_id = self.cur_waypoint.lane_id
                cur_point_index = self.cur_waypoint.lane_point_idx
            cur_waypoint_config = Scene._map.build_waypoint_config(location, rotation, self.direction, cur_lane_id, cur_point_index)
            self.future_trajectory.append(Waypoint(cur_waypoint_config))
            if idx % 5 == 0:
                new_waypoint_plan.append(Waypoint(cur_waypoint_config))
            last_elem = elem
        self._local_planner.set_current_plan(new_waypoint_plan)
        
    def set_future_trajectory_w_yaw(self, trajectory):
        """
        Set predictive future trajectory with explicit yaw angles for precise orientation control.
        
        This method creates a detailed future trajectory by interpolating between provided
        positions and orientations, ensuring smooth vehicle motion with precise yaw control.
        Unlike set_future_trajectory, this method accepts pre-defined yaw angles for each point.
        
        Args:
            trajectory (np.ndarray): Array of future positions with yaw angles [[x,y,z,yaw], ...]
        """
        # Insert current position and orientation as trajectory starting point
        trajectory = np.insert(trajectory, 0, 
                               self.cur_waypoint.transform.location + [self.cur_waypoint.transform.rotation[-1]], 
                               axis=0)
        
        # Generate intermediate waypoints for smooth motion (5 steps between each)
        interpolate_locations = interpolate_locations_by_steps(trajectory, 5)
        self.future_trajectory.clear()
        
        new_waypoint_plan = []
        last_elem = None
        offset = 1
        
        for idx, elem in enumerate(interpolate_locations):
            # Skip first point (already current position)
            if idx == 0:
                continue
                
            # Adjust offset for end of trajectory to prevent index out of bounds
            if idx >= len(interpolate_locations) - 5:
                offset = 1
                
            # Ensure valid direction for waypoint planning
            self.direction = self._local_planner.target_road_option
            if self.direction is None:
                self.direction = RoadOption.LANEFOLLOW
                
            # Extract position and orientation from trajectory data
            location = elem[:3]  # [x, y, z] coordinates
            rotation = [0, 0, elem[-1]]  # [roll, pitch, yaw] with yaw from trajectory
            
            # Handle potential lane changes during trajectory following
            neighbour_ids = Scene._map.get_all_neighbor_lane_ids(self.cur_waypoint.lane_id)
            cur_lane_id, cur_point_index, _ = Scene._map.find_nearest_lane_point(location)
            
            # Maintain current lane if not in neighboring lanes
            if cur_lane_id not in neighbour_ids:
                cur_lane_id = self.cur_waypoint.lane_id
                cur_point_index = self.cur_waypoint.lane_point_idx
                
            # Create waypoint with precise orientation control
            cur_waypoint_config = Scene._map.build_waypoint_config(
                location, rotation, self.direction, cur_lane_id, cur_point_index
            )
            self.future_trajectory.append(Waypoint(cur_waypoint_config))
            
            # Add every 5th waypoint to the planner for efficiency
            if idx % 5 == 0:
                new_waypoint_plan.append(Waypoint(cur_waypoint_config))
            last_elem = elem
            
        self._local_planner.set_current_plan(new_waypoint_plan)
    

    def get_one_step_trajectory(self):
        """
        Retrieve the next step in the planned trajectory with calculated motion parameters.
        
        This method calculates the immediate next trajectory waypoint along with the
        required speed, acceleration, and angular velocity to smoothly transition
        to the next position in the planned path.
        
        Returns:
            tuple: (waypoint, speed, acceleration, angular_velocity) where:
                - waypoint: Next target waypoint in trajectory
                - speed: Target speed in km/h for smooth transition
                - acceleration: Required acceleration in m/s²
                - angular_velocity: Required angular velocity in rad/s
        """
        # Get current and next trajectory waypoints
        curr_trajectory_waypoint = self.cur_waypoint
        target_trajectory_waypoint = self.future_trajectory.popleft()
        next_trajectory_waypoint = self.future_trajectory[0]
        
        # Calculate target speed based on distance to next waypoint
        # Assuming 0.1s time step for trajectory following
        distance_to_next = calculate_distance(
            next_trajectory_waypoint.transform.location, 
            target_trajectory_waypoint.transform.location
        )
        speed = distance_to_next / 0.1  # Speed in m/s
        speed = speed * 3.6  # Convert to km/h
        
        # Calculate required acceleration for smooth speed transition
        current_speed_ms = self.speed / 3.6  # Convert current speed to m/s
        acceleration = (speed - current_speed_ms) / 0.5  # Acceleration over 0.5s
        
        # Calculate required angular velocity for orientation alignment
        current_yaw = target_trajectory_waypoint.transform.rotation[2]
        next_yaw = next_trajectory_waypoint.transform.rotation[2]
        omega = next_yaw - current_yaw
        
        # Normalize angular velocity to [-π, π] range
        if np.abs(omega) > 5:
            omega = omega - 2 * np.pi * np.sign(omega)
        omega = omega / 0.1  # Angular velocity in rad/s
        
        return target_trajectory_waypoint, speed, acceleration, omega

    def get_current_plan(self):
        """
        Retrieve the current navigation plan with all waypoints and directions.
        
        This method provides access to the agent's immediate navigation plan,
        including all queued waypoints and their associated driving directions.
        
        Returns:
            list: Current plan as list of (waypoint, direction) tuples where:
                - waypoint: Waypoint object with location and rotation
                - direction: RoadOption enum indicating driving direction
        """
        return self._local_planner.get_current_plan()

    def emergency_stop(self):
        """
        Execute immediate emergency braking for collision avoidance or route completion.
        
        This method generates a full-brake control command that immediately brings
        the vehicle to a complete stop. It is used for emergency situations,
        collision avoidance, or when the vehicle reaches its destination.
        
        Returns:
            Control: Emergency braking control with maximum brake force and zero throttle/steering
        """
        # Create emergency braking control
        control = Control()
        control.steer = 0.0      # Neutral steering to maintain stability
        control.throttle = 0.0   # Zero throttle to stop acceleration
        control.brake = 1.0      # Maximum braking force
        control.hand_brake = False  # Disable handbrake for controlled stop
        
        # Update current control state
        self.cur_control = control
        return control

    def get_next_lane_id(self):
        """
        Determine the next lane ID in the planned route for lane change detection.
        
        This method analyzes the planned trajectory to identify when the vehicle
        will transition to a different lane, which is crucial for lane change
        maneuvers and traffic interaction management.
        
        Returns:
            str or None: Next lane ID if different from current lane, None otherwise
        """
        # Start from current lane
        cur_lane_id = self.cur_waypoint.lane_id
        next_lane_id = None
        step_num = 1
        
        # Search through upcoming waypoints for lane changes
        while next_lane_id is None or next_lane_id == cur_lane_id:
            # Safety limit to prevent infinite loops
            if step_num > 10:
                break
                
            # Get next waypoint in trajectory
            next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=step_num)[0]
            
            # Handle edge case where no next waypoint exists
            if next_wpt is None:
                next_wpt = self.cur_waypoint
                next_lane_id = next_wpt.lane_id
                break
                
            next_lane_id = next_wpt.lane_id
            
            # Return the next lane ID if it's different from current
            if next_lane_id != cur_lane_id:
                return next_lane_id
                
            step_num += 1
            
        return None

    def run_step(self, debug=False):
        """
        Execute one complete navigation step with path following and speed control.
        
        This method represents the main control loop of the behavior agent, processing
        sensor data, updating the local planner, and generating appropriate control
        commands for the vehicle. It handles route completion detection and
        emergency stopping when necessary.
        
        Args:
            debug (bool): Enable debug mode for detailed logging and visualization
            
        Returns:
            Control: Vehicle control commands (steering, throttle, brake) for next time step
        """
        # Check if route is complete (within 3m of final destination)
        route_complete = (
            len(self._local_planner.waypoints_queue) < 1 and 
            calculate_distance(self.cur_waypoint.transform.location, 
                             self.last_target_waypoint.transform.location) < 3
        )
        
        if route_complete:
            # Signal route completion and initiate emergency stop
            self.end_route_flag = True
            self.cur_control = self.emergency_stop()
            return self.emergency_stop()
            
        # Calculate target speed considering behavior limits and safety margins
        target_speed = min(
            self.behavior.max_speed, 
            self.speed_limit - self.behavior.speed_lim_dist
        )
        
        # Execute local planner step to generate control commands
        control = self._local_planner.run_step(
            target_speed=target_speed, 
            debug=debug
        )
        
        # Update current control state for monitoring
        self.cur_control = control
        return control
    
    def run_step_w_trajectory(self, debug=False):
        """
        Execute trajectory-based navigation step with minimal control intervention.
        
        This method provides a trajectory-following mode where the vehicle follows
        a pre-defined trajectory with minimal active control. It's used for
        replaying recorded trajectories or testing specific motion patterns.
        
        Args:
            debug (bool): Enable debug mode for trajectory visualization
            
        Returns:
            Control: Neutral control commands for trajectory following mode
        """
        # Check if route is complete (within 3m of final destination)
        route_complete = (
            len(self._local_planner.waypoints_queue) < 1 and 
            calculate_distance(self.cur_waypoint.transform.location, 
                             self.last_target_waypoint.transform.location) < 3
        )
        
        if route_complete:
            # Signal route completion and initiate emergency stop
            self.end_route_flag = True
            self.cur_control = self.emergency_stop()
            return self.emergency_stop()
            
        # Execute local planner for trajectory processing (no active control)
        self._local_planner.run_step(
            target_speed=min(self.behavior.max_speed, 
                           self.speed_limit - self.behavior.speed_lim_dist), 
            debug=debug
        )
        
        # Return neutral control for trajectory following mode
        control = Control()
        control.steer = 0.0      # Neutral steering
        control.throttle = 0.0   # No active acceleration
        control.brake = 0.0      # No active braking
        control.hand_brake = False
        
        # Update current control state
        self.cur_control = control
        return control