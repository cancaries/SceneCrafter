
#!/usr/bin/env python3
"""
Advanced Behavior-Based Autonomous Vehicle Agent for Traffic Simulation
"""

import re
import random
import math
import numpy as np
from sklearn import neighbors
from enum import Enum
from navigation.controller import Control
from navigation.local_planner_behavior import LocalPlanner, RoadOption
from navigation.types_behavior import Cautious, Aggressive, Normal, ExtremeAggressive, \
    Cautious_fast, Aggressive_fast, Normal_fast, ExtremeAggressive_fast,\
    Cautious_highway, Aggressive_highway, Normal_highway, ExtremeAggressive_highway
import sys
from SceneController.scene import Scene
from .waypoint import Waypoint
from .tools.misc import is_within_distance, calculate_distance, positive, calculate_rotation
import copy

def normalize_angle(angle):
    """
    Normalize angle to [-π, π] range for consistent angular calculations.
    
    This function ensures all angles are represented within the standard range
    to prevent discontinuities in angular calculations and maintain consistent
    directional representations.
    
    Args:
        angle (float): Angle in radians, can be any real value
        
    Returns:
        float: Normalized angle within [-π, π] range
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

class BehaviorAgent():
    """
    Advanced autonomous vehicle agent with behavior-based decision making.
    
    This agent implements sophisticated navigation capabilities including:
    - Dynamic route planning and waypoint following
    - Multi-level collision avoidance (straight, ahead, general)
    - Behavior-based speed adaptation (cautious to aggressive)
    - Lane changing and overtaking maneuvers
    - Junction and intersection handling
    - Emergency response systems
    - Static and dynamic obstacle avoidance
    
    The agent maintains awareness of its surroundings through continuous sensor
    updates and adapts its behavior based on traffic conditions, road geometry,
    and predefined safety parameters.
    
    Attributes:
        vehicle_name (str): Unique identifier for this agent
        behavior: Behavior profile defining driving characteristics
        _local_planner: Local path planner for waypoint navigation
        cur_waypoint: Current position waypoint
        speed (float): Current vehicle speed in km/h
        bounding_box: Vehicle dimensions for collision detection
        
    Example:
        >>> config = {'name': 'ego_vehicle', 'behavior': 'normal', ...}
        >>> agent = BehaviorAgent(config)
        >>> control = agent.run_step()
    """

    def __init__(self, config):
        """
        Initialize the behavior agent with configuration parameters.
        
        Sets up the agent's initial state, behavior profile, and planning systems.
        Configures safety parameters, speed limits, and route planning based on
        the provided configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary containing:
                - name: Vehicle identifier
                - behavior: Driving behavior type ('cautious', 'normal', 'aggressive', 'extreme_aggressive')
                - speed_type: Speed profile ('normal', 'fast', 'highway')
                - initial_path: List of waypoints for initial route
                - vehicle_type: Type of vehicle
                - vehicle_bbox: Bounding box dimensions [length, width]
                - f_len: Front axle to center distance
                - r_len: Rear axle to center distance
                - control_angel: Steering angle parameter
                - max_speed: Optional maximum speed override
                
        Note:
            The behavior profile determines safety parameters like braking distance,
            proximity thresholds, and overtaking aggressiveness. Speed variations
            are added randomly to create more realistic traffic patterns.
        """
        self.vehicle_name = config['name']
        self.look_ahead_steps = 0
        self.end_route_flag = False

        # Vehicle information
        self.cur_waypoint = config['initial_path'][0]
        self.last_waypoint = config['initial_path'][0]
        self.speed = 0 # km/h
        self.velocity_xy = [0, 0] # km/h
        self.acceleration = 0 # m/s^2
        self.acceleration_xy = [0, 0] # m/s^2
        self.omega = 0 # rad/s
        self.steer_value = 0
        self.speed_limit = 0
        self.direction = None
        self.incoming_direction = None
        self.incoming_waypoint = None
        self.cur_status = None
        self.start_waypoint = None
        self.end_waypoint = None
        self.plan_path = config['initial_path']
        self.min_speed = 5
        self.behavior = None
        self.vehicle_type = config['vehicle_type']
        self.bounding_box = config['vehicle_bbox']
        self.f_len = config['f_len']
        self.r_len = config['r_len']
        self.control_angel = config['control_angel']
        self.speed_type = config['speed_type']
        self.vehicle_type = config['vehicle_type']
        self._sampling_resolution = 4.5
        self.cur_control = Control()
        self.if_overtake = False
        self.overtake_direction = None
        self.if_tailgate = False
        self.overtake_end_waypoint = None
        self.vehicle_name_to_overtake = None
        self.if_static = False
        
        # Initialize behavior profile based on configuration
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

        # Apply speed variations for realistic traffic patterns
        if 'max_speed' in config.keys():
            self.behavior.max_speed = config['max_speed']
        
        # Add random speed variation to create diverse traffic
        if 'ego' in self.vehicle_name:
            self.behavior.max_speed = self.behavior.max_speed * (1 + random.uniform(-0.15, 0.15))
        else:
            self.behavior.max_speed = self.behavior.max_speed * (1 + random.uniform(-0.2, 0.2))
            
        self.original_braking_distance = self.behavior.braking_distance
        
        # Adjust safety parameters based on vehicle speed and size
        if self.behavior.max_speed < 50.0:
            self.behavior.min_proximity_threshold = self.behavior.min_proximity_threshold + self.behavior.max_speed / 3.6 / 1.5 + max(self.bounding_box[0], self.bounding_box[1])
        else:
            self.behavior.min_proximity_threshold = self.behavior.min_proximity_threshold * 2.0 * (self.behavior.max_speed/100) + self.behavior.max_speed / 3.6 + max(self.bounding_box[0], self.bounding_box[1])

        self.cur_speed_limit = self.behavior.max_speed
        self.update_braking_distance()
        
        # Set maximum acceleration and braking based on vehicle capabilities
        self.max_acceleration = max(self.behavior.max_speed / 3.6 / 3.0, 6.0)
        self.max_braking = max(self.behavior.max_speed / 3.6 / 2.0, 8.0)
        self.scenario_trigger = self.behavior.scenario_trigger
        
        # Initialize local planner for waypoint navigation
        self._local_planner = LocalPlanner(self)
        self.speed_limit = self.behavior.max_speed
        self._local_planner.set_speed(self.speed_limit)
        self._local_planner.set_global_plan(self.plan_path, clean=True)
        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)

    def update_braking_distance(self):
        """
        Dynamically update braking distance based on current speed and conditions.
        
        Adjusts the braking distance parameter based on current vehicle speed,
        speed limits, and vehicle dimensions. This ensures realistic stopping
        distances that account for reaction time and vehicle physics.
        
        The calculation considers:
        - Current speed vs speed limit
        - Vehicle size (larger vehicles need more distance)
        - Behavior profile (aggressive vs cautious)
        """
        tmp_standard = max(self.speed, self.cur_speed_limit/2)
        if self.behavior.max_speed < 50.0:
            self.behavior.braking_distance = self.original_braking_distance * (self.behavior.max_speed/50) + self.cur_speed_limit / 3.6 / 6.0 + max(self.bounding_box[0], self.bounding_box[1])
        else:
            self.behavior.braking_distance = self.original_braking_distance * 2.0 * (self.behavior.max_speed/100) + self.cur_speed_limit / 3.6 / 4.0 + max(self.bounding_box[0], self.bounding_box[1])

    def update_information(self, update_transform):
        """
        Update agent state based on current vehicle position and surroundings.
        
        This critical method synchronizes the agent's internal state with the
        actual vehicle position in the simulation. It updates:
        - Current waypoint and lane information
        - Direction and road option planning
        - Look-ahead distance for planning
        - Neighboring lane relationships for lane changes
        
        Args:
            update_transform: Current vehicle transform containing location and rotation
            
        Note:
            During driving, the nearest lane may change due to vehicle movement
            across lanes, so we need to re-identify the current lane_id and
            lane_point_idx. The cur_point_index is not critical for subsequent
            processes as we mainly use lane_id for neighbor relationships.
        """
        self.direction = self._local_planner.target_road_option
        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW
        self.look_ahead_steps = int((self.speed_limit) / 15)
        
        location = update_transform.location
        rotation = update_transform.rotation
        
        # Update current lane information based on actual position
        # During driving, may need to re-identify current lane due to lane changes
        neighbour_ids = Scene._map.get_all_neighbor_lane_ids(self.cur_waypoint.lane_id)
        cur_lane_id, cur_point_index, _ = Scene._map.find_nearest_lane_point(location)
        
        # If current lane not in neighbor lanes, keep original lane_id
        # cur_point_index correctness is not critical as it's not used subsequently
        if cur_lane_id not in neighbour_ids:
            cur_lane_id = self.cur_waypoint.lane_id
            cur_point_index = self.cur_waypoint.lane_point_idx
            
        cur_waypoint_config = Scene._map.build_waypoint_config(location, rotation, self.direction, cur_lane_id, cur_point_index)
        self.cur_waypoint = Waypoint(cur_waypoint_config)
        
        # Update incoming waypoint and direction for planning
        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)
            
        # Update current status with different look-ahead for ego vs other vehicles
        if self.vehicle_name == 'ego_vehicle':
            _, self.cur_status = self._local_planner.get_incoming_waypoint_and_direction(
                steps=int((self.speed) / 11))
        else:
            _, self.cur_status = self._local_planner.get_incoming_waypoint_and_direction(
                steps=2)
        
        # Handle edge cases where waypoints might be None
        if self.incoming_waypoint is None:
            self.incoming_waypoint = self.cur_waypoint
        if self.incoming_direction is None:
            self.incoming_direction = RoadOption.LANEFOLLOW
            
        # Update braking distance based on new speed and conditions
        self.update_braking_distance()

    def set_last_waypoint(self, last_waypoint):
        """
        Store the last valid waypoint for reference and debugging.
        
        Args:
            last_waypoint: The last valid waypoint position
        """
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
        Set vehicle speed and update related velocity components.
        
        Updates both the scalar speed and the 2D velocity vector based on
        current vehicle orientation.
        
        Args:
            speed (float): New speed in km/h
        """
        self.speed = speed
        self.velocity_xy = [speed * np.cos(self.cur_waypoint.get_rotation()[2]),
                           speed * np.sin(self.cur_waypoint.get_rotation()[2])]

    def set_omega(self, omega):
        """
        Set angular velocity for vehicle rotation control.
        
        Args:
            omega (float): Angular velocity in rad/s
        """
        self.omega = omega

    def set_static(self, if_static):
        """
        Set whether the vehicle should remain static (parked).
        
        Args:
            if_static (bool): True if vehicle should not move
        """
        self.if_static = if_static

    def set_steer_value(self, steer_value):
        """
        Set steering value for direct control.
        
        Args:
            steer_value (float): Steering angle value
        """
        self.steer_value = steer_value

    def set_acceleration(self, acceleration):
        """
        Set acceleration and update related acceleration components.
        
        Updates both scalar acceleration and 2D acceleration vector based on
        current vehicle orientation.
        
        Args:
            acceleration (float): Linear acceleration in m/s²
        """
        self.acceleration = acceleration
        self.acceleration_xy = [acceleration * np.cos(self.cur_waypoint.get_rotation()[2]),
                               acceleration * np.sin(self.cur_waypoint.get_rotation()[2])]

    def rearrange_route(self, additional_route):
        """
        Integrate additional route segment into current plan.
        
        This method allows dynamic route modification by adding new waypoints
        to the current plan while maintaining connectivity to the original destination.
        
        Args:
            additional_route: List of waypoints to insert into current route
            
        Note:
            If the additional route is invalid or empty, the current plan is preserved.
            The method handles route connectivity by tracing from the end of the
            additional route back to the original destination.
        """
        if additional_route is None or len(additional_route) == 0:
            return
            
        end_point_of_additional_route = copy.deepcopy(additional_route[-1])
        end_point_of_plan_path = self.plan_path[-1]
        new_route = self._trace_route(end_point_of_additional_route, end_point_of_plan_path)
        
        if new_route:
            route_combined = additional_route + new_route[1:]
            route_combined = route_combined[1:]  # Remove duplicate waypoint
            self._local_planner.set_global_plan(route_combined, clean=True, clean_global=True)
            self.plan_path = copy.deepcopy(route_combined)
        else:
            print("Route Invalid. Keeping the current plan.")

    def reroute(self, cur_loc):
        """
        Generate new route when approaching destination or blocked path.
        
        Creates a completely new route plan from the current location to a
        new destination, useful for handling blocked paths or reaching final destinations.
        
        Args:
            cur_loc: Current vehicle location for route planning
        """
        print("Target almost reached, setting new destination...")
        new_route = Scene._map.generate_overall_plan_waypoints(cur_loc)
        self._local_planner.set_global_plan(new_route, clean=True, clean_global=True)

    def reroute_all(self, new_plan):
        """
        Replace entire route plan with new waypoints.
        
        Completely replaces the current route with a new plan, useful for
        scenario-based testing or external route modifications.
        
        Args:
            new_plan: Complete new route as list of waypoints
        """
        new_plan = Scene._map.refine_plan_waypoints(new_plan, 2.0)
        self.plan_path = copy.deepcopy(new_plan)
        self._local_planner.set_global_plan(new_plan, clean=True, clean_global=True)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        Generate optimal route between two waypoints.
        
        Uses the global routing system to find the shortest path between
        start and end waypoints, considering road network topology and traffic rules.
        
        Args:
            start_waypoint: Starting position waypoint
            end_waypoint: Destination waypoint
            
        Returns:
            list: Optimal route as sequence of waypoints
        """
        return Scene.generate_route_w_waypoints(start_waypoint, end_waypoint)

    def emergency_stop(self):
        """
        Generate emergency braking control command.
        
        Creates a control command that immediately applies maximum braking
        force to bring the vehicle to a complete stop as quickly as possible.
        
        Returns:
            Control: Emergency stop control command with full brake
        """
        control = Control()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        self.cur_control = control
        return control

    def slow_down(self):
        """
        Generate gradual braking control command.
        
        Similar to emergency_stop but allows for more controlled deceleration.
        Useful for situations requiring speed reduction rather than full stop.
        
        Returns:
            Control: Braking control command
        """
        control = Control()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        self.cur_control = control
        return control

    def get_trajectory(self, step_num=10):
        """
        Get planned trajectory for specified number of steps.
        
        Returns the agent's planned future trajectory based on current route
        and speed planning.
        
        Args:
            step_num (int): Number of future steps to include in trajectory
            
        Returns:
            list: Planned trajectory as sequence of waypoints
        """
        return self._local_planner.get_trajectory(step_num)

    def get_final_trajectory(self, step_num=10):
        """
        Get final refined trajectory for specified number of steps.
        
        Similar to get_trajectory but returns a more refined version suitable
        for final execution or visualization.
        
        Args:
            step_num (int): Number of future steps to include
            
        Returns:
            list: Refined trajectory as sequence of waypoints
        """
        return self._local_planner.get_final_trajectory(step_num)

    def _bh_is_vehicle_hazard(self, ego_wpt, vehicle_info_list,
                          proximity_th, up_angle_th, low_angle_th=0, lane_offset=0, scenario_flag=False):
        """
        Check for hazardous vehicles within specified parameters.
        
        Comprehensive vehicle hazard detection system that evaluates other vehicles
        based on proximity, angular position, lane relationships, and future trajectory
        interactions. This is the core collision avoidance component.
        
        Args:
            ego_wpt: Ego vehicle waypoint information
            vehicle_info_list: List of other vehicle information dictionaries
            proximity_th: Distance threshold for hazard detection
            up_angle_th: Upper angle limit for detection cone
            low_angle_th: Lower angle limit for detection cone (default 0)
            lane_offset: Lane offset for lane change scenarios (-1 left, 1 right)
            scenario_flag: Enable scenario-specific lane checking
            
        Returns:
            tuple: (is_hazard, closest_vehicle_name, closest_distance)
            
        Detection Process:
        1. Filter vehicles by lane relationships
        2. Check angular position within detection cone
        3. Evaluate future trajectory interactions
        4. Calculate distances and identify closest hazard
        """
        ego_loc = ego_wpt.transform.location
        ego_yaw = ego_wpt.transform.rotation[2]
        ego_lane_id = ego_wpt.lane_id
        ego_exit_lane_id = self.get_next_lane_id()
        
        # Determine relevant neighbor lanes based on lane offset
        neighbour_ids = []
        if lane_offset == -1:
            neighbour_ids = Scene._map.get_left_neighbor_lane_ids(ego_lane_id)
        elif lane_offset == 1:
            neighbour_ids = Scene._map.get_right_neighbor_lane_ids(ego_lane_id)
            
        # Store qualifying vehicle information
        vehicle_dict = {}
        
        # Evaluate each target vehicle
        for target_vehicle_info in vehicle_info_list:
            # Skip static vehicles if configured to ignore them
            if 'static' in target_vehicle_info['name'] and self.behavior.ignore_static:
                continue
                
            target_vehicle_loc = target_vehicle_info['location']
            target_vehicle_yaw = target_vehicle_info['yaw']
            target_lane_id = target_vehicle_info['lane_id']
            target_exit_lane_id = target_vehicle_info['exit_lane_id']
            
            # Check lane relationships for scenario-specific cases
            if scenario_flag:
                if target_lane_id not in neighbour_ids:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if next_wpt is None:
                        next_wpt = ego_wpt
                    
                    # Determine next neighbor lanes
                    next_neighbour_ids = []
                    if lane_offset == -1:
                        next_neighbour_ids = Scene._map.get_left_neighbor_lane_ids(next_wpt.lane_id)
                    elif lane_offset == 1:
                        next_neighbour_ids = Scene._map.get_right_neighbor_lane_ids(next_wpt.lane_id)
                        
                    if target_lane_id not in next_neighbour_ids:
                        continue
            
            # Check future trajectory interaction
            if_interaction = Scene.judge_future_trajectory_interaction(self.vehicle_name, target_vehicle_info['name'])
            
            # Check if vehicle is within detection parameters
            if is_within_distance(target_vehicle_loc, ego_loc, ego_yaw,
                              proximity_th, up_angle_th, low_angle_th) and if_interaction:
                distance = calculate_distance(target_vehicle_loc, ego_loc)
                # Store vehicle information with distance and lane data
                vehicle_dict[target_vehicle_info['name']] = distance, target_lane_id, target_exit_lane_id, target_vehicle_loc
            

        # 如果字典不为空，找到距离最小的车辆
        if vehicle_dict:
            min_distance = float('inf')
            closest_vehicle_name = None
            filtered_vehicle_e2t_dict = {}
            filtered_vehicle_t2e_dict = {}
            filtered_vehicle_remaining_dict = {}
            for vehicle_name, (distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc) in vehicle_dict.items():
                ego_target_side_flag = Scene.judge_future_trajectory_interaction_side(self.vehicle_name, vehicle_name)
                target_ego_side_flag = Scene.judge_future_trajectory_interaction(vehicle_name, self.vehicle_name)
                if ego_target_side_flag and target_ego_side_flag:
                    filtered_vehicle_remaining_dict[vehicle_name] = distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc
                if ego_target_side_flag and not target_ego_side_flag:
                    return (True, vehicle_name, distance)
            if filtered_vehicle_remaining_dict:
                for vehicle_name, (distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc) in filtered_vehicle_remaining_dict.items():
                    if distance < min_distance:
                        min_distance = distance
                        closest_vehicle_name = vehicle_name
                closest_distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc = filtered_vehicle_remaining_dict[closest_vehicle_name]
                if ego_exit_lane_id is None:
                    ego_exit_lane_distance = calculate_distance(ego_wpt.transform.location, Scene._map.features[ego_lane_id].polyline[-1])
                else:
                    ego_exit_lane_distance = calculate_distance(ego_wpt.transform.location, Scene._map.features[ego_exit_lane_id].polyline[0])
                if ref_exit_lane_id is None:
                    ref_exit_lane_distance = calculate_distance(ref_vehicle_loc, Scene._map.features[ref_lane_id].polyline[-1])
                else:
                    ref_exit_lane_distance = calculate_distance(ref_vehicle_loc,Scene._map.features[ref_exit_lane_id].polyline[0])
                if ref_exit_lane_distance > ego_exit_lane_distance:
                    return (False, None, -1)
                
                return (True, closest_vehicle_name, closest_distance)
            
        return (False, None, -1)

    def _bh_is_vehicle_hazard_straight(self, ego_wpt, vehicle_info_list,
                           proximity_th, up_angle_th, low_angle_th=0, lane_offset=0):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle. We also check the next waypoint, just to be
        sure there's not a sudden road id change.
        """
        ego_loc = ego_wpt.transform.location
        ego_yaw = ego_wpt.transform.rotation[2]
        ego_lane_id = ego_wpt.lane_id
        ego_exit_lane_id = self.get_next_lane_id()
        ego_exit_lanes = Scene._map.features[self.cur_waypoint.lane_id].exit_lanes
        neighbour_lanes = Scene._map.get_all_neighbor_lane_ids(ego_lane_id)
        vehicle_dict = {}
        for target_vehicle_info in vehicle_info_list:

            target_vehicle_loc = target_vehicle_info['location']
            target_vehicle_yaw = target_vehicle_info['yaw']
            target_lane_id = target_vehicle_info['lane_id']
            target_exit_lane_id = target_vehicle_info['exit_lane_id']
            angle_diff = normalize_angle(ego_yaw - target_vehicle_yaw)
            if abs(angle_diff) > np.pi/6 and abs(angle_diff) < np.pi/6*5:
                continue
            if target_lane_id != ego_lane_id and target_lane_id not in ego_exit_lanes and target_lane_id not in neighbour_lanes:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                if next_wpt is None:
                    next_wpt = ego_wpt
                if  target_lane_id != next_wpt.lane_id:
                    continue
            if is_within_distance(target_vehicle_loc, ego_loc,
                                  ego_yaw,
                                  proximity_th, up_angle_th, low_angle_th) :
                distance = calculate_distance(target_vehicle_loc, ego_loc)
                vehicle_dict[target_vehicle_info['name']] = distance, target_lane_id, target_exit_lane_id, target_vehicle_loc
        if vehicle_dict:
            min_distance = float('inf')
            closest_vehicle_name = None
            for vehicle_name, (distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc) in vehicle_dict.items():
                if distance < min_distance:
                    min_distance = distance
                    closest_vehicle_name = vehicle_name
            closest_distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc = vehicle_dict[closest_vehicle_name]
            if ego_exit_lane_id is None:
                ego_exit_lane_distance = calculate_distance(ego_wpt.transform.location, Scene._map.features[ego_lane_id].polyline[-1])
            else:
                ego_exit_lane_distance = calculate_distance(ego_wpt.transform.location, Scene._map.features[ego_exit_lane_id].polyline[0])
            if ref_exit_lane_id is None:
                ref_exit_lane_distance = calculate_distance(ref_vehicle_loc, Scene._map.features[ref_lane_id].polyline[-1])
            else:
                ref_exit_lane_distance = calculate_distance(ref_vehicle_loc,Scene._map.features[ref_exit_lane_id].polyline[0])
            if ref_exit_lane_distance > ego_exit_lane_distance:
                return (False, None, closest_distance)
            
            return (True, closest_vehicle_name, closest_distance)
        return (False, None, -1)


    def _bh_is_vehicle_hazard_ahead(self, ego_wpt, vehicle_info_list,
                           proximity_th, up_angle_th, low_angle_th=0, lane_offset=0):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle. We also check the next waypoint, just to be
        sure there's not a sudden road id change.
        """
        ego_loc = ego_wpt.transform.location
        ego_yaw = ego_wpt.transform.rotation[2]
        ego_lane_id = ego_wpt.lane_id
        ego_exit_lane_id = self.get_next_lane_id()
        ego_exit_lanes = Scene._map.features[self.cur_waypoint.lane_id].exit_lanes
        vehicle_dict = {}
        for target_vehicle_info in vehicle_info_list:
            target_vehicle_loc = target_vehicle_info['location']
            target_vehicle_yaw = target_vehicle_info['yaw']
            target_lane_id = target_vehicle_info['lane_id']
            target_exit_lane_id = target_vehicle_info['exit_lane_id']
            angle_diff = normalize_angle(ego_yaw - target_vehicle_yaw)
            if abs(angle_diff) > np.pi/6*5:
                continue
            if target_lane_id != ego_lane_id and target_lane_id not in ego_exit_lanes:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=2)[0]
                if next_wpt is None:
                    next_wpt = ego_wpt
                if  target_lane_id != next_wpt.lane_id:
                    continue
            if is_within_distance(target_vehicle_loc, ego_loc,
                                  ego_yaw,
                                  proximity_th, up_angle_th, low_angle_th):
                distance = calculate_distance(target_vehicle_loc, ego_loc)
                vehicle_dict[target_vehicle_info['name']] = distance, target_lane_id, target_exit_lane_id, target_vehicle_loc
        if vehicle_dict:
            min_distance = float('inf')
            closest_vehicle_name = None
            for vehicle_name, (distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc) in vehicle_dict.items():
                if distance < min_distance:
                    min_distance = distance
                    closest_vehicle_name = vehicle_name
            closest_distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc = vehicle_dict[closest_vehicle_name]
            return (True, closest_vehicle_name, closest_distance)
        
        return (False, None, -1)


    def _overtake(self, waypoint, vehicle_list):
        """
        This method is in charge of overtaking behaviors.
        """

        left_lines = waypoint.get_left_lane()
        right_lines = waypoint.get_right_lane()
        location = waypoint.transform.get_location()
        if left_lines is not None:
            new_vehicle_state, vehicle_name, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=-1,scenario_flag=True)
            if not new_vehicle_state:
                self.behavior.overtake_counter = 200
                driving_mode, additional_route = Scene._map.get_plan_waypoints_w_refine(location, driving_mode='CHANGELANELEFT')
                if additional_route is not None:
                    self.overtake_end_waypoint = additional_route[-1]
                    self.rearrange_route(additional_route)
                    self.if_overtake = True
                    self.vehicle_name_to_overtake = vehicle_name
                    self.overtake_direction = 'left'
        elif right_lines is not None:
            new_vehicle_state, vehicle_name, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=1,scenario_flag=True)
            if not new_vehicle_state:
                self.behavior.overtake_counter = 200
                driving_mode, additional_route = Scene._map.get_plan_waypoints(location, driving_mode='CHANGELANERIGHT')
                if additional_route is not None:
                    self.overtake_end_waypoint = additional_route[-1]
                    self.rearrange_route(additional_route)
                    self.if_overtake = True
                    self.vehicle_name_to_overtake = vehicle_name
                    self.overtake_direction = 'right'
        else:
            new_vehicle_state, vehicle_name, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=30)
            if vehicle_name:
                if Scene._agent_dict[vehicle_name].cur_waypoint.lane_id == waypoint.lane_id:
                    self.vehicle_name_to_overtake = vehicle_name
                    self._force_overtake()

    def _force_overtake(self,direction='left'):
        cur_plan = list(self._local_planner._waypoint_buffer)+list(self._local_planner.waypoints_queue)
        driving_mode, additional_route = Scene._map.generate_overtake_path_from_reference_path(cur_plan,direction=direction)
        if additional_route is not None: 
            self.behavior.overtake_counter = 200
            self.overtake_end_waypoint = additional_route[-1]
            self.rearrange_route(additional_route)
            self.overtake_direction = direction
            self.if_overtake = True
        else:
            pass
        
    def _force_turn_back(self):
        next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=1)[0]
        if next_wpt is None:
            return
        if self.overtake_direction == 'left':
            direction = 'CHANGELANERIGHT'
        elif self.overtake_direction == 'right':
            direction = 'CHANGELANELEFT'
        driving_mode, additional_route = Scene._map.generate_waypoint_path_from_two_points(self.cur_waypoint, next_wpt, direction=direction)
        self.rearrange_route(additional_route)
        self.overtake_direction = None
        self.if_overtake = False
        
    # 原地掉头
    def _force_turn_around(self):
        next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=1)[0]
        if next_wpt.is_junction:
            return
        driving_mode, turn_around_route = Scene._map.generate_turn_around_path(next_wpt)
        if turn_around_route is not None:
            self.reroute_all(turn_around_route)


    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.
        """

        left_lines = waypoint.get_left_lane()
        right_lines = waypoint.get_right_lane()
        location = waypoint.transform.location
        behind_vehicle_state, behind_vehicle, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
            self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self.speed < Scene.get_vehicle_speed(behind_vehicle):
            if  right_lines is not None:
                new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1,scenario_flag=True)
                if not new_vehicle_state:
                    # print("Tailgating, moving to the right!")
                    self.behavior.tailgate_counter = 200
                    driving_mode, additional_route = Scene._map.get_plan_waypoints_w_refine(location, driving_mode='CHANGELANERIGHT')
                    self.rearrange_route(additional_route)
                    self.if_tailgate = True
            elif left_lines is not None:
                new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1,scenario_flag=True)
                if not new_vehicle_state:
                    # print("Tailgating, moving to the left!")
                    self.behavior.tailgate_counter = 200
                    driving_mode, additional_route = Scene._map.get_plan_waypoints_w_refine(location, driving_mode='CHANGELANELEFT')
                    self.rearrange_route(additional_route)
                    self.if_tailgate = True

    def get_next_lane_id(self):
        """
        This method is in charge of getting the next lane id.

        :return next_lane_id: next lane id
        """
        cur_lane_id = self.cur_waypoint.lane_id
        next_lane_id = None
        step_num = 1
        
        # Search ahead in the planned route
        while next_lane_id is None or next_lane_id == cur_lane_id:
            if step_num > 10:  # Limit search depth
                break
                
            next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=step_num)[0]
            if next_wpt is None:
                next_wpt = self.cur_waypoint
                next_lane_id = next_wpt.lane_id
                break
                
            next_lane_id = next_wpt.lane_id
            if next_lane_id != cur_lane_id:
                return next_lane_id
            step_num += 1
            
        return None

    def get_near_vehicle_info(self, max_distance=100):
        """
        Gather information about nearby vehicles within specified distance.
        
        Collects comprehensive information about all vehicles within the
        specified maximum distance, including position, orientation, and
        lane information for hazard detection and planning.
        
        Args:
            max_distance (float): Maximum distance to consider vehicles (default 100m)
            
        Returns:
            list: Vehicle information dictionaries with keys:
                - name: Vehicle identifier
                - location: 3D position
                - yaw: Orientation angle
                - lane_id: Current lane identifier
                - exit_lane_id: Next lane in planned route
        """
        vehicle_info_list = []
        
        for vehicle_name, agent in Scene._agent_dict.items():
            if vehicle_name != self.vehicle_name and 'static' not in vehicle_name:
                vehicle_wp = Scene.get_vehicle_waypoint(vehicle_name)
                vehicle_loc = vehicle_wp.transform.location
                yaw = vehicle_wp.transform.rotation[2]
                distance = calculate_distance(vehicle_loc, self.cur_waypoint.transform.location)
                
                # Get exit lane ID for the vehicle
                vehicle_agent = Scene._agent_dict[vehicle_name]
                exit_lane_id = vehicle_agent.get_next_lane_id()
                
                if distance < max_distance:
                    vehicle_info_list.append({
                        'name': vehicle_name,
                        'location': vehicle_loc,
                        'yaw': yaw,
                        'lane_id': vehicle_wp.lane_id,
                        'exit_lane_id': exit_lane_id
                    })

        return vehicle_info_list

    def collision_and_car_avoid_manager(self, waypoint):
        """
        Main collision avoidance manager for general driving scenarios.
        
        Coordinates the primary collision avoidance system, evaluating all
        nearby vehicles and determining appropriate responses including
        braking, lane changes, or overtaking maneuvers.
        
        Args:
            waypoint: Current vehicle waypoint for context
            
        Returns:
            tuple: (vehicle_state, vehicle_name, distance)
            - vehicle_state: True if hazard detected, False otherwise
            - vehicle_name: Identifier of closest hazardous vehicle
            - distance: Distance to hazardous vehicle
        """
        detect_distance = max(self.behavior.min_proximity_threshold, self.speed / 2)
        vehicle_info_list = self.get_near_vehicle_info(max_distance=detect_distance)

        # Check hazard based on current direction/maneuver
        if self.direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, vehicle_info_list, detect_distance, up_angle_th=180, lane_offset=-1)
        elif self.direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, vehicle_info_list, detect_distance, up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, vehicle_info_list, detect_distance, up_angle_th=120)

        # Handle overtaking and tailgating scenarios
        if self.scenario_trigger:
            # Check for overtaking opportunities
            if (vehicle_state and not waypoint.is_junction and self.speed > 10 and
                    self.behavior.overtake_counter == 0 and self.speed > Scene.get_vehicle_speed(vehicle) and 
                    not self.if_overtake):
                self._overtake(waypoint, vehicle_info_list)

            # Check for tailgating situations
            elif (not vehicle_state and not waypoint.is_junction and self.speed > 10 and
                    self.behavior.tailgate_counter == 0 and not self.if_overtake):
                self._tailgating(waypoint, vehicle_info_list)

        return vehicle_state, vehicle, distance

    def straight_collision_and_car_avoid_manager(self, waypoint):
        """
        Collision avoidance for straight-ahead driving scenarios.
        
        Specialized collision detection focused on vehicles directly ahead
        in the same lane, with reduced detection angles for more focused
        hazard identification.
        
        Args:
            waypoint: Current vehicle waypoint
            
        Returns:
            tuple: (vehicle_state, vehicle_name, distance) for straight-ahead hazards
        """
        detect_distance = min(self.behavior.min_proximity_threshold / 1.5, self.speed_limit / 3.0)
        vehicle_info_list = self.get_near_vehicle_info(max_distance=detect_distance)

        vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard_straight(
            waypoint, vehicle_info_list, detect_distance, up_angle_th=30)

        return vehicle_state, vehicle, distance
    
    def ahead_collision_and_car_avoid_manager(self, waypoint):
        """
        Extended collision detection for vehicles further ahead.
        
        Longer-range collision detection focused on vehicles in the distance,
        using extended braking distance parameters for early hazard detection.
        
        Args:
            waypoint: Current vehicle waypoint
            
        Returns:
            tuple: (vehicle_state, vehicle_name, distance) for distant hazards
        """
        vehicle_info_list = self.get_near_vehicle_info()
        vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard_ahead(
            waypoint, vehicle_info_list, self.behavior.braking_distance * 2.0, up_angle_th=45)

        return vehicle_state, vehicle, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Manage car-following behavior when following another vehicle.
        
        Implements intelligent car-following using time-to-collision (TTC)
        calculations to maintain safe following distances while maximizing
        traffic flow efficiency.
        
        Args:
            vehicle: Vehicle to follow (target vehicle identifier)
            distance: Current distance to target vehicle
            debug (bool): Enable debug output
            
        Returns:
            Control: Appropriate control command for following behavior
            
        Algorithm:
        - TTC < 2*safety_time: Reduce speed to maintain safety
        - 2*safety_time < TTC < 3*safety_time: Follow target speed
        - TTC > 3*safety_time: Maintain normal speed limit
        """
        vehicle_speed = Scene.get_vehicle_speed(vehicle)
        delta_v = max(1, (self.speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)
        
        # Adjust speed based on time-to-collision
        if 2 * self.behavior.safety_time > ttc > 0.0:
            # Too close - reduce speed
            control = self._local_planner.run_step(
                target_speed=vehicle_speed-1.0, debug=debug)
        elif 3 * self.behavior.safety_time > ttc >= self.behavior.safety_time:
            # Safe following distance - match target speed
            control = self._local_planner.run_step(
                target_speed=self.cur_speed_limit, debug=debug)
        else:
            # Normal driving - maintain speed limit
            control = self._local_planner.run_step(
                target_speed=self.cur_speed_limit, debug=debug)

        return control
    
    def list_vehicles_in_junction(self, waypoint):
        """
        Identify all vehicles currently within junction boundaries.
        
        Useful for junction priority management and collision avoidance
        at intersections.
        
        Args:
            waypoint: Current vehicle waypoint for junction context
            
        Returns:
            list: Vehicles within junction with location and lane information
        """
        vehicle_list = []
        for vehicle_name, agent in Scene._agent_dict.items():
            if vehicle_name != self.vehicle_name:
                vehicle_wp = Scene.get_vehicle_waypoint(vehicle_name)
                if vehicle_wp.is_junction:
                    vehicle_list.append({
                        'name': vehicle_name,
                        'location': vehicle_wp.transform.location,
                        'lane_id': vehicle_wp.lane_id
                    })
        return vehicle_list

    def calculate_speed_limit(self, steer_angle, max_speed, steer_threshold=0.05, sensitivity=0.3):
        """
        Calculate appropriate speed limit based on steering angle and road conditions.
        
        Implements dynamic speed adaptation that reduces speed for curves and
        turns based on steering angle magnitude. This creates more realistic
        cornering behavior.
        
        Args:
            steer_angle (float): Current steering angle
            max_speed (float): Maximum allowed speed
            steer_threshold (float): Minimum steering angle to trigger speed reduction
            sensitivity (float): How aggressively speed reduces with steering
            
        Returns:
            float: Adjusted speed limit for current conditions
            
        The calculation considers:
        - Steering angle magnitude
        - Road type (turn, lane change, straight)
        - Minimum speed constraints to prevent complete stops
        """
        abs_steer = abs(steer_angle)
        
        # Determine road type scaling factor
        scale_factor = 1.0
        if self.cur_status in (RoadOption.TURNLEFT, RoadOption.TURNRIGHT, RoadOption.INTURN):
            scale_factor = 1.5  # Sharp turns require more speed reduction
        elif self.cur_status in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            scale_factor = 1.4  # Lane changes need moderate reduction
        elif self.cur_status in (RoadOption.LANEFOLLOW, RoadOption.STRAIGHT):
            scale_factor = 1.0  # Normal straight driving

        # Calculate speed reduction factor based on steering
        speed_factor = np.exp(-sensitivity * max(0, abs_steer - steer_threshold))
        
        # Apply scaling and minimum speed constraint
        cur_speed_limit = max_speed * speed_factor / scale_factor
        min_speed_limit = 0.3 * max_speed  # Prevent complete stops
        cur_speed_limit = max(min_speed_limit, cur_speed_limit)
        
        return cur_speed_limit

    def run_step(self, debug=False):
        """
        Execute one complete step of autonomous navigation.
        
        This is the main control loop that orchestrates all aspects of
        autonomous driving including:
        - Speed adaptation based on road geometry
        - Multi-layer collision detection and avoidance
        - Route following and replanning
        - Emergency response systems
        - Overtaking and lane change management
        
        Args:
            debug (bool): Enable debug output for development
            
        Returns:
            Control: Final control command for vehicle actuation
            
        Control Flow:
        1. Calculate dynamic speed limits
        2. Check for route completion
        3. Evaluate multiple collision detection layers
        4. Apply appropriate response (braking, following, rerouting)
        5. Handle special cases (static obstacles, overtaking)
        """
        cur_steer = Scene.get_vehicle_control(self.vehicle_name).steer
        target_flag = False
        
        # Calculate dynamic speed limit based on current conditions
        self.cur_speed_limit = self.calculate_speed_limit(cur_steer, self.speed_limit)
        
        # Check for route completion
        if len(self._local_planner.waypoints_queue) < 1:
            self.end_route_flag = True
            self.cur_control = self.emergency_stop()
            return self.emergency_stop()

        # Handle speed limit violations
        if self.speed > self.cur_speed_limit:
            control = self._local_planner.run_step(
                target_speed=self.cur_speed_limit, debug=debug)
            self.cur_control = control
            return control

        # Update behavior counters
        control = None
        if self.behavior.tailgate_counter > 0:
            self.behavior.tailgate_counter -= 1
        if self.behavior.overtake_counter > 0:
            self.behavior.overtake_counter -= 1

        # Handle overtaking completion
        if self.if_overtake:
            if calculate_distance(self.overtake_end_waypoint.transform.location, 
                                self.cur_waypoint.transform.location) < 2:
                # Reset overtaking state
                self.if_overtake = False
                self.overtake_direction = None
                self.overtake_end_waypoint = None
                self.vehicle_name_to_overtake = None
            if self.if_overtake and self.behavior.overtake_counter % 40 == 0:
                ego_loc = self.cur_waypoint.transform.location
                ego_yaw = self.cur_waypoint.transform.rotation[2]
                if not self.vehicle_name_to_overtake is None and self.vehicle_name_to_overtake in Scene._agent_dict:
                    target_vehicle_wp = Scene.get_vehicle_waypoint(self.vehicle_name_to_overtake)
                    target_vehicle_loc = target_vehicle_wp.transform.location
                    target_vehicle_yaw = target_vehicle_wp.transform.rotation[2]
                    target_vehicle_lane_id = target_vehicle_wp.lane_id
                    target_vehicle_exit_lane_id = Scene._agent_dict[self.vehicle_name_to_overtake].get_next_lane_id()
                    vehicle_list = [{'name': self.vehicle_name_to_overtake, 'location': target_vehicle_loc, 'yaw': target_vehicle_yaw, 'lane_id': target_vehicle_lane_id, 'exit_lane_id': target_vehicle_exit_lane_id}]
                    overtake_vehicle_state, _, _ = self._bh_is_vehicle_hazard(self.cur_waypoint, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180,low_angle_th=90)
                    if not overtake_vehicle_state:
                        if self.overtake_direction == 'left':
                            driving_mode = 'CHANGELANERIGHT'
                        else:
                            driving_mode = 'CHANGELANELEFT'
                        self._force_turn_back()
                    control = self._local_planner.run_step(
                        target_speed=self.cur_speed_limit, debug=debug)
                    self.cur_control = control
                    return control

        ego_vehicle_wp = Scene.get_vehicle_waypoint(self.vehicle_name)

        vehicle_state, vehicle_name, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        straight_vehicle_state, straight_vehicle_name, straight_distance = self.straight_collision_and_car_avoid_manager(ego_vehicle_wp)
        ahead_vehicle_state, ahead_vehicle_name, ahead_distance = self.ahead_collision_and_car_avoid_manager(ego_vehicle_wp)

        # Record interaction relationships for analysis
        self._record_interactions(vehicle_name, straight_vehicle_name, ahead_vehicle_name)

        # Handle static obstacle avoidance
        if ahead_vehicle_state and 'static' in ahead_vehicle_name:
            return self._handle_static_obstacle()
        elif straight_vehicle_state and 'static' in straight_vehicle_name:
            return self._handle_static_obstacle()
        elif vehicle_state and 'static' in vehicle_name:
            return self._handle_static_obstacle()

        # Handle dynamic vehicle interactions
        return self._handle_vehicle_interactions(vehicle_name, distance, 
                                               straight_vehicle_name, straight_distance,
                                               ahead_vehicle_name, ahead_distance)

    def _record_interactions(self, vehicle_name, straight_vehicle_name, ahead_vehicle_name):
        """Record interaction relationships for traffic analysis."""
        for name in [vehicle_name, straight_vehicle_name, ahead_vehicle_name]:
            if name is not None:
                relation_list = [self.vehicle_name, name]
                Scene._interaction_relation.append(relation_list)

    def _handle_static_obstacle(self):
        """Handle static obstacle avoidance by rerouting."""
        cur_plan = list(self._local_planner._waypoint_buffer) + list(self._local_planner.waypoints_queue)
        
        if 'ego' in self.vehicle_name:
            new_path = Scene.refine_route_w_static(cur_plan, self.bounding_box, if_ego=True)
        else:
            new_path = Scene.refine_route_w_static(cur_plan, self.bounding_box)
            
        self.reroute_all(new_path)
        control = self._local_planner.run_step(
            target_speed=self.cur_speed_limit, debug=False)
        self.cur_control = control
        return control

    def _handle_vehicle_interactions(self, vehicle_name, distance, straight_name, straight_distance, ahead_name, ahead_distance):
        """Handle dynamic vehicle interactions based on proximity and behavior."""
        # Handle ahead vehicle interactions
        if ahead_name and 'static' not in ahead_name:
            return self._handle_ahead_vehicle(ahead_name, ahead_distance)
            
        # Handle straight vehicle interactions  
        if straight_name and 'static' not in straight_name:
            return self._handle_straight_vehicle(straight_name, straight_distance)
            
        # Handle general vehicle interactions
        if vehicle_name and 'static' not in vehicle_name:
            return self._handle_general_vehicle(vehicle_name, distance)
            
        # No hazards - normal driving
        control = self._local_planner.run_step(
            target_speed=self.cur_speed_limit, debug=False)
        self.cur_control = control
        return control

    def _handle_ahead_vehicle(self, vehicle_name, distance):
        """Handle interactions with vehicles ahead."""
        target_vehicle_bbox = Scene.get_vehicle_bbox(vehicle_name)
        adjusted_distance = distance - max(target_vehicle_bbox[1]/2, target_vehicle_bbox[0]/2) - max(
            self.bounding_box[1]/2, self.bounding_box[0]/2)
            
        if adjusted_distance < self.behavior.braking_distance:
            return self._emergency_brake()
        else:
            control = self.car_following_manager(vehicle_name, adjusted_distance)
            self.cur_control = control
            return control

    def _handle_straight_vehicle(self, vehicle_name, distance):
        """Handle interactions with vehicles straight ahead."""
        target_vehicle_bbox = Scene.get_vehicle_bbox(vehicle_name)
        adjusted_distance = distance - max(target_vehicle_bbox[1]/2, target_vehicle_bbox[0]/2) - max(
            self.bounding_box[1]/2, self.bounding_box[0]/2)
            
        if adjusted_distance < self.behavior.braking_distance:
            return self._emergency_brake()
        else:
            control = self.car_following_manager(vehicle_name, adjusted_distance)
            self.cur_control = control
            return control

    def _handle_general_vehicle(self, vehicle_name, distance):
        """Handle general vehicle interactions."""
        target_vehicle_bbox = Scene.get_vehicle_bbox(vehicle_name)
        adjusted_distance = distance - max(target_vehicle_bbox[1]/2, target_vehicle_bbox[0]/2) - max(
            self.bounding_box[1]/2, self.bounding_box[0]/2)
            
        if adjusted_distance < self.behavior.braking_distance:
            return self._emergency_brake()
        else:
            if self.if_overtake:
                control = self._local_planner.run_step(
                    target_speed=self.cur_speed_limit, debug=False)
            else:
                control = self.car_following_manager(vehicle_name, adjusted_distance)
            self.cur_control = control
            return control

    def _emergency_brake(self):
        """Generate emergency braking control."""
        control = self._local_planner.run_step(target_speed=0, debug=False)
        control.throttle = 0.0
        control.brake = max(0.6, control.brake)
        self.cur_control = control
        return control