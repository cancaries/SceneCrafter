"""
Scene Controller - Main Scene Management Module

This module provides the core Scene class for managing all entities, vehicles, and simulation
state in the SceneCrafter autonomous driving simulation framework. It serves as the central
orchestration layer for autonomous driving scenarios, handling all aspects of scene management
from initialization to data collection.

MIT License

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

from __future__ import print_function
import os
import gc
import copy
import json
import random
import time
import math
import datetime
import numpy as np
from scipy.sparse import csr_matrix
from SceneController.agents.navigation.tools.misc import is_within_distance,calculate_distance,\
                                        get_bbox_corners,calculate_movement_vector,\
                                        calculate_angle_between_vectors,is_collision,calculate_relative_vector,\
                                        detect_route_interaction,build_transform_path_from_ego_pose_data,\
                                        calculate_angel_from_vector1_to_vector2

def KinematicModel(x, y, yaw, v, a, delta, f_len, r_len, dt):
    """
    Kinematic bicycle model for vehicle motion simulation.
    
    Implements a simplified vehicle dynamics model using bicycle approximation.
    Calculates next state based on current state and control inputs.
    
    Args:
        x, y: Current position coordinates (m)
        yaw: Current heading angle (rad)
        v: Current velocity (m/s)
        a: Acceleration input (m/s²)
        delta: Steering angle (rad)
        f_len: Front wheelbase distance from center (m)
        r_len: Rear wheelbase distance from center (m)  
        dt: Time step duration (s)
        
    Returns:
        tuple: (x, y, yaw, v, omega) - updated position, heading, velocity, and angular velocity
    """
    beta = math.atan((r_len / (r_len + f_len)) * math.tan(delta))
    x = x + v * math.cos(yaw + beta) * dt
    y = y + v * math.sin(yaw + beta) * dt
    yaw = yaw + (v / f_len) * math.sin(beta) * dt
    v = v + a * dt
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    v = max(0, v)
    omega = (v / f_len) * math.sin(beta)
    return x, y, yaw, v, omega

def EgoKinematicModel(x, y, yaw, v, a, delta, f_len, r_len, dt):
    """
    Enhanced kinematic model specifically for ego vehicle with smoother dynamics.
    
    Uses weighted velocity blending and reduced steering influence for more realistic
    ego vehicle behavior compared to standard kinematic model.
    
    Args:
        x, y: Current position coordinates (m)
        yaw: Current heading angle (rad)
        v: Current velocity (m/s)
        a: Acceleration input (m/s²)
        delta: Steering angle (rad)
        f_len: Front wheelbase distance from center (m)
        r_len: Rear wheelbase distance from center (m)
        dt: Time step duration (s)
        
    Returns:
        tuple: (x, y, yaw, v, omega) - updated position, heading, velocity, and angular velocity
    """
    u1, u2 = 0.47, 0.35  # Velocity blending and steering reduction coefficients
    beta = math.atan((r_len / (r_len + f_len)) * math.tan(delta))

    v_next = v + a * dt
    v_update = (1-u1)*v + u1*v_next

    x = x + v_update * math.cos(yaw + u2*beta) * dt
    y = y + v_update * math.sin(yaw + u2*beta) * dt
    yaw = yaw + (v_update / f_len) * math.sin(beta) * dt
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    omega = (v_next / f_len) * math.sin(beta)
    return x, y, yaw, v_next, omega


def compare_grids(grid1, grid2):
    # Convert occupancy grids to sparse matrices
    sparse_grid1 = csr_matrix(grid1)
    sparse_grid2 = csr_matrix(grid2)
    
    # Calculate overlapping area
    overlap = sparse_grid1.multiply(sparse_grid2).sum()
    return overlap


class Scene(object):
    """
    Scene class for managing all entities in the scene
    
    The Scene class serves as the central management system for the autonomous driving simulation.
    It maintains all vehicles (agents), handles collision detection, manages traffic flow generation,
    and coordinates the simulation step execution. This is a static class that provides global
    access to scene state and operations.
    
    Key Responsibilities:
    - Scene initialization and configuration management
    - Agent (vehicle) lifecycle management (spawn, update, remove)
    - Collision detection and resolution between vehicles
    - Traffic flow generation with configurable patterns
    - Simulation state updates and data collection
    - Integration with map system for navigation
    
    Usage:
        Scene.initialize_scene(config)  # Initialize with configuration
        Scene.spawn_agents(spawn_config)  # Generate traffic
        Scene.run_step_w_ego(time_step, ego_control=True)  # Execute simulation step
    """
    _data_root = None
    _save_dir = None
    _available_asset_dir = None
    _use_asset = False
    _scene_name = None
    _map = None
    _ego_vehicle = None
    _ego_vehicle_ori_end_point = None
    _ego_vehicle_control = None
    _static_agent_loc_dict = dict()
    _agent_dict = dict()
    _agent_control_dict = dict()
    _agent_del_dict = dict()
    _all_actor_dict = dict()
    _collision_dict = dict()
    _FPS = 10
    _start_time = None
    _car_dict_sequence = None
    _mode = None
    _ego_time_step = 0
    _skip_ego = False
    _behaviour_type_list = ['cautious', 'normal', 'aggressive', 'extreme_aggressive']
    _vehicle_type_list = ['car', 'SUV', 'truck', 'bus']
    _vehcile_type_proportion = [0.8,0.2,0.0,0.0]
    _vehicle_type_fr_len_dict = {'car':1.3,'SUV':1.5,'truck':2.5,'bus':3.0}
    _vehicle_type_max_control_angle = {'car':40.0,'SUV':40.0,'truck':40.0,'bus':40.0}
    _vehicle_bbox_dict = {'car':(1.5,3.0,1.5),'SUV':(1.8,4.0,1.8),'truck':(2.0,6.0,2.0),'bus':(2.5,12.0,2.5)}
    _behaviour_type_proportion = [0.2, 0.4, 0.35, 0.05]
    _end_scene = False
    _time_out = False
    _end_scene_ori_end_point = False
    _cnt_before_end = 0
    _exit_ori_end_point = False
    _ego_slow_speed_flag = 0
    _speed_type = 'fast'
    _end2end_sim = False
    _interaction_relation = []
    _interaction_relation_last = []
    _only_ego_move = False
    _ego_ori_route = []
    _max_speed = 0.0
    _agent_tmp_state = {}
    _scene_path = None
    _scene_info = None

    @staticmethod
    def initialize_scene(config):
        """
        Initialize the scene with comprehensive configuration
        """
        if 'end2end_sim' in config.keys():
            Scene._end2end_sim = config['end2end_sim']
        Scene._data_root = config['_data_root']
        Scene._save_dir = config['_save_dir']
        os.makedirs(Scene._save_dir, exist_ok=True)
        if '_available_asset_dir' in config.keys():
            Scene._available_asset_dir = config['_available_asset_dir']
        os.makedirs(Scene._save_dir, exist_ok=True)
        Scene._scene_name = config['_scene_name']
        if '_FPS' in config.keys():
            Scene._FPS = config['_FPS']
        # print(config['_map_config_path'])
        Scene.create_map(config['_map_config_path'])
        Scene.initilize_static_agents(config['_static_actor_config_path'])
        Scene.initialize_agent_parameter(config['other_agents_config'])
        Scene.initialize_ego_vehicle(config)
        if config['mode'] == 'debug':
            Scene.set_mode('debug')
        elif config['mode'] == 'datagen':
            Scene.set_mode('datagen')
        Scene.spawn_agents(config['agent_spawn_config'])
        Scene._start_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        Scene.check_agents()
        print('Scene initialized!')
    
    @staticmethod
    def initialize_example_scene(config):
        """
        initialize example scene

        Args:
            config (dict): Scene configuration
        """
        if 'end2end_sim' in config.keys():
            Scene._end2end_sim = config['end2end_sim']
        Scene._data_root = config['_data_root']
        Scene._save_dir = config['_save_dir']
        os.makedirs(Scene._save_dir, exist_ok=True)
        if '_available_asset_dir' in config.keys():
            Scene._available_asset_dir = config['_available_asset_dir']
        os.makedirs(Scene._save_dir, exist_ok=True)
        Scene._scene_name = config['_scene_name']
        if '_FPS' in config.keys():
            Scene._FPS = config['_FPS']
        # print(config['_map_config_path'])
        Scene.create_map(config['_map_config_path'])
        Scene._scene_path = config['_scene_path']
        with open(Scene._scene_path, 'r') as file:
            Scene._scene_info = json.load(file)
        Scene.initilize_static_agents(config['_static_actor_config_path'])
        Scene.initialize_agent_parameter(config['other_agents_config'])
        Scene.initialize_ego_vehicle_w_json(config)
        if config['mode'] == 'debug':
            Scene.set_mode('debug')
        elif config['mode'] == 'datagen':
            Scene.set_mode('datagen')
        Scene.spawn_agents(config['agent_spawn_config'])
        Scene._start_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        Scene.check_agents()
        print('Scene initialized!')
    

    @staticmethod
    def reset():
        """
        Reset the scene to initial state
        """
        Scene._static_agent_loc_dict.clear()
        Scene._agent_dict.clear()
        Scene._agent_tmp_state.clear()
        Scene._agent_control_dict.clear()
        Scene._agent_del_dict.clear()
        Scene._all_actor_dict.clear()
        Scene._collision_dict.clear()

        Scene._data_root = None
        Scene._save_dir = None
        Scene._available_asset_dir = None
        Scene._use_asset = False
        Scene._scene_name = None
        Scene._map = None
        Scene._ego_vehicle = None
        Scene._ego_vehicle_ori_end_point = None
        Scene._ego_vehicle_control = None
        Scene._static_agent_loc_dict = dict()
        Scene._agent_dict = dict()
        Scene._agent_control_dict = dict()
        Scene._agent_del_dict = dict()
        Scene._all_actor_dict = dict()
        Scene._FPS = 10
        Scene._start_time = None
        Scene._car_dict_sequence = None
        Scene._mode = None
        Scene._ego_time_step = 0
        Scene._skip_ego = False
        Scene._behaviour_type_list = ['cautious', 'normal', 'aggressive', 'extreme_aggressive']
        Scene._vehicle_type_list = ['car', 'SUV', 'truck', 'bus']
        Scene._vehcile_type_proportion = [0.8,0.2,0.0,0.0]
        Scene._vehicle_type_fr_len_dict = {'car':1.3,'SUV':1.5,'truck':2.5,'bus':3.0}
        Scene._vehicle_type_max_control_angle = {'car':40.0,'SUV':40.0,'truck':40.0,'bus':40.0}
        Scene._vehicle_bbox_dict = {'car':(1.5,3.0,1.5),'SUV':(1.8,4.0,1.8),'truck':(2.0,6.0,2.0),'bus':(2.5,12.0,2.5)}
        Scene._behaviour_type_proportion = [0.2, 0.4, 0.35, 0.05]
        Scene._end_scene = False
        Scene._time_out = False
        Scene._end_scene_ori_end_point = False
        Scene._cnt_before_end = 0
        Scene._exit_ori_end_point = False
        Scene._ego_slow_speed_flag = 0
        Scene._speed_type = 'fast'
        Scene._end2end_sim = False
        Scene._collision_dict = dict()
        Scene._interaction_relation = []
        Scene._interaction_relation_last = []
        Scene._only_ego_move = False
        Scene._ego_ori_route = []
        Scene._max_speed = 0.0
        Scene._agent_tmp_state = dict()
        Scene._scene_path = None
        Scene._scene_info = None

        gc.collect()

    @staticmethod
    def create_map(map_config_path):
        map_config_path = map_config_path
        from SceneController.agents.map import Map
        # Create map
        Scene._map = Map(map_config_path)

    @staticmethod
    def search_ego_max_speed(scene_name, base_data_path='/GPFS/public/junhaoge/data/SceneCrafter/waymo_train_data'):
        scene_data_folder = os.path.join(base_data_path, scene_name)
        speed_list = []
        for data_folder in os.listdir(scene_data_folder):
            if not data_folder.isdigit():
                continue
            with open(os.path.join(scene_data_folder, data_folder, 'ego_pose.json'), 'r') as file:
                ego_pose_data = json.load(file)
            speed_list.append(ego_pose_data['speed'])
        return max(speed_list) * 3.6*0.8

    @staticmethod
    def initialize_agent_parameter(agent_parameter):
        """
        Initialize agent parameters

        Args:
            agent_parameter (dict): Agent parameters
        """
        Scene._max_speed = Scene.search_ego_max_speed(Scene._scene_name)
        Scene._speed_type = agent_parameter['speed_type']
        if Scene._available_asset_dir is not None:
            Scene._use_asset = True
            with open(Scene._available_asset_dir, 'r') as file:
                vehicle_gaussian_info = json.load(file)
            Scene._vehicle_type_list = list(vehicle_gaussian_info.keys())
            if 'exclude_vehicle_type' in agent_parameter.keys():
                for vehicle_type in agent_parameter['exclude_vehicle_type']:
                    if vehicle_type in Scene._vehicle_type_list:
                        Scene._vehicle_type_list.remove(vehicle_type)
            print('Vehicle type list:', Scene._vehicle_type_list)
            tmp_vehicle_bbox_dict = {}
            tmp_vehcile_type_proportion = []
            tmp_vehicle_type_fr_len_dict = {}
            tmp_vehicle_type_max_control_angle = {}
            for vehicle_type in Scene._vehicle_type_list:
                tmp_vehicle_bbox_dict[vehicle_type] = vehicle_gaussian_info[vehicle_type]['size']
                tmp_vehicle_type_fr_len_dict[vehicle_type] = {}
                tmp_vehicle_type_fr_len_dict[vehicle_type]['f_len'] = (vehicle_gaussian_info[vehicle_type]['size'][1])/5
                tmp_vehicle_type_fr_len_dict[vehicle_type]['r_len'] = (vehicle_gaussian_info[vehicle_type]['size'][1])/3.5
                tmp_vehicle_type_max_control_angle[vehicle_type] = 40.0
            Scene._vehicle_bbox_dict = tmp_vehicle_bbox_dict
            Scene._vehicle_type_fr_len_dict = tmp_vehicle_type_fr_len_dict
            Scene._vehicle_type_max_control_angle = tmp_vehicle_type_max_control_angle

        else:
            Scene._use_asset = False
            Scene._vehicle_type_list = agent_parameter['vehicle_type_list']
            Scene._vehcile_type_proportion = agent_parameter['vehicle_type_proportion']
            Scene._vehicle_type_fr_len_dict = agent_parameter['vehicle_type_fr_len_dict']
            Scene._vehicle_type_max_control_angle = agent_parameter['vehicle_type_max_control_angle']
        Scene._behaviour_type_proportion = agent_parameter['behaviour_type_proportion']

    @staticmethod
    def initialize_ego_vehicle_w_json(config,behavior_type='normal'):
        from agents.navigation.behavior_agent import BehaviorAgent
        ego_vehicle_config = config['ego_vehicle_config']
        
        agent_config = dict()
        agent_config['name'] = 'ego_vehicle'
        agent_config['vehicle_type'] = 'car'
        agent_config['vehicle_bbox'] = ego_vehicle_config['vehicle_bbox']
        agent_config['f_len'] = ego_vehicle_config['vehicle_fr_length'][0]
        agent_config['r_len'] = ego_vehicle_config['vehicle_fr_length'][1]
        agent_config['control_angel'] = ego_vehicle_config['vehicle_max_control_angle']
        ego_start_point = Scene._scene_info['ego_vehicle']['start_loc']
        ego_end_point = Scene._scene_info['ego_vehicle']['target_loc']
        
        waypoint_path = Scene._map.plan_waypoint_path(ego_start_point,ego_end_point)
        Scene._ego_vehicle_ori_end_point = waypoint_path[-1]
        extended_path = Scene._map.extend_plan_waypoints(waypoint_path)
        Scene._map.redefine_ego_road_option(extended_path)
        waypoint_path_refine = Scene._map.refine_plan_waypoints(extended_path,3.0)
        agent_config['initial_path'] = waypoint_path_refine
        agent_config['behavior'] = ego_vehicle_config['behavior_type']
        agent_config['speed_type'] = Scene._speed_type
        agent_config['stop_idx'] = Scene._scene_info['ego_vehicle']['stop_idx']
        agent_config['resume_idx'] = Scene._scene_info['ego_vehicle']['resume_idx']
        agent_config['max_speed'] = Scene._max_speed
        initial_speed = Scene._scene_info['ego_vehicle']['initial_speed']
        
        Scene._ego_vehicle = BehaviorAgent(agent_config)
        Scene._agent_dict['ego_vehicle'] = BehaviorAgent(agent_config)
        Scene._agent_dict['ego_vehicle'].speed = Scene._agent_dict['ego_vehicle'].speed_limit * float(initial_speed)
        return True

    @staticmethod
    def initialize_ego_vehicle(config,behavior_type='normal'):
        """
        Initialize ego vehicle

        Args:
            original_path (list): Starting path

        Returns:
            bool: Whether initialization was successful
        """
        
        if Scene._end2end_sim:
            from agents.navigation.behavior_agent_ego import BehaviorAgent
            with open(config['_ego_pose_path'], 'r') as file:
                ego_pose_data = json.load(file)
            ego_vehicle_config = config['ego_vehicle_config']
            transform_path = build_transform_path_from_ego_pose_data(ego_pose_data)
            waypoint_path = Scene._map.generate_waypoint_path_from_transform_path(transform_path)
            Scene._ego_ori_route = waypoint_path
            agent_config = dict()
            agent_config['name'] = 'ego_vehicle'
            agent_config['vehicle_type'] = 'car'
            agent_config['vehicle_bbox'] = ego_vehicle_config['vehicle_bbox']
            agent_config['f_len'] = ego_vehicle_config['vehicle_fr_length'][0]
            agent_config['r_len'] = ego_vehicle_config['vehicle_fr_length'][1]
            agent_config['control_angel'] = ego_vehicle_config['vehicle_max_control_angle']
            Scene._ego_vehicle_ori_end_point = waypoint_path[-1]
            Scene._map.redefine_ego_road_option(waypoint_path)
            agent_config['initial_path'] = waypoint_path
            agent_config['behavior'] = ego_vehicle_config['behavior_type']
            agent_config['speed_type'] = Scene._speed_type
            agent_config['max_speed'] = Scene._max_speed
            Scene._ego_vehicle = BehaviorAgent(agent_config)
            Scene._agent_dict['ego_vehicle'] = BehaviorAgent(agent_config)
            Scene._map.refine_spawn_points(ego_init_point=transform_path[0].location)
            return True
        else:
            from agents.navigation.behavior_agent import BehaviorAgent
            with open(config['_ego_pose_path'], 'r') as file:
                ego_pose_data = json.load(file)
            ego_vehicle_config = config['ego_vehicle_config']
            transform_path = build_transform_path_from_ego_pose_data(ego_pose_data)
            waypoint_path = Scene._map.generate_waypoint_path_from_transform_path(transform_path)
            Scene._ego_ori_route = waypoint_path
            agent_config = dict()
            agent_config['name'] = 'ego_vehicle'
            agent_config['vehicle_type'] = 'car'
            agent_config['vehicle_bbox'] = ego_vehicle_config['vehicle_bbox']
            agent_config['f_len'] = ego_vehicle_config['vehicle_fr_length'][0]
            agent_config['r_len'] = ego_vehicle_config['vehicle_fr_length'][1]
            agent_config['control_angel'] = ego_vehicle_config['vehicle_max_control_angle']
            
            Scene._ego_vehicle_ori_end_point = waypoint_path[-1]
            extended_path = Scene._map.extend_plan_waypoints(waypoint_path)
            Scene._map.redefine_ego_road_option(extended_path)
            waypoint_path_refine = Scene._map.refine_plan_waypoints(extended_path,3)
            agent_config['initial_path'] = waypoint_path_refine
            agent_config['behavior'] = ego_vehicle_config['behavior_type']
            agent_config['speed_type'] = Scene._speed_type
            agent_config['max_speed'] = Scene._max_speed
            
            Scene._ego_vehicle = BehaviorAgent(agent_config)
            Scene._agent_dict['ego_vehicle'] = BehaviorAgent(agent_config)
            Scene._agent_dict['ego_vehicle'].speed = Scene._agent_dict['ego_vehicle'].speed_limit * (0.25 * random.random())
            Scene._map.refine_spawn_points(ego_init_point=transform_path[0].location)
            return True

    @staticmethod
    def initilize_static_agents(static_agent_config_path):
        from agents.navigation.behavior_agent_static import BehaviorAgent
        with open(static_agent_config_path, 'r') as file:
            static_agent_config = json.load(file)
        for agent_idx, (agent_id, static_agent_config) in enumerate(static_agent_config.items()):
            if static_agent_config['obj_class'] != 'vehicle' and static_agent_config['obj_class'] != 'pedestrian':
                continue
            agent_config = dict()
            agent_config['name'] = 'static_agent_' + str(agent_idx)
            agent_config['vehicle_type'] = static_agent_config['obj_class']
            agent_config['vehicle_bbox'] = static_agent_config['size']
            location = static_agent_config['location']
            rotation = static_agent_config['rotation']
            waypoint_config = Scene._map.build_waypoint_config(location, rotation)
            waypoint = Scene._map.build_waypoint(waypoint_config)
            agent_config['static_waypoint'] = waypoint
            agent = BehaviorAgent(agent_config)
            Scene._agent_dict[agent_config['name']] = agent
            Scene._static_agent_loc_dict[agent_config['name']] = dict()
            Scene._static_agent_loc_dict[agent_config['name']]['location'] = waypoint.transform.location[:2]
            Scene._static_agent_loc_dict[agent_config['name']]['yaw'] = waypoint.transform.rotation[2]
            Scene._static_agent_loc_dict[agent_config['name']]['bbox'] = agent.bounding_box
        Scene.refine_spawn_points_w_static()

    @staticmethod
    def refine_spawn_points_w_static():
        """
        Remove spawn points around static vehicles
        """
        for _, agent in Scene._static_agent_loc_dict.items():
            Scene._map.refine_spawn_points_w_location(agent['location']+[0.0],distance_thre=1.0)

    @staticmethod
    def refine_route_w_static(route,self_size,distance_threshold=10,if_ego=False):
        """
        Modify random routes that conflict with static vehicles

        Args:
            route (list): Random route

        Returns:
            list: Modified random route
        """
        for idx, path_waypoint in enumerate(route):
            loc = path_waypoint.transform.get_location()[:2]
            yaw = path_waypoint.transform.get_rotation()[2]
            collision_flag = False
            for agent_name, agent in Scene._static_agent_loc_dict.items():
                offset = 0.1
                direction = None
                static_loc = agent['location']
                static_yaw = agent['yaw']
                static_bbox = agent['bbox']
                if calculate_distance(route[idx].transform.get_location()[:2],static_loc) > distance_threshold:
                    continue
                if direction is None:
                    relative_vector = calculate_relative_vector(loc, static_loc)
                    route_vector = np.array([math.cos(yaw), math.sin(yaw)])
                    angle = calculate_angel_from_vector1_to_vector2(route_vector,relative_vector)
                    angle = (angle + 2 * math.pi) % (2 * math.pi)
                    if angle < math.pi:
                        direction = 'left'
                    else:
                        direction = 'right'
                while is_collision(loc, yaw, self_size, static_loc, static_yaw, static_bbox):
                    collision_flag = True
                    offset += 0.1
                    waypoint_new = copy.deepcopy(path_waypoint)
                    waypoint_new = Scene._map.get_waypoint_w_offset(waypoint_new, offset, direction=direction)
                    route[idx] = waypoint_new
                    loc = waypoint_new.transform.location[:2]
                    
                if collision_flag:
                    offset += 0.05
                    waypoint_new = Scene._map.get_waypoint_w_offset(waypoint_new, offset, direction=direction)
                    route[idx] = waypoint_new
                    collision_flag = False
                    break
        return route

    @staticmethod
    def check_route_valid_w_ego_route(route,width_threshold=10):
        """
        Check if route is within ego route's width threshold range (10m left/right, 5m front/back)
        Returns False if within range

        Args:
            route (list): Random route

        Returns:
            bool: Whether route is valid
        """
        if Scene._ego_vehicle is not None:
            ego_route = [x.transform.location[:2] for x in Scene._ego_vehicle.plan_path]
        else:
            print('Ego vehicle is not initialized!')
            return True
        
        if route is not None:
            test_route = [x.transform.location[:2] for x in route]
        else:
            print('Route is None!')
            return True
        return detect_route_interaction(test_route,ego_route,interaction_range_1=10.0,interaction_range_2=10.0)
    
    @staticmethod
    def check_route_valid_w_static(route,close_threshold=0.75):
        """
        Check if route distance to static vehicles is less than close_threshold
        Returns False if too close

        Args:
            route (list): Random route

        Returns:
            bool: Whether route is valid
        """
        for agent_name, agent in Scene._static_agent_loc_dict.items():
            static_loc = agent['location']
            static_yaw = agent['yaw']
            static_bbox = agent['bbox']
            for path_waypoint in route:         
                loc = path_waypoint.transform.get_location()[:2]
                yaw = path_waypoint.transform.get_rotation()[2]
                if is_collision(loc, yaw, [1.0,2.5,1.0], static_loc, static_yaw, static_bbox):
                    return False
        return True
    
    @staticmethod
    def get_ego_endpoints_to_spawn():
        """
        Get ego vehicle's start and end points

        Returns:
            tuple: Start and end points
        """
        ego_trajectory = Scene._ego_ori_route[-10:]
        ego_trajectory = [x.transform.location for x in ego_trajectory]
        ego_trajectory_new = []
        # filter with distance
        for idx, loc in enumerate(ego_trajectory):
            if idx == 0:
                ego_trajectory_new.append(loc)
                continue
            if calculate_distance(ego_trajectory_new[-1],loc) > 7.0:
                ego_trajectory_new.append(loc)
        if len(ego_trajectory_new) > 2:
            ego_trajectory_new = random.sample(ego_trajectory_new,2)
        return ego_trajectory_new

    @staticmethod
    def refine_spawn_points_w_ego_route(spawn_points):
        """
        Get ego vehicle's start and end points

        Returns:
            tuple: Start and end points
        """
        ego_trajectory = Scene._ego_ori_route
        ego_trajectory = [x.transform.location for x in ego_trajectory]
        # filter with distance
        spawn_points_new = []
        for idx, spawn_point in enumerate(spawn_points):
            valid_flag = True
            for loc in ego_trajectory:
                if calculate_distance(spawn_point,loc) > 3.0:
                    pass
                else:
                    valid_flag = False
                    break
            if valid_flag:
                spawn_points_new.append(spawn_point)
        return spawn_points_new

    @staticmethod
    def generate_background_agents(spawn_num=10, 
                                   random_agents=False,
                                   close_to_ego=False,
                                   close_threshold=30,
                                   too_close_threshold=10,
                                   ego_forward_clear=False,
                                   forward_threshold=10,
                                   same_lane=False):
        """
        Generate background agents with random routes and add to scene

        Args:
            random (bool, optional): Whether to use random agents. Defaults to True.
            spawn_num (int, optional): Number of agents to spawn. Defaults to 10.
        """
        from agents.navigation.behavior_agent import BehaviorAgent
        spawn_points = Scene._map.get_spawn_points()
        spawn_points_num = len(spawn_points)
        vehicle_type_list = Scene._vehicle_type_list
        weights = Scene._vehcile_type_proportion
        vehicle_bbox_dict = Scene._vehicle_bbox_dict
        vehicle_fr_len_dict = Scene._vehicle_type_fr_len_dict
        vehicle_control_angle_dict = Scene._vehicle_type_max_control_angle
        behavior_type_list = Scene._behaviour_type_list
        behavior_type_proportion = Scene._behaviour_type_proportion
        spawned_num = 0
        spawned_points = []
        if random_agents:
            # Randomly select half of the spawn points for deployment
            random.shuffle(spawn_points)
            # spawn_points_filter = random.sample(spawn_points, int(spawn_points_num/2))
            for idx, spawn_point in enumerate(spawn_points):
                # Skip if spawn_point is less than 10m from points in spawned_points
                if len(spawned_points) > 0:
                    for point in spawned_points:
                        if calculate_distance(spawn_point,point) < 5.0:
                            continue
                if spawned_num >= spawn_num:
                    break
                if ego_forward_clear:
                    if Scene._ego_vehicle is not None:
                        ego_loc = Scene._ego_vehicle.cur_waypoint.transform.location
                        ego_yaw = Scene._ego_vehicle.cur_waypoint.transform.rotation[2]
                        ego_bbox = Scene._ego_vehicle.bounding_box
                        if is_within_distance(spawn_point, ego_loc, ego_yaw,
                              forward_threshold, 45, 0):
                            continue
                
                agent_config = dict()
                agent_config['name'] = 'background_agent_' + str(idx)
                if Scene._use_asset:
                    vehicle_type = random.choices(vehicle_type_list)[0]
                else:
                    vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
                agent_config['vehicle_type'] = vehicle_type
                agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
                random_path = Scene._map.generate_overall_plan_waypoints_w_refine(spawn_point)
                if not Scene.check_route_valid_w_static(random_path):
                    continue
                random_path = Scene.refine_route_w_static(random_path,agent_config['vehicle_bbox'])
                if not Scene.check_route_valid_w_ego_route(random_path):
                    continue
                agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]['f_len']
                agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]['r_len']
                agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
                agent_config['initial_path'] = random_path
                agent_config['behavior'] = random.choices(behavior_type_list,weights=behavior_type_proportion)[0]
                agent_config['speed_type'] = Scene._speed_type
                agent_config['max_speed'] = Scene._max_speed
                agent = BehaviorAgent(agent_config)
                Scene._agent_dict[agent_config['name']] = agent
                Scene._agent_dict[agent_config['name']].speed = Scene._agent_dict[agent_config['name']].speed_limit * (0.25 * random.random())
                spawned_num += 1
                spawned_points.append(spawn_point)
                
        elif close_to_ego and Scene._ego_vehicle is not None:
            ego_loc = Scene._ego_vehicle.cur_waypoint.transform.location
            ego_yaw = Scene._ego_vehicle.cur_waypoint.transform.rotation[2]
            ego_bbox = Scene._ego_vehicle.bounding_box
            random.shuffle(spawn_points)
            for idx, spawn_point in enumerate(spawn_points):
                if same_lane:
                    if Scene._ego_vehicle is not None:
                        ego_lane_id = Scene._ego_vehicle.cur_waypoint.lane_id
                        spawn_lane_id, _, _ = Scene._map.find_nearest_lane_point(spawn_point)
                        if ego_lane_id != spawn_lane_id:
                            continue
                spawn_loc = spawn_point
                if ego_forward_clear:
                    if Scene._ego_vehicle is not None:
                        ego_loc = Scene._ego_vehicle.cur_waypoint.transform.location
                        ego_yaw = Scene._ego_vehicle.cur_waypoint.transform.rotation[2]
                        ego_bbox = Scene._ego_vehicle.bounding_box
                        if is_within_distance(spawn_point, ego_loc, ego_yaw,
                              forward_threshold, 45, 0):
                            continue
                spawn_bbox = vehicle_bbox_dict['car']
                if calculate_distance(spawn_loc, ego_loc) < close_threshold and \
                    too_close_threshold < calculate_distance(spawn_loc, ego_loc):
                    agent_config = dict()
                    agent_config['name'] = 'background_agent_' + str(spawned_num)
                    if Scene._use_asset:
                        vehicle_type = random.choices(vehicle_type_list)[0]
                    else:
                        vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
                    agent_config['vehicle_type'] = vehicle_type
                    agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
                    random_path = Scene._map.generate_overall_plan_waypoints_w_refine(spawn_point)
                    if not Scene.check_route_valid_w_static(random_path):
                        continue
                    random_path = Scene.refine_route_w_static(random_path,agent_config['vehicle_bbox'])
                    if not Scene.check_route_valid_w_ego_route(random_path):
                        continue
                    agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]
                    agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]
                    agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
                    agent_config['initial_path'] = random_path
                    agent_config['behavior'] = random.choices(behavior_type_list,weights=behavior_type_proportion)[0]
                    agent_config['speed_type'] = Scene._speed_type
                    # agent = None
                    agent = BehaviorAgent(agent_config)
                    Scene._agent_dict[agent_config['name']] = agent
                    spawned_num += 1
                    if spawned_num >= spawn_num:
                        break
                
        else:
            spawn_points_filter = random.shuffle(spawn_points)
            for idx, spawn_point in enumerate(spawn_points_filter):
                if ego_forward_clear:
                    if Scene._ego_vehicle is not None:
                        ego_loc = Scene._ego_vehicle.cur_waypoint.transform.location
                        ego_yaw = Scene._ego_vehicle.cur_waypoint.transform.rotation[2]
                        ego_bbox = Scene._ego_vehicle.bounding_box
                        if is_within_distance(spawn_point, ego_loc, ego_yaw,
                              forward_threshold, 45, 0):
                            continue
                agent_config = dict()
                agent_config['name'] = 'background_agent_' + str(spawned_num)
                if Scene._use_asset:
                    vehicle_type = random.choices(vehicle_type_list)[0]
                else:
                    vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
                agent_config['vehicle_type'] = vehicle_type
                agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
                random_path = Scene._map.generate_overall_plan_waypoints_w_refine(spawn_point)
                if not Scene.check_route_valid_w_static(random_path):
                    continue
                random_path = Scene.refine_route_w_static(random_path,agent_config['vehicle_bbox'])
                if not Scene.check_route_valid_w_ego_route(random_path):
                    continue
                agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]
                agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]
                agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
                agent_config['initial_path'] = random_path
                agent_config['behavior'] = random.choices(behavior_type_list,weights=behavior_type_proportion)[0]
                agent_config['speed_type'] = Scene._speed_type
                agent = BehaviorAgent(agent_config)
                Scene._agent_dict[agent_config['name']] = agent
                spawned_num += 1
                if spawned_num >= spawn_num:
                    break

    @staticmethod
    def generate_static_background_agents(spawn_num=10, away_from_ego=False):
        """
        generate static background agents

        Args:
            random (bool, optional): _description_. Defaults to True.
            spawn_num (int, optional): _description_. Defaults to 10.
        """
        from agents.navigation.behavior_agent import BehaviorAgent
        
        if away_from_ego and Scene._ego_vehicle is not None:
            ego_plan_path = Scene._ego_vehicle.plan_path
            spawn_points = Scene._map.generate_spawn_points_w_ego_path(ego_plan_path)
        else:
            spawn_points = Scene._map.get_spawn_points()
        spawn_points_num = len(spawn_points)
        vehicle_type_list = Scene._vehicle_type_list
        weights = Scene._vehcile_type_proportion
        vehicle_bbox_dict = Scene._vehicle_bbox_dict
        vehicle_fr_len_dict = Scene._vehicle_type_fr_len_dict
        vehicle_control_angle_dict = Scene._vehicle_type_max_control_angle
        behavior_type_list = Scene._behaviour_type_list
        behavior_type_proportion = Scene._behaviour_type_proportion
        spawned_num = 0
        random.shuffle(spawn_points)
        for idx, spawn_point in enumerate(spawn_points):
            if spawned_num >= spawn_num:
                break
            agent_config = dict()
            agent_config['name'] = 'background_agent_' + str(idx)
            if Scene._use_asset:
                vehicle_type = random.choices(vehicle_type_list)[0]
            else:
                vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
            agent_config['vehicle_type'] = vehicle_type
            agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
            random_path = Scene._map.generate_overall_plan_waypoints_w_refine(spawn_point)
            if not Scene.check_route_valid_w_static(random_path):
                continue
            random_path = Scene.refine_route_w_static(random_path,agent_config['vehicle_bbox'])
            if not Scene.check_route_valid_w_ego_route(random_path):
                continue
            agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]['f_len']
            agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]['r_len']
            agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
            agent_config['initial_path'] = random_path
            agent_config['behavior'] = random.choices(behavior_type_list,weights=behavior_type_proportion)[0]
            agent_config['speed_type'] = Scene._speed_type
            agent = BehaviorAgent(agent_config)
            Scene._agent_dict[agent_config['name']] = agent
            Scene._agent_dict[agent_config['name']].speed = 0.0
            spawned_num += 1

    @staticmethod
    def generate_json_agents():
        """
        generate agents defined in json and add to scene

        Args:
            random (bool, optional): _description_. Defaults to True.
            spawn_num (int, optional): _description_. Defaults to 10.
        """
        from agents.navigation.behavior_agent import BehaviorAgent
        vehicle_type_list = Scene._vehicle_type_list
        weights = Scene._vehcile_type_proportion
        vehicle_bbox_dict = Scene._vehicle_bbox_dict
        vehicle_fr_len_dict = Scene._vehicle_type_fr_len_dict
        vehicle_control_angle_dict = Scene._vehicle_type_max_control_angle
        behavior_type_list = Scene._behaviour_type_list
        behavior_type_proportion = Scene._behaviour_type_proportion
        for agent_name, agent_setting in Scene._scene_info.items():
            if 'ego_vehicle' in agent_name:
                continue
            agent_config = dict()
            print(agent_setting)
            agent_config['name'] = agent_name
            vehicle_type = agent_setting['type']
            agent_config['vehicle_type'] = vehicle_type
            agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
            agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]['f_len']
            agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]['r_len']
            agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
            start_point = agent_setting['start_loc']
            end_point = agent_setting['target_loc']
            
            waypoint_path = Scene._map.plan_waypoint_path(start_point,end_point)
            extended_path = Scene._map.extend_plan_waypoints(waypoint_path)
            Scene._map.redefine_ego_road_option(extended_path)
            waypoint_path_refine = Scene._map.refine_plan_waypoints(extended_path,3.0)
            agent_config['initial_path'] = waypoint_path_refine
            agent_config['speed_type'] = Scene._speed_type
            agent_config['behavior'] = random.choices(behavior_type_list,weights=behavior_type_proportion)[0]
            agent_config['stop_idx'] = Scene._scene_info['ego_vehicle']['stop_idx']
            agent_config['resume_idx'] = Scene._scene_info['ego_vehicle']['resume_idx']
            agent_config['max_speed'] = Scene._max_speed
            initial_speed = agent_setting['initial_speed']
            agent = BehaviorAgent(agent_config)
            Scene._agent_dict[agent_name] = agent
            Scene._agent_dict[agent_name].speed = Scene._agent_dict['ego_vehicle'].speed_limit * 0.5
            

    @staticmethod
    def generate_occlusion_background_agents(spawn_num=10):
        """
        generate agents defined in json and add to scene

        Args:
            random (bool, optional): _description_. Defaults to True.
            spawn_num (int, optional): _description_. Defaults to 10.
        """
        from agents.navigation.behavior_agent import BehaviorAgent
        spawn_points = Scene.get_ego_endpoints_to_spawn()
        spawn_points_num = len(spawn_points)
        vehicle_type_list = Scene._vehicle_type_list
        weights = Scene._vehcile_type_proportion
        vehicle_bbox_dict = Scene._vehicle_bbox_dict
        vehicle_fr_len_dict = Scene._vehicle_type_fr_len_dict
        vehicle_control_angle_dict = Scene._vehicle_type_max_control_angle
        behavior_type_list = Scene._behaviour_type_list
        behavior_type_proportion = Scene._behaviour_type_proportion
        spawned_num = 0
        random.shuffle(spawn_points)
        for idx, spawn_point in enumerate(spawn_points):
            if spawned_num >= spawn_num:
                break
            agent_config = dict()
            agent_config['name'] = 'background_agent_' + str(spawned_num)
            if Scene._use_asset:
                vehicle_type = random.choices(vehicle_type_list)[0]
            else:
                vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
            agent_config['vehicle_type'] = vehicle_type
            agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
            random_path = Scene._map.generate_overall_plan_waypoints_w_refine(spawn_point)
            if not Scene.check_route_valid_w_static(random_path):
                continue
            random_path = Scene.refine_route_w_static(random_path,agent_config['vehicle_bbox'])
            if not Scene.check_route_valid_w_ego_route(random_path):
                continue
            agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]['f_len']
            agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]['r_len']
            agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
            agent_config['initial_path'] = random_path
            agent_config['behavior'] = random.choices(behavior_type_list,weights=behavior_type_proportion)[0]
            agent_config['speed_type'] = Scene._speed_type
            agent_config['max_speed'] = Scene._max_speed
            agent = BehaviorAgent(agent_config)
            Scene._agent_dict[agent_config['name']] = agent
            Scene._agent_dict[agent_config['name']].speed = 0.0
            Scene._agent_dict[agent_config['name']].set_static(True)
            spawned_num += 1 
        spawn_points = Scene._map.get_spawn_points()
        spawn_points_new = Scene.refine_spawn_points_w_ego_route(spawn_points)
        for idx, spawn_point in enumerate(spawn_points_new):
            if spawned_num >= spawn_num:
                break
            agent_config = dict()
            agent_config['name'] = 'background_agent_' + str(spawned_num)
            if Scene._use_asset:
                vehicle_type = random.choices(vehicle_type_list)[0]
            else:
                vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
            agent_config['vehicle_type'] = vehicle_type
            agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
            random_path = Scene._map.generate_overall_plan_waypoints_w_refine(spawn_point)
            if not Scene.check_route_valid_w_static(random_path):
                continue
            random_path = Scene.refine_route_w_static(random_path,agent_config['vehicle_bbox'])
            if not Scene.check_route_valid_w_ego_route(random_path):
                continue
            agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]['f_len']
            agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]['r_len']
            agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
            agent_config['initial_path'] = random_path
            agent_config['behavior'] = random.choices(behavior_type_list,weights=behavior_type_proportion)[0]
            agent_config['speed_type'] = Scene._speed_type
            agent = BehaviorAgent(agent_config)
            Scene._agent_dict[agent_config['name']] = agent
            Scene._agent_dict[agent_config['name']].speed = Scene._agent_dict[agent_config['name']].speed_limit * (0.25 * random.random())
            Scene._agent_dict[agent_config['name']].set_static(False)
            spawned_num += 1 
        

    @staticmethod
    def check_agents():
        """
        Check for collisions between agents and remove those that collide
        """
        for agent_name, agent in Scene._agent_dict.items():
            if 'ego' in agent_name or 'static' in agent_name:
                continue
            agent_loc = agent.cur_waypoint.transform.get_location()[:2]
            agent_yaw = agent.cur_waypoint.transform.get_rotation()[2]
            agent_bbox = agent.bounding_box
            for agent_name_other, agent_other in Scene._agent_dict.items():
                if agent_name == agent_name_other:
                    continue
                agent_loc_other = agent_other.cur_waypoint.transform.get_location()[:2]
                agent_yaw_other = agent_other.cur_waypoint.transform.get_rotation()[2]
                agent_bbox_other = agent_other.bounding_box
                if is_collision(agent_loc, agent_yaw, agent_bbox, agent_loc_other, agent_yaw_other, agent_bbox_other): 
                    Scene._agent_del_dict[agent_name] = agent
                    break
        for agent_name_del, agent_del in Scene._agent_del_dict.items():
            del Scene._agent_dict[agent_name_del]

    @staticmethod
    def spawn_agents(spawn_config):
        """
        Spawn traffic agents based on configuration parameters
        
        Generates background traffic using various spawning strategies including
        random placement, proximity-based placement, static vehicles, occlusion scenarios,
        and JSON-defined configurations.
        
        Args:
            spawn_config (dict): Configuration dictionary containing:
                - spawn_mode: Type of spawning strategy ('random', 'close_to_ego', 'static', 
                             'occlusion', 'map_test', 'json')
                - max_spawn_num: Maximum number of agents to spawn
                - close_spawn_distance: Distance threshold for close-to-ego spawning (when applicable)
        """
        spawn_mode = spawn_config['spawn_mode']
        if spawn_mode == 'random':
            Scene.generate_background_agents(random_agents=True,
                                                       spawn_num=spawn_config['max_spawn_num'])
        elif spawn_mode == 'close_to_ego':
            Scene.generate_background_agents(spawn_config['max_spawn_num'], 
                                                       close_to_ego=True,
                                                       close_threshold=spawn_config['close_spawn_distance'])
        elif spawn_mode == 'static':
            Scene._only_ego_move = True
            Scene.generate_static_background_agents(spawn_num=spawn_config['max_spawn_num'])
        elif spawn_mode == 'occlusion':
            Scene.generate_occlusion_background_agents(spawn_num=spawn_config['max_spawn_num'])
        elif spawn_mode == 'map_test':
            Scene._only_ego_move = True
            Scene.generate_static_background_agents(spawn_num=spawn_config['max_spawn_num'],away_from_ego=True)
        elif spawn_mode == 'json':
            Scene.generate_json_agents()

    @staticmethod
    def get_vehicle_speed(vehicle_name):
        vehicle_agent = Scene._agent_dict[vehicle_name]
        return vehicle_agent.get_speed()

    @staticmethod
    def get_vehicle_speed_m(vehicle_name):
        vehicle_agent = Scene._agent_dict[vehicle_name]
        return vehicle_agent.get_speed()/3.6

    @staticmethod
    def get_vehicle_bbox(vehicle_name):
        vehicle_agent = Scene._agent_dict[vehicle_name]
        return vehicle_agent.bounding_box

    @staticmethod
    def get_vehicle_location(vehicle_name):
        vehicle_agent = Scene._agent_dict[vehicle_name]
        return vehicle_agent.cur_waypoint.transform.location

    @staticmethod
    def get_vehicle_waypoint(vehicle_name):
        if vehicle_name not in Scene._agent_dict.keys():
            return None
        vehicle_agent = Scene._agent_dict[vehicle_name]
        return vehicle_agent.cur_waypoint

    @staticmethod
    def get_vehicle_control(vehicle_name):
        if vehicle_name not in Scene._agent_control_dict.keys():
            from agents.navigation.controller import Control
            control = Control()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            return control
        vehicle_agent_control = Scene._agent_control_dict[vehicle_name]
        return vehicle_agent_control

    @staticmethod
    def generate_route_w_waypoints(start_waypoint, end_waypoint):
        """
        Generate path between two waypoints

        Args:
            start_waypoint (Waypoint): Starting waypoint
            end_waypoint (Waypoint): Ending waypoint

        Returns:    
            list: Waypoints along the path
        """
        return Scene._map.plan_path_w_waypoints(start_waypoint, end_waypoint)

    @staticmethod
    def adjust_position(target_vehicle_agent, reference_vehicle_agent):
        # Get current and previous frame positions
        current_position = target_vehicle_agent.cur_waypoint.transform.location[:2]
        previous_position = target_vehicle_agent.last_waypoint.transform.location[:2]
        target_vehicle_yaw = target_vehicle_agent.last_waypoint.transform.rotation[2]
        target_vehicle_agent.cur_waypoint.transform.set_rotation([0, 0, target_vehicle_yaw])
        # Calculate vehicle movement direction unit vector
        movement_direction = calculate_relative_vector(previous_position, current_position)
        movement_distance = np.linalg.norm(movement_direction)
        
        if movement_distance > 0:
            movement_direction /= movement_distance

        # Binary search for closest non-collision position
        low, high = 0, movement_distance
        best_position = current_position
        reference_vehicle_loc = reference_vehicle_agent.cur_waypoint.transform.location[:2]
        reference_vehicle_prev_loc = reference_vehicle_agent.last_waypoint.transform.location[:2]
        reference_vehicle_yaw = reference_vehicle_agent.cur_waypoint.transform.rotation[2]
        reference_vehicle_bbox = reference_vehicle_agent.bounding_box
        if calculate_distance(current_position, previous_position) < 0.05 and calculate_distance(reference_vehicle_loc, reference_vehicle_prev_loc) < 0.05:
            return
        if_adjust = False
        while high - low > 1e-3:  # Precision control
            mid = (low + high) / 2
            test_position = previous_position + movement_direction * mid
            target_vehicle_yaw = np.arctan2(test_position[1] - previous_position[1], test_position[0] - previous_position[0])
            target_vehicle_agent.cur_waypoint.transform.set_location(test_position)
            target_vehicle_agent.cur_waypoint.transform.set_rotation([0, 0, target_vehicle_yaw])
            target_vehicle_loc = target_vehicle_agent.cur_waypoint.transform.location[:2]
            target_vehicle_bbox = target_vehicle_agent.bounding_box
            if is_collision(target_vehicle_loc, 
                            target_vehicle_yaw, 
                            target_vehicle_bbox, 
                            reference_vehicle_loc, 
                            reference_vehicle_yaw, 
                            reference_vehicle_bbox):
                high = mid  # Narrow range, closer to previous frame position
            else:
                if_adjust = True
                best_position = test_position
                low = mid  # Narrow range, closer to current frame position
        if 'ego' in target_vehicle_agent.vehicle_name:
            Scene._ego_time_step = Scene._ego_time_step - 1 if Scene._ego_time_step > 0 else 0
        
        if not 'static' in reference_vehicle_agent.vehicle_name:
            # Adjust to optimal position
            if if_adjust:
                target_vehicle_agent.cur_waypoint.transform.set_location(best_position)
            else:
                # print(calculate_distance(current_position, previous_position))
                if calculate_distance(current_position, previous_position) < 0.05:
                    target_vehicle_agent.cur_waypoint.transform.set_location(best_position)
                else:
                    target_vehicle_agent.cur_waypoint.transform.set_location(previous_position)
        else:
            target_vehicle_agent.cur_waypoint.transform.set_location(best_position)
        # target_vehicle_agent.cur_waypoint.transform.set_location(best_position)
        # cur_speed = target_vehicle_agent.get_speed()
        actual_speed = calculate_distance(best_position, previous_position) * Scene._FPS
        target_vehicle_agent.set_speed(actual_speed)  # Stop vehicle


    @staticmethod
    def check_collision(angle_threshold=np.pi/3):
        """
        Detect and resolve collisions between all vehicles in the scene
        """
        agent_dict = Scene._agent_dict  # Extract agent dict to avoid accessing Scene._agent_dict each time
        agent_ids = list(agent_dict.keys())  # Get all vehicle ID lists

        for i, id1 in enumerate(agent_ids):
            target_vehicle = agent_dict[id1]
            target_vehicle_loc = target_vehicle.cur_waypoint.transform.location[:2]
            target_vehicle_yaw = target_vehicle.cur_waypoint.transform.rotation[2]
            target_vehicle_bbox = target_vehicle.bounding_box
            target_vehicle_bbox = [target_vehicle_bbox[0]*0.9, target_vehicle_bbox[1]*0.9, target_vehicle_bbox[2]*0.9]
            target_vehicle_last_loc = target_vehicle.last_waypoint.transform.location[:2]

            # Inner loop starts from next element of outer loop to avoid duplicate detection
            for id2 in agent_ids[i + 1:]:
                if Scene._skip_ego and ('ego' in id1 or 'ego' in id2):
                    continue
                reference_vehicle = agent_dict[id2]
                reference_vehicle_loc = reference_vehicle.cur_waypoint.transform.location[:2]

                reference_vehicle_yaw = reference_vehicle.cur_waypoint.transform.rotation[2]
                reference_vehicle_bbox = reference_vehicle.bounding_box
                reference_vehicle_bbox = [reference_vehicle_bbox[0], reference_vehicle_bbox[1], reference_vehicle_bbox[2]]
                reference_vehicle_last_loc = reference_vehicle.last_waypoint.transform.location[:2]
                if calculate_distance(target_vehicle_loc, reference_vehicle_loc) > 20:
                    continue
                if is_collision(target_vehicle_loc,
                                target_vehicle_yaw,
                                target_vehicle_bbox,
                                reference_vehicle_loc,
                                reference_vehicle_yaw,
                                reference_vehicle_bbox):
                    Scene._collision_dict[id1] = id2
                    Scene._collision_dict[id2] = id1
                    catch_relation = False
                    for relation in Scene._interaction_relation:
                        if id1 in relation and id2 in relation:
                            Scene.adjust_position(reference_vehicle,target_vehicle)
                            catch_relation = True
                            break
                    if catch_relation:
                        continue

                    if 'static' in id1 or 'static' in id2:
                        continue
                    # print(f'Collision between {id1} and {id2}')
                    if 'static' in id1:
                        Scene.adjust_position(reference_vehicle, target_vehicle)
                        continue
                    elif 'static' in id2:
                        Scene.adjust_position(target_vehicle, reference_vehicle)
                        continue

                    relative_vector = calculate_relative_vector(target_vehicle_loc, reference_vehicle_loc)

                    # Calculate heading vectors of two vehicles
                    heading_vector1 = np.array([np.cos(target_vehicle_yaw), np.sin(target_vehicle_yaw)])
                    heading_vector2 = np.array([np.cos(reference_vehicle_yaw), np.sin(reference_vehicle_yaw)])

                    # Calculate angle between movement direction and relative position vector
                    angle1 = calculate_angle_between_vectors(heading_vector1, relative_vector)
                    angle2 = calculate_angle_between_vectors(heading_vector2, -relative_vector)

                    # Determine which vehicle's heading is more likely to cause collision
                    if angle1 < angle_threshold:
                        Scene.adjust_position(target_vehicle, reference_vehicle)  # Adjust active vehicle position based on collision angle
                        Scene._collision_dict[id1] = id2
                        Scene._collision_dict[id2] = id1
                    elif angle2 < angle_threshold:
                        Scene.adjust_position(reference_vehicle, target_vehicle)  # Adjust position of active vehicle
                        Scene._collision_dict[id1] = id2
                        Scene._collision_dict[id2] = id1
                    else:
                        pass

    @staticmethod
    def check_collision_w_layout():
        occ_grid = Scene._map.get_occupancy_grid()
        ego_agent = Scene._agent_dict['ego_vehicle']
        ego_loc = ego_agent.cur_waypoint.transform.location[:2]
        ego_yaw = ego_agent.cur_waypoint.transform.rotation[2]
        ego_bbox = ego_agent.bounding_box
        ego_bbox = [ego_bbox[0], ego_bbox[1], ego_bbox[2]]
        ego_last_loc = ego_agent.last_waypoint.transform.location[:2]
        ego_occ_grid = Scene._map.get_occupancy_grid_w_loc(ego_loc, ego_yaw, ego_bbox)
        # Check if collision with road edge occurs
        collision_grid = np.logical_and(occ_grid, ego_occ_grid)
        ego_collision = np.sum(collision_grid)
        del ego_occ_grid
        del collision_grid
        return ego_collision
    
    @staticmethod
    def is_collision_w_grid(target_vehicle_loc,
                                target_vehicle_yaw,
                                target_vehicle_bbox,
                                reference_vehicle_loc,
                                reference_vehicle_yaw,
                                reference_vehicle_bbox):
        target_vehicle_occ_grid = Scene._map.get_occupancy_grid_w_loc(target_vehicle_loc, target_vehicle_yaw, target_vehicle_bbox)
        reference_vehicle_occ_grid = Scene._map.get_occupancy_grid_w_loc(reference_vehicle_loc, reference_vehicle_yaw, reference_vehicle_bbox)
        if_overlap = compare_grids(target_vehicle_occ_grid, reference_vehicle_occ_grid)
        return if_overlap
    

    @staticmethod
    def set_mode(mode):
        """
        Set debug mode, in debug mode vehicle information for each step will be recorded for final information recording and plotting

        Args:
            debug_mode (bool): Whether to enable debug mode
        """
        Scene._mode = mode

    @staticmethod
    def save_traffic_flow():
        """
        Save traffic flow information
        """
        if Scene._mode == 'datagen':
            Scene._map.save_map_convertion(Scene._save_dir)
            vehicle_info_path = os.path.join(Scene._save_dir, 'vehicle_info')
            os.makedirs(vehicle_info_path, exist_ok=True)
            if Scene._car_dict_sequence is not None:
                car_info_dict_path = os.path.join(Scene._save_dir, 'car_info_dict.json')
                car_info_dict_renew = []
                for scene_time_step in range(0, len(Scene._car_dict_sequence)-1):
                    car_dict = Scene._car_dict_sequence[scene_time_step].copy()
                    car_dict_next = Scene._car_dict_sequence[scene_time_step + 1].copy()
                    car_dict_renew = dict()
                    cur_car_dict_folder_path = os.path.join(vehicle_info_path, str(scene_time_step).zfill(3))
                    os.makedirs(cur_car_dict_folder_path, exist_ok=True)
                    new_car_info_dict = dict()
                    ego_info_dict = dict()
                    for agent_name, agent_info in car_dict.items():
                        if 'ego' in agent_name:
                            ego_info_dict = agent_info
                        agent_info_next = car_dict_next[agent_name].copy()
                        agent_info_next['loc'] = car_dict[agent_name]['loc']
                        agent_info_next['rot'] = car_dict[agent_name]['rot']
                        new_car_info_dict[agent_name] = agent_info_next
                        # new_car_info_dict[agent_name] = agent_info
                    car_info_dict_renew.append(new_car_info_dict)
                    cur_ego_dict_save_path = os.path.join(cur_car_dict_folder_path, 'ego_info.json')
                    with open(cur_ego_dict_save_path, 'w') as file:
                        json.dump(ego_info_dict, file, indent=2)
                    cur_car_dict_save_path = os.path.join(cur_car_dict_folder_path, 'car_info.json')
                    with open(cur_car_dict_save_path, 'w') as file:
                        json.dump(new_car_info_dict, file, indent=2)
                with open(car_info_dict_path, 'w') as file:
                    json.dump(car_info_dict_renew, file, indent=2)
 
    @staticmethod
    def save_traffic_flow_end2end():
        """
        Save traffic flow information
        """
        Scene._map.save_map_convertion(Scene._save_dir)
        vehicle_info_path = os.path.join(Scene._save_dir, 'vehicle_info')
        os.makedirs(vehicle_info_path, exist_ok=True)
        if Scene._car_dict_sequence is not None:
            # car_info_dict_path = os.path.join(Scene._save_dir, 'car_info_dict.json')
            # with open(car_info_dict_path, 'w') as file:
            #     json.dump(Scene._car_dict_sequence, file, indent=2)
            scene_time_step = len(Scene._car_dict_sequence) - 1
            car_dict = Scene._car_dict_sequence[-1]
            cur_car_dict_folder_path = os.path.join(vehicle_info_path, str(scene_time_step))
            os.makedirs(cur_car_dict_folder_path, exist_ok=True)
            new_car_info_dict = dict()
            ego_info_dict = dict()
            for agent_name, agent_info in car_dict.items():
                if 'ego' in agent_name:
                    ego_info_dict = agent_info
                    continue
                new_car_info_dict[agent_name] = agent_info
            cur_ego_dict_save_path = os.path.join(cur_car_dict_folder_path, 'ego_info.json')
            with open(cur_ego_dict_save_path, 'w') as file:
                json.dump(ego_info_dict, file, indent=2)
            cur_car_dict_save_path = os.path.join(cur_car_dict_folder_path, 'car_info.json')
            with open(cur_car_dict_save_path, 'w') as file:
                json.dump(new_car_info_dict, file, indent=2)

    @staticmethod
    def get_agent_trajectory(agent_name, step_num=10):
        """
        Get agent trajectory

        Args:
            agent_name (str): Agent name
            step_num (int, optional): Number of steps. Defaults to 10.

        Returns:
            list: Trajectory
        """
        agent = Scene._agent_dict[agent_name]
        return agent.get_trajectory(step_num)

    @staticmethod
    def get_agent_final_trajectory(agent_name, step_num=10):
        """
        Get agent trajectory

        Args:
            agent_name (str): Agent name
            step_num (int, optional): Number of steps. Defaults to 10.

        Returns:
            list: Trajectory
        """
        agent = Scene._agent_dict[agent_name]
        return agent.get_final_trajectory(step_num)


    @staticmethod
    def judge_future_trajectory_interaction(agent_name_1, agent_name_2, step_num=20):
        """
        Determine if two agents' future trajectories interact

        Args:
            agent_name_1 (str): Agent 1 name
            agent_name_2 (str): Agent 2 name
            step_num (int, optional): Number of steps. Defaults to 10.

        Returns:
            bool: Whether interaction occurs
        """
        agent_1 = Scene._agent_dict[agent_name_1]
        agent_2 = Scene._agent_dict[agent_name_2]
        trajectory_1 = agent_1.get_trajectory(step_num)
        trajectory_2 = agent_2.get_trajectory(step_num)
        interaction_range_1 = agent_1.bounding_box[0]/2 * 0.8
        interaction_range_2 = agent_2.bounding_box[0]/2 * 0.8
        trajectory_1 = [x.transform.location[:2] for x in trajectory_1]
        trajectory_2 = [x.transform.location[:2] for x in trajectory_2]
        # Add current position at the beginning
        trajectory_1.insert(0, agent_1.cur_waypoint.transform.location[:2])
        trajectory_2.insert(0, agent_2.cur_waypoint.transform.location[:2])
        # Add previous position at the beginning
        trajectory_1.insert(0, agent_1.last_waypoint.transform.location[:2])
        trajectory_2.insert(0, agent_2.last_waypoint.transform.location[:2])
        return detect_route_interaction(trajectory_1, trajectory_2, interaction_range_1, interaction_range_2)

    @staticmethod
    def judge_future_trajectory_interaction_side(agent_name_1, agent_name_2, step_num_1=20, step_num_2=5):
        """
        Determine if two agents' future trajectories interact

        Args:
            agent_name_1 (str): Agent 1 name
            agent_name_2 (str): Agent 2 name
            step_num (int, optional): Number of steps. Defaults to 10.

        Returns:
            bool: Whether interaction occurs
        """
        agent_1 = Scene._agent_dict[agent_name_1]
        agent_2 = Scene._agent_dict[agent_name_2]
        trajectory_1 = agent_1.get_trajectory(step_num_1)
        trajectory_2 = agent_2.get_trajectory(step_num_2)
        interaction_range_1 = agent_1.bounding_box[0]/2 * 0.8
        interaction_range_2 = agent_2.bounding_box[0]/2 * 0.8
        trajectory_1 = [x.transform.location[:2] for x in trajectory_1]
        trajectory_2 = [x.transform.location[:2] for x in trajectory_2]
        # Add current position at the beginning
        trajectory_1.insert(0, agent_1.cur_waypoint.transform.location[:2])
        trajectory_2.insert(0, agent_2.cur_waypoint.transform.location[:2])
        # Add previous position at the beginning
        trajectory_1.insert(0, agent_1.last_waypoint.transform.location[:2])
        trajectory_2.insert(0, agent_2.last_waypoint.transform.location[:2])
        return detect_route_interaction(trajectory_1, trajectory_2, interaction_range_1, interaction_range_2)

    @staticmethod
    def get_close_z(location):
        """
        Get closest z coordinate

        Args:
            location (list): Position coordinates

        Returns:
            float: Closest z coordinate
        """
        return Scene._map.get_close_z(location)

    @staticmethod
    def judge_end_scene():
        """
        Determine if scene should end
        """
        if Scene._agent_dict['ego_vehicle'].end_route_flag:
            Scene._end_scene = True
        
        if Scene._ego_vehicle_ori_end_point is not None:
            if calculate_distance(Scene._ego_vehicle_ori_end_point.transform.location[:2], Scene._agent_dict['ego_vehicle'].cur_waypoint.transform.location[:2]) <= 0.75:
                Scene._end_scene_ori_end_point = True
        if Scene._end_scene_ori_end_point:
            Scene._cnt_before_end += 1
        if Scene._cnt_before_end > Scene._FPS:
            Scene._end_scene = True

        # end scene if ego vehicle is too slow
        if calculate_distance(Scene._agent_dict['ego_vehicle'].last_waypoint.transform.location[:2], Scene._agent_dict['ego_vehicle'].cur_waypoint.transform.location[:2]) < 0.1:
            Scene._ego_slow_speed_flag += 1
        else:
            Scene._ego_slow_speed_flag = 0
        if Scene._ego_slow_speed_flag > Scene._FPS * 4:
            Scene._end_scene = True

    @staticmethod
    def run_step_w_ego(cur_time_step, ego_control=False):
        """
        Execute one simulation step with ego vehicle control
        """
        Scene.judge_end_scene()
        time_step = 1.0/Scene._FPS # 1s/5 = 0.2s, as each time interval
        # Initialize based on _mode setting
        if Scene._mode == 'debug':
            if Scene._car_dict_sequence is None:
                Scene._car_dict_sequence = []
            car_dict = dict()
        elif Scene._mode == 'datagen':
            if Scene._car_dict_sequence is None:
                Scene._car_dict_sequence = []
            car_dict = dict()
        # Iterate through all existing agents
        if not ego_control:
            Scene._skip_ego = True
        else:
            Scene._skip_ego = False
        for agent_name, agent in Scene._agent_dict.items():
            if agent_name == 'ego_vehicle':
                if not ego_control:
                    agent.scenario_trigger = False
                    next_ego_waypoint = agent.plan_path[Scene._ego_time_step]
                    agent.set_last_waypoint(agent.cur_waypoint)
                    next_ego_loc = next_ego_waypoint.transform.location[:2] + [0]
                    next_ego_rot = next_ego_waypoint.transform.rotation
                    agent.cur_waypoint.transform.set_rotation(next_ego_rot)
                    agent.cur_waypoint.transform.set_location(next_ego_loc)
                    agent.update_information(next_ego_waypoint.transform)
                    Scene._ego_time_step += 1
                    continue
            if 'static' in agent_name:
                continue
            if Scene._only_ego_move and 'ego' not in agent_name:
                continue

            if agent.if_static:
                continue
            # Get current agent control information
            agent_control = Scene._agent_control_dict[agent_name]
            if agent.end_route_flag:
                Scene._agent_del_dict[agent_name] = agent
                Scene._agent_control_dict[agent_name] = agent.emergency_stop()
                continue
            # Temporarily set front and rear wheelbase to 1.3m
            agent_f_len = agent.f_len # Front wheelbase unit: m
            agent_r_len = agent.r_len # Rear wheelbase unit: m
            agent_control_angel = agent.control_angel
            # Get specific agent control information including throttle, brake, steering
            throttle = agent_control.throttle # throttle range:[0, 1] maximum value set to 0.75 in local planner unit: None
            brake = agent_control.brake # brake range:[0, 1] maximum value set to 0.3 in local planner unit: None
            steer = agent_control.steer # steer range:[-1, 1] maximum value set to 0.8 in local planner unit: None
            delta = steer*agent_control_angel*math.pi/180 # steer range:[-1, 1] -> delta range:[-40, 40] unit: deg
            
            cur_agent_loc = agent.cur_waypoint.transform.location # location [x, y, z] unit：m
            cur_agent_rot = agent.cur_waypoint.transform.rotation # rotation [roll, pitch, yaw] unit: rad 
            agent_x = cur_agent_loc[0] # x location unit: m
            agent_y = cur_agent_loc[1] # y location unit: m
            agent_yaw = cur_agent_rot[2] # yaw range:[-pi, pi] unit: rad
            agent_v = copy.deepcopy(agent.speed) # velocity unit: km/h
            max_acceleration = agent.max_acceleration # max_acceleration unit: m/s^2
            max_braking = agent.max_braking # max_braking unit: m/s^2
            if brake >= 0.001:
                agent_a = -max_braking * brake # brake deceleration
            else:
                agent_a = max_acceleration * throttle # acceleration
            if agent_name == 'ego_vehicle':
                agent_x_update, agent_y_update, agent_yaw_update, agent_speed_update, agent_omega_update= \
                    EgoKinematicModel(x=agent_x, 
                                y=agent_y,
                                yaw=agent_yaw,
                                v=agent_v / 3.6, # velocity unit: m/s
                                a=agent_a, # acceleration unit: m/s^2
                                delta=delta,
                                f_len=agent_f_len,
                                r_len=agent_r_len,
                                dt=time_step)
            else:
                agent_x_update, agent_y_update, agent_yaw_update, agent_speed_update, agent_omega_update = \
                    KinematicModel(x=agent_x, 
                                y=agent_y,
                                yaw=agent_yaw,
                                v=agent_v / 3.6, # velocity unit: m/s
                                a=agent_a, # acceleration unit: m/s^2
                                delta=delta,
                                f_len=agent_f_len,
                                r_len=agent_r_len,
                                dt=time_step)
            agent_next_loc = [agent_x_update ,agent_y_update , 0.0] # location [x, y, z] unit: m
            agent_next_rot = [0.0, 0.0, agent_yaw_update] # rotation [roll, pitch, yaw] unit: rad
            agent.set_last_waypoint(agent.cur_waypoint)
            agent.cur_waypoint.transform.set_rotation(agent_next_rot)
            agent.cur_waypoint.transform.set_location(agent_next_loc)
            updated_transform = agent.cur_waypoint.transform
            agent.update_information(updated_transform)
            agent_speed_update_in_km = agent_speed_update * 3.6 # velocity unit: km/h
            agent.set_speed(agent_speed_update_in_km) # velocity unit: km/h
            agent.set_omega(agent_omega_update) # angular velocity unit: rad/s
            agent.set_acceleration(agent_a) # acceleration unit: m/s^2
            agent.set_steer_value(delta) # steer value range:[-1*max_steer, 1*max_steer] unit: rad
        Scene.check_collision()
        Scene._interaction_relation_last = Scene._interaction_relation.copy()
        Scene._interaction_relation = []
        if Scene._mode == 'debug':
            for agent_name, agent in Scene._agent_dict.items():
                car_dict[agent_name] = dict()
                car_dict[agent_name]['loc'] = agent.cur_waypoint.transform.location
                car_dict[agent_name]['rot'] = agent.cur_waypoint.transform.rotation
                car_dict[agent_name]['bbox'] = agent.bounding_box
                car_dict[agent_name]['if_overtake'] = agent.if_overtake
                car_dict[agent_name]['if_tailgate'] = agent.if_tailgate
                car_dict[agent_name]['if_static'] = agent.if_static
        elif Scene._mode == 'datagen':
            for agent_name, agent in Scene._agent_dict.items():
                if 'static' in agent_name:
                    car_dict[agent_name] = dict()
                    car_dict[agent_name]['loc'] = agent.cur_waypoint.transform.location
                    car_dict[agent_name]['rot'] = agent.cur_waypoint.transform.rotation
                    car_dict[agent_name]['bbox'] = agent.bounding_box
                    car_dict[agent_name]['control'] = dict()
                    car_dict[agent_name]['control']['throttle'] = 0.0
                    car_dict[agent_name]['control']['brake'] = 0.0
                    car_dict[agent_name]['control']['steer'] = 0.0
                    car_dict[agent_name]['speed'] = 0.0
                    car_dict[agent_name]['omega'] = 0.0
                    car_dict[agent_name]['velocity_xy'] = [0.0, 0.0]
                    car_dict[agent_name]['acceleration_xy'] = [0.0, 0.0]
                    car_dict[agent_name]['steer_value'] = 0.0
                    car_dict[agent_name]['type'] = agent.vehicle_type
                    car_dict[agent_name]['if_overtake'] = agent.if_overtake
                    car_dict[agent_name]['if_tailgate'] = agent.if_tailgate
                    car_dict[agent_name]['if_static'] = agent.if_static
                else:
                    car_dict[agent_name] = dict()
                    car_dict[agent_name]['loc'] = agent.cur_waypoint.transform.location
                    car_dict[agent_name]['rot'] = agent.cur_waypoint.transform.rotation
                    car_dict[agent_name]['bbox'] = agent.bounding_box
                    car_dict[agent_name]['control'] = dict()
                    car_dict[agent_name]['control']['throttle'] = Scene._agent_control_dict[agent_name].throttle
                    car_dict[agent_name]['control']['brake'] = Scene._agent_control_dict[agent_name].brake
                    car_dict[agent_name]['control']['steer'] = Scene._agent_control_dict[agent_name].steer
                    car_dict[agent_name]['speed'] = agent.speed / 3.6
                    car_dict[agent_name]['omega'] = agent.omega
                    car_dict[agent_name]['velocity_xy'] = [agent.velocity_xy[0] / 3.6, agent.velocity_xy[1] / 3.6]
                    car_dict[agent_name]['acceleration'] = agent.acceleration
                    car_dict[agent_name]['acceleration_xy'] = [agent.acceleration_xy[0], agent.acceleration_xy[1]]
                    car_dict[agent_name]['steer_value'] = agent.steer_value
                    car_dict[agent_name]['type'] = agent.vehicle_type
                    car_dict[agent_name]['if_overtake'] = agent.if_overtake
                    car_dict[agent_name]['if_tailgate'] = agent.if_tailgate
                    car_dict[agent_name]['if_static'] = agent.if_static
        Scene.check_collision_w_layout()
        if Scene._mode:
            Scene._car_dict_sequence.append(car_dict)
        Scene._agent_del_dict.clear()
    
    @staticmethod
    def run_step_w_ego_end2end(cur_time_step, ego_control=False):
        Scene.judge_end_scene()
        time_step = 1.0/Scene._FPS # 1s/5 = 0.2s, as each time interval
        if Scene._mode == 'debug':
            if Scene._car_dict_sequence is None:
                Scene._car_dict_sequence = []
            car_dict = dict()
        elif Scene._mode == 'datagen':
            if Scene._car_dict_sequence is None:
                Scene._car_dict_sequence = []
            car_dict = dict()
        # Iterate through all existing agents
        if not ego_control:
            Scene._skip_ego = True
        else:
            Scene._skip_ego = False
        for agent_name, agent in Scene._agent_dict.items():
            if agent_name == 'ego_vehicle':
                if not ego_control:
                    agent.scenario_trigger = False
                    next_ego_waypoint = agent.plan_path[Scene._ego_time_step]                                        
                    next_ego_waypoint = agent._local_planner.target_waypoint
                    next_ego_waypoint.transform.location = list(next_ego_waypoint.transform.location)
                    next_ego_waypoint.transform.rotation = list(next_ego_waypoint.transform.rotation)
                    
                    agent.set_last_waypoint(agent.cur_waypoint)
                    next_ego_loc = next_ego_waypoint.transform.location[:2] + [0]
                    next_ego_rot = next_ego_waypoint.transform.rotation
                    agent.cur_waypoint.transform.set_rotation(next_ego_rot)
                    agent.cur_waypoint.transform.set_location(next_ego_loc)
                    agent.update_information(next_ego_waypoint.transform)
                    Scene._ego_time_step += 1
                    continue

            if 'static' in agent_name:
                continue
            if Scene._only_ego_move and 'ego' not in agent_name:
                continue
            if agent.if_static:
                continue
            # Get current agent control information
            agent_control = Scene._agent_control_dict[agent_name]
            if agent.end_route_flag:
                Scene._agent_del_dict[agent_name] = agent
                Scene._agent_control_dict[agent_name] = agent.emergency_stop()
                continue
            # Temporarily set front and rear wheelbase to 1.3m
            agent_f_len = agent.f_len # front wheelbase unit: m
            agent_r_len = agent.r_len # rear wheelbase unit: m
            agent_control_angel = agent.control_angel
            # Get specific agent control information including throttle, brake, and steering
            throttle = agent_control.throttle # throttle range:[0, 1] max value set to 0.75 in local planner unit: None
            brake = agent_control.brake # brake range:[0, 1] max value set to 0.3 in local planner unit: None
            steer = agent_control.steer # steer range:[-1, 1] max value set to 0.8 in local planner unit: None
            delta = steer*agent_control_angel*math.pi/180 # steer range:[-1, 1] -> delta range:[-40, 40] unit: deg
            
            cur_agent_loc = agent.cur_waypoint.transform.location # location [x, y, z] unit：m
            cur_agent_rot = agent.cur_waypoint.transform.rotation # rotation [roll, pitch, yaw] unit: rad 
            agent_x = cur_agent_loc[0] # x_loc unit：m
            agent_y = cur_agent_loc[1] # y_loc unit：m
            agent_yaw = cur_agent_rot[2] # yaw range:[-pi, pi] unit: rad
            agent_v = copy.deepcopy(agent.speed) # velocity unit:km/h
            max_acceleration = agent.max_acceleration # max_acceleration unit: m/s^2
            max_braking = agent.max_braking # max_braking unit: m/s^2
            if brake >= 0.001:
                agent_a = -max_braking * brake # brake deceleration
            else:
                agent_a = max_acceleration * throttle # acceleration
                
            agent_x_update, agent_y_update, agent_yaw_update, agent_speed_update, agent_omega_update = \
                KinematicModel(x=agent_x, 
                               y=agent_y,
                               yaw=agent_yaw,
                               v=agent_v / 3.6, # velocity unit: m/s
                               a=agent_a, # acceleration unit: m/s^2
                               delta=delta,
                               f_len=agent_f_len,
                               r_len=agent_r_len,
                               dt=time_step)
            agent_next_loc = [agent_x_update ,agent_y_update , 0.0] # location [x, y, z] unit：m
            agent_next_rot = [0.0, 0.0, agent_yaw_update] # rotation [roll, pitch, yaw] unit: rad
            agent.set_last_waypoint(agent.cur_waypoint)
            agent.cur_waypoint.transform.set_rotation(agent_next_rot)
            agent.cur_waypoint.transform.set_location(agent_next_loc)
            updated_transform = agent.cur_waypoint.transform
            agent.update_information(updated_transform)
            agent_speed_update_in_km = agent_speed_update * 3.6 # velocity unit: km/h
            agent.set_speed(agent_speed_update_in_km) # velocity unit: km/h
            agent.set_omega(agent_omega_update) # angular velocity unit: rad/s
            agent.set_acceleration(agent_a) # acceleration unit: m/s^2
            agent.set_steer_value(delta) # steer value range:[-1*max_steer, 1*max_steer] unit: rad
        Scene.check_collision()
        Scene._interaction_relation_last = Scene._interaction_relation.copy()
        Scene._interaction_relation = []
        ego_collision_w_layout = Scene.check_collision_w_layout()
        if Scene._mode == 'debug':
            for agent_name, agent in Scene._agent_dict.items():
                car_dict[agent_name] = dict()
                car_dict[agent_name]['loc'] = agent.cur_waypoint.transform.location
                car_dict[agent_name]['rot'] = agent.cur_waypoint.transform.rotation
                car_dict[agent_name]['bbox'] = agent.bounding_box
                car_dict[agent_name]['if_overtake'] = agent.if_overtake
                car_dict[agent_name]['if_tailgate'] = agent.if_tailgate
                car_dict[agent_name]['if_static'] = agent.if_static
        elif Scene._mode == 'datagen':
            for agent_name, agent in Scene._agent_dict.items():
                if 'static' in agent_name:
                    car_dict[agent_name] = dict()
                    car_dict[agent_name]['loc'] = agent.cur_waypoint.transform.location
                    car_dict[agent_name]['rot'] = agent.cur_waypoint.transform.rotation
                    car_dict[agent_name]['bbox'] = agent.bounding_box
                    car_dict[agent_name]['control'] = dict()
                    car_dict[agent_name]['control']['throttle'] = 0.0
                    car_dict[agent_name]['control']['brake'] = 0.0
                    car_dict[agent_name]['control']['steer'] = 0.0
                    car_dict[agent_name]['speed'] = 0.0
                    car_dict[agent_name]['omega'] = 0.0
                    car_dict[agent_name]['velocity_xy'] = [0.0, 0.0]
                    car_dict[agent_name]['acceleration_xy'] = [0.0, 0.0]
                    car_dict[agent_name]['steer_value'] = 0.0
                    car_dict[agent_name]['type'] = agent.vehicle_type
                    car_dict[agent_name]['if_overtake'] = agent.if_overtake
                    car_dict[agent_name]['if_tailgate'] = agent.if_tailgate
                    car_dict[agent_name]['if_static'] = agent.if_static
                    if agent_name in Scene._collision_dict.keys():
                        car_dict[agent_name]['collision'] = Scene._collision_dict[agent_name]
                    else:
                        car_dict[agent_name]['collision'] = None
                else:
                    car_dict[agent_name] = dict()
                    car_dict[agent_name]['loc'] = agent.cur_waypoint.transform.location
                    car_dict[agent_name]['rot'] = agent.cur_waypoint.transform.rotation
                    car_dict[agent_name]['bbox'] = agent.bounding_box
                    car_dict[agent_name]['control'] = dict()
                    car_dict[agent_name]['control']['throttle'] = Scene._agent_control_dict[agent_name].throttle
                    car_dict[agent_name]['control']['brake'] = Scene._agent_control_dict[agent_name].brake
                    car_dict[agent_name]['control']['steer'] = Scene._agent_control_dict[agent_name].steer
                    car_dict[agent_name]['speed'] = agent.speed / 3.6
                    car_dict[agent_name]['omega'] = agent.omega
                    car_dict[agent_name]['velocity_xy'] = [agent.velocity_xy[0] / 3.6, agent.velocity_xy[1] / 3.6]
                    car_dict[agent_name]['acceleration_xy'] = [agent.acceleration_xy[0], agent.acceleration_xy[1]]
                    car_dict[agent_name]['steer_value'] = agent.steer_value
                    car_dict[agent_name]['type'] = agent.vehicle_type
                    car_dict[agent_name]['if_overtake'] = agent.if_overtake
                    car_dict[agent_name]['if_tailgate'] = agent.if_tailgate
                    car_dict[agent_name]['if_static'] = agent.if_static
                    if agent_name in Scene._collision_dict.keys():
                        car_dict[agent_name]['collision'] = Scene._collision_dict[agent_name]
                    else:
                        car_dict[agent_name]['collision'] = None
                    if agent_name == 'ego_vehicle':
                        # print(type(ego_collision_w_layout))
                        car_dict[agent_name]['collision_w_layout'] = bool(ego_collision_w_layout)
                    else:
                        car_dict[agent_name]['collision_w_layout'] = None
        
        Scene._collision_dict = dict()

        if Scene._mode:
            Scene._car_dict_sequence.append(car_dict)
        Scene._agent_del_dict.clear()

    @staticmethod
    def run_step_w_ego_end2end_by_trajectory(cur_time_step, ego_control=False):
        Scene.judge_end_scene()
        time_step = 1.0/Scene._FPS # 1s/5 = 0.2s, as each time interval
        # Initialize based on mode settings
        if Scene._mode == 'debug':
            if Scene._car_dict_sequence is None:
                Scene._car_dict_sequence = []
            car_dict = dict()
        elif Scene._mode == 'datagen':
            if Scene._car_dict_sequence is None:
                Scene._car_dict_sequence = []
            car_dict = dict()
        # Iterate through all existing agents
        if not ego_control:
            Scene._skip_ego = True
        else:
            Scene._skip_ego = False
        for agent_name, agent in Scene._agent_dict.items():
            if 'static' in agent_name:
                continue
            # Get current agent control information
            agent_control = Scene._agent_control_dict[agent_name]
            if agent.end_route_flag:
                Scene._agent_del_dict[agent_name] = agent
                Scene._agent_control_dict[agent_name] = agent.emergency_stop()
                continue
            if 'ego' in agent_name:
                cur_agent_loc = agent.cur_waypoint.transform.location # location [x, y, z] unit：m
                cur_agent_rot = agent.cur_waypoint.transform.rotation # rotation [roll, pitch, yaw] unit: rad 
                next_waypoint, agent_speed_update, agent_a, agent_omega_update = agent.get_one_step_trajectory()
                agent.set_last_waypoint(agent.cur_waypoint)
                agent.cur_waypoint.transform.set_rotation(next_waypoint.transform.rotation)
                agent.cur_waypoint.transform.set_location(next_waypoint.transform.location)
                updated_transform = agent.cur_waypoint.transform
                agent.update_information(updated_transform)

                if cur_time_step%5==0: #or cur_time_step%5==0:
                    agent_speed_update_in_km = agent_speed_update * 3.6 # velocity unit: km/h
                    agent.set_speed(agent_speed_update_in_km) # velocity unit: km/h
                    agent.set_omega(agent_omega_update) # angular velocity unit: rad/s
                    agent.set_acceleration(agent_a) # acceleration unit: m/s^2
                    agent.set_steer_value(0.0) # steer value range:[-1*max_steer, 1*max_steer] unit: rad
            else:
                # Temporarily set front and rear wheelbase to 1.3m
                agent_f_len = agent.f_len # front wheelbase unit: m
                agent_r_len = agent.r_len # rear wheelbase unit: m
                agent_control_angel = agent.control_angel
                # Get specific agent control information including throttle, brake, and steering
                throttle = agent_control.throttle # throttle range:[0, 1] max value set to 0.75 in local planner unit: None
                brake = agent_control.brake # brake range:[0, 1] max value set to 0.3 in local planner unit: None
                steer = agent_control.steer # steer range:[-1, 1] max value set to 0.8 in local planner unit: None
                delta = steer*agent_control_angel*math.pi/180 # steer range:[-1, 1] -> delta range:[-40, 40] unit: deg
                
                cur_agent_loc = agent.cur_waypoint.transform.location # location [x, y, z] unit：m
                cur_agent_rot = agent.cur_waypoint.transform.rotation # rotation [roll, pitch, yaw] unit: rad 
                agent_x = cur_agent_loc[0] # x_loc unit：m
                agent_y = cur_agent_loc[1] # y_loc unit：m
                agent_yaw = cur_agent_rot[2] # yaw range:[-pi, pi] unit: rad
                agent_v = copy.deepcopy(agent.speed) # velocity unit:km/h
                max_acceleration = agent.max_acceleration # max_acceleration unit: m/s^2
                max_braking = agent.max_braking # max_braking unit: m/s^2
                if brake >= 0.001:
                    agent_a = -max_braking * brake # brake deceleration
                else:
                    agent_a = max_acceleration * throttle # acceleration
                agent_x_update, agent_y_update, agent_yaw_update, agent_speed_update, agent_omega_update = \
                    KinematicModel(x=agent_x, 
                                y=agent_y,
                                yaw=agent_yaw,
                                v=agent_v / 3.6, # velocity unit: m/s
                                a=agent_a, # acceleration unit: m/s^2
                                delta=delta,
                                f_len=agent_f_len,
                                r_len=agent_r_len,
                                dt=time_step)
                agent_next_loc = [agent_x_update ,agent_y_update , 0.0] # location [x, y, z] unit：m
                agent_next_rot = [0.0, 0.0, agent_yaw_update] # rotation [roll, pitch, yaw] unit: rad
                agent.set_last_waypoint(agent.cur_waypoint)
                agent.cur_waypoint.transform.set_rotation(agent_next_rot)
                agent.cur_waypoint.transform.set_location(agent_next_loc)
                updated_transform = agent.cur_waypoint.transform
                agent.update_information(updated_transform)
                agent_speed_update_in_km = agent_speed_update * 3.6 # velocity unit: km/h
                agent.set_speed(agent_speed_update_in_km) # velocity unit: km/h
                agent.set_omega(agent_omega_update) # angular velocity unit: rad/s
                agent.set_acceleration(agent_a) # acceleration unit: m/s^2
                agent.set_steer_value(delta) # steer value range:[-1*max_steer, 1*max_steer] unit: rad
        Scene.check_collision()
        Scene._interaction_relation_last = Scene._interaction_relation.copy()
        Scene._interaction_relation = []
        ego_collision_w_layout = Scene.check_collision_w_layout()
        if Scene._mode == 'debug':
            for agent_name, agent in Scene._agent_dict.items():
                car_dict[agent_name] = dict()
                car_dict[agent_name]['loc'] = agent.cur_waypoint.transform.location
                car_dict[agent_name]['rot'] = agent.cur_waypoint.transform.rotation
                car_dict[agent_name]['bbox'] = agent.bounding_box
                car_dict[agent_name]['if_overtake'] = agent.if_overtake
                car_dict[agent_name]['if_tailgate'] = agent.if_tailgate
                car_dict[agent_name]['if_static'] = agent.if_static
        elif Scene._mode == 'datagen':
            for agent_name, agent in Scene._agent_dict.items():
                if 'static' in agent_name:
                    car_dict[agent_name] = dict()
                    car_dict[agent_name]['loc'] = agent.cur_waypoint.transform.location
                    car_dict[agent_name]['rot'] = agent.cur_waypoint.transform.rotation
                    car_dict[agent_name]['bbox'] = agent.bounding_box
                    car_dict[agent_name]['control'] = dict()
                    car_dict[agent_name]['control']['throttle'] = 0.0
                    car_dict[agent_name]['control']['brake'] = 0.0
                    car_dict[agent_name]['control']['steer'] = 0.0
                    car_dict[agent_name]['speed'] = 0.0
                    car_dict[agent_name]['omega'] = 0.0
                    car_dict[agent_name]['velocity_xy'] = [0.0, 0.0]
                    car_dict[agent_name]['acceleration_xy'] = [0.0, 0.0]
                    car_dict[agent_name]['steer_value'] = 0.0
                    car_dict[agent_name]['type'] = agent.vehicle_type
                    car_dict[agent_name]['if_overtake'] = agent.if_overtake
                    car_dict[agent_name]['if_tailgate'] = agent.if_tailgate
                    car_dict[agent_name]['if_static'] = agent.if_static
                    if agent_name in Scene._collision_dict.keys():
                        car_dict[agent_name]['collision'] = Scene._collision_dict[agent_name]
                    else:
                        car_dict[agent_name]['collision'] = None
                else:
                    car_dict[agent_name] = dict()
                    car_dict[agent_name]['loc'] = agent.cur_waypoint.transform.location
                    car_dict[agent_name]['rot'] = agent.cur_waypoint.transform.rotation
                    car_dict[agent_name]['bbox'] = agent.bounding_box
                    car_dict[agent_name]['control'] = dict()
                    car_dict[agent_name]['control']['throttle'] = Scene._agent_control_dict[agent_name].throttle
                    car_dict[agent_name]['control']['brake'] = Scene._agent_control_dict[agent_name].brake
                    car_dict[agent_name]['control']['steer'] = Scene._agent_control_dict[agent_name].steer
                    car_dict[agent_name]['speed'] = agent.speed / 3.6
                    car_dict[agent_name]['omega'] = agent.omega
                    car_dict[agent_name]['velocity_xy'] = [agent.velocity_xy[0] / 3.6, agent.velocity_xy[1] / 3.6]
                    car_dict[agent_name]['acceleration_xy'] = [agent.acceleration_xy[0], agent.acceleration_xy[1]]
                    car_dict[agent_name]['steer_value'] = agent.steer_value
                    car_dict[agent_name]['type'] = agent.vehicle_type
                    car_dict[agent_name]['if_overtake'] = agent.if_overtake
                    car_dict[agent_name]['if_tailgate'] = agent.if_tailgate
                    car_dict[agent_name]['if_static'] = agent.if_static
                    if agent_name in Scene._collision_dict.keys():
                        car_dict[agent_name]['collision'] = Scene._collision_dict[agent_name]
                        #Scene._agent_dict[agent_name].end_route_flag = True
                    else:
                        car_dict[agent_name]['collision'] = None
                    if agent_name == 'ego_vehicle':
                        # print(type(ego_collision_w_layout))
                        car_dict[agent_name]['collision_w_layout'] = bool(ego_collision_w_layout)
                    else:
                        car_dict[agent_name]['collision_w_layout'] = None
        
        Scene._collision_dict = dict()
        if Scene._mode:
            Scene._car_dict_sequence.append(car_dict)
        Scene._agent_del_dict.clear()