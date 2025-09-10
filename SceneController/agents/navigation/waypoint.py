#!/usr/bin/env python3
"""
This module contains the Waypoint and Transform classes for navigation.
"""
import numpy as np
import math

class Transform():
    def __init__(self, transform_input):
        self.location = transform_input['location']
        self.rotation = transform_input['rotation']

    def set_transform(self, transform):
        self.location = self.set_location(transform.location)
        self.rotation = self.set_rotation(transform.rotation)

    def set_location(self, location):
        from SceneController.scene import Scene
        if type(location) == np.ndarray:
            location = location.tolist()
        if len(location) == 3:  
            z = Scene._map.get_close_z(location)
            self.location = location[:2] + [z]
        elif len(location) == 2:
            z = Scene._map.get_close_z(location+[0.0])
            self.location = location + [z]
        else:
            raise ValueError('Location should have 2 or 3 elements but got {}'.format(len(location)))

    def set_rotation(self, rotation):
        self.rotation = rotation

    def get_location(self):
        from SceneController.scene import Scene
        if len(self.location) == 3:
            return self.location
        elif len(self.location) == 2:
            z = Scene._map.get_close_z(self.location+[0.0])
            self.location = self.location + [z]
            return self.location
        else:
            raise ValueError('Location should have 2 or 3 elements but got {}'.format(len(self.location)))
    
    def get_rotation(self):
        return self.rotation
    
    def get_yaw(self):
        return self.rotation[2]

    def get_forward_vector(self):
        forward_vector = np.array([math.cos(self.rotation[2]), math.sin(self.rotation[2])])
        return forward_vector
    
    def get_right_vector(self):
        right_vector = np.array([-math.sin(self.rotation[2]), math.cos(self.rotation[2])])
        return right_vector
    
    def get_left_vector(self):
        left_vector = np.array([math.sin(self.rotation[2]), -math.cos(self.rotation[2])])
        return left_vector
    
    def get_backward_vector(self):
        backward_vector = np.array([-math.cos(self.rotation[2]), -math.sin(self.rotation[2])])
        return backward_vector

class Waypoint():
    def __init__(self, waypoint_config):
        self.transform = Transform(waypoint_config['transform'])
        self.road_option = waypoint_config['road_option']
        self.lane_id = waypoint_config['lane_id']
        self.lane_point_idx = waypoint_config['lane_point_idx']
        self.is_junction = waypoint_config['is_junction']
        self.left_lane = waypoint_config['left_lane']
        self.right_lane = waypoint_config['right_lane']
    
    def set_transform(self, transform):
        self.transform = Transform(transform)
    
    def set_location(self, location):
        self.transform.set_location(location)

    def set_rotation(self, rotation):
        self.transform.set_rotation(rotation)

    def set_road_option(self, road_option):
        self.road_option = road_option

    def get_transform(self):
        return self.transform
    
    def get_location(self):
        return self.transform.get_location()
    
    def get_rotation(self):
        return self.transform.get_rotation()
    
    def get_road_option(self):
        return self.road_option
    
    def get_left_lane(self):
        if self.left_lane:
            return self.left_lane
        else:
            return None
    
    def get_right_lane(self):
        if self.right_lane:
            return self.right_lane
        else:
            return None

