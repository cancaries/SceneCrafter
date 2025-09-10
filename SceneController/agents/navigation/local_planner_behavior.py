#!/usr/bin/env python3
"""
This module contains a local planner to perform
low-level waypoint following based on PID controllers. 
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../..'))
from collections import deque
from enum import Enum
from agents.navigation.controller import VehiclePIDController, Control
from .tools.misc import calculate_distance, calculate_local_global_angle, interpolate_locations_by_steps
from SceneController.scene import Scene

class RoadOption(Enum):
    """
    Enumeration of possible navigation actions for autonomous vehicle routing.
    
    This enum defines the topological configurations available when transitioning
    between lane segments. These options guide the local planner in making
    navigation decisions at intersections, lane changes, and route following.
    
    Values:
        VOID (-1): No valid navigation option, typically indicates an error state
        TURNLEFT (1): Turn left at intersection or junction
        TURNRIGHT (2): Turn right at intersection or junction
        STRAIGHT (3): Continue straight through intersection
        LANEFOLLOW (4): Follow the current lane without changing
        CHANGELANELEFT (5): Change lanes to the left
        CHANGELANERIGHT (6): Change lanes to the right
        INTURN (7): Special case for in-turn navigation
        
    Usage:
        Used by route planners to specify navigation intentions at waypoints.
        The local planner interprets these options to generate appropriate
        steering and speed commands.
    """
    VOID = -1
    TURNLEFT = 1
    TURNRIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6
    INTURN = 7


class LocalPlanner(object):
    """
    Advanced local trajectory planner for autonomous vehicle waypoint following.
    
    This class implements comprehensive waypoint following using cascaded PID controllers
    for both lateral (steering) and longitudinal (speed) control. It manages dynamic
    trajectory updates, waypoint buffering, and collision-safe navigation.
    """

    # Minimum distance to target waypoint as a percentage
    # (e.g. within 80% of total distance)

    # FPS used for dt
    FPS = 10

    def __init__(self, agent, min_distance=4.5):
        self._vehicle_name = agent.vehicle_name
        self._speed_limit = agent.speed_limit
        self._target_speed = None
        self.sampling_radius = None
        self._min_distance = None
        self._current_waypoint = agent.cur_waypoint
        self.target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        self._pid_controller = None
        self.waypoints_queue = deque(maxlen=20000)  # queue with tuples of (waypoint, RoadOption)
        self._buffer_size = 6
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._init_controller(min_distance=min_distance)  # initializing controller

    def _init_controller(self,min_distance=4.5):
        """
        Controller initialization.

        dt -- time difference between physics control in seconds.
        This is can be fixed from server side
        using the arguments -benchmark -fps=F, since dt = 1/F

        target_speed -- desired cruise speed in km/h

        min_distance -- minimum distance to remove waypoint from queue

        lateral_dict -- dictionary of arguments to setup the lateral PID controller
                            {'K_P':, 'K_D':, 'K_I':, 'dt'}

        longitudinal_dict -- dictionary of arguments to setup the longitudinal PID controller
                            {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        # Default parameters
        self.args_lat_hw_dict = {
            'K_P': 0.75,
            'K_D': 0.02,
            'K_I': 0.3,
            'dt': 1.0 / self.FPS}
        self.args_lat_city_dict = {
            'K_P': 0.65,
            'K_D': 0.02,
            'K_I': 0.4,
            'dt': 1.0 / self.FPS}
        self.args_long_hw_dict = {
            'K_P': 0.37,
            'K_D': 0.024,
            'K_I': 0.032,
            'dt': 1.0 / self.FPS}
        self.args_long_city_dict = {
            'K_P': 0.20,
            'K_D': 0.05,
            'K_I': 0.07,
            'dt': 1.0 / self.FPS}

        self._global_plan = False

        self._target_speed = self._speed_limit

        self._min_distance = min_distance

        #print('target_speed:',target_speed)
        if self._speed_limit > 50:
            args_lat = self.args_lat_hw_dict
            args_long = self.args_long_hw_dict
        else:
            args_lat = self.args_lat_city_dict
            args_long = self.args_long_city_dict

        max_throttle = 0.45 if self._vehicle_name == 'ego_vehicle' else 0.75
        max_brake = 0.4 if self._vehicle_name == 'ego_vehicle' else 1.0
        self._pid_controller = VehiclePIDController(self._vehicle_name,
                                                    max_throttle=max_throttle,
                                                    max_brake=max_brake,
                                                    args_lateral=args_lat,
                                                    args_longitudinal=args_long)
        

    def set_speed(self, speed):
        self._target_speed = speed

    def set_global_plan(self, current_plan, clean=False, clean_global=False):
        """
        Set a new global route plan for the local planner to follow.
        
        This method initializes the planner with a complete route consisting of
        waypoints with associated navigation options. It handles both setting new
        plans and clearing existing ones based on the parameters.
        
        Args:
            current_plan (list): List of waypoints forming the global route.
                               Each waypoint should be a tuple containing:
                               (waypoint, RoadOption) where waypoint is a
                               location and RoadOption specifies the navigation
                               action at that point.
            clean (bool, optional): If True, clears the waypoint buffer before
                                  setting the new plan. Default is False.
            clean_global (bool, optional): If True, completely resets the planner
                                         by clearing global plan, buffers, and
                                         current plan. Default is False. 
        Note:
            If the waypoint queue is empty after setting the plan, a warning
            message 'empty waypoint' will be printed.
        """
        if clean_global:
            self.waypoints_queue.clear()
        for elem in current_plan:
            self.waypoints_queue.append(elem)
        if clean:
            self._waypoint_buffer.clear()
            for _ in range(self._buffer_size):
                if self.waypoints_queue:
                    self._waypoint_buffer.append(
                        self.waypoints_queue.popleft())
                else:
                    break
        self._global_plan = True

    def set_current_plan(self, current_plan):
        interpolate_current_plan = interpolate_locations_by_steps([elem.transform.location for elem in current_plan],steps=2)
        for elem in list(self.waypoints_queue):
            for current_elem in interpolate_current_plan:
                if calculate_distance(elem.transform.location, current_elem) < 3 :
                    self.waypoints_queue.remove(elem)
                    break
        
        self._waypoint_buffer.clear()
        for elem in current_plan:
            self._waypoint_buffer.append(elem)

    def update_current_plan_w_trajectory(self, current_plan):
        for elem in list(self.waypoints_queue):
            for current_elem in current_plan:
                if calculate_distance(elem.transform.location, current_elem.transform.location) < 3:
                    self.waypoints_queue.remove(elem)
                    break

        self._waypoint_buffer.clear()
        for elem in current_plan:
            self._waypoint_buffer.append(elem)

    def get_current_plan(self):
        return self._waypoint_buffer

    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        Retrieve waypoint and navigation direction at a specified distance ahead.
        
        This method provides lookahead information for advanced planning by
        returning the waypoint and road option that will be encountered after
        a specified number of steps.
        
        Args:
            steps (int): Number of steps ahead to look. Default is 3.
                       Each step corresponds to one waypoint in the buffer.
        
        Returns:
            tuple: (waypoint, road_option) where:
                   - waypoint: The waypoint at the specified lookahead distance
                   - road_option: The navigation action (RoadOption) at that waypoint
        """
        if len(self.waypoints_queue) > steps:
            wpt = self.waypoints_queue[steps]
            direction = wpt.road_option
            return wpt, direction

        else:
            try:
                wpt = self.waypoints_queue[-1]
                direction  = wpt.road_option
                return wpt, direction
            except IndexError as i:
                return None, RoadOption.VOID

    def get_trajectory(self, step_num=10):
        """
        Retrieve the complete planned trajectory as a list of waypoints.
        
        This method returns the combined trajectory consisting of both the active
        waypoint buffer and the remaining waypoints in the queue. The trajectory
        is ordered from the current position to the final destination.
        
        Args:
            step_num (int): Maximum number of waypoints to return from buffer+queue.
                           If step_num exceeds total waypoints, all are returned.
                           Default is 10.
        
        Returns:
            list: Complete trajectory as a list of waypoints. The list contains:
                  1. All waypoints in the active buffer (currently being followed)
                  2. All remaining waypoints in the queue (future path)
        """
        trajectory = []
        for i in range(len(self._waypoint_buffer)):
            trajectory.append(self._waypoint_buffer[i])
        for i in range(len(self.waypoints_queue)):
            trajectory.append(self.waypoints_queue[i])
        return trajectory

    def get_final_trajectory(self, step_num=10):
        """
        Retrieve the complete planned trajectory up to a specified number of steps.
        
        This method returns a combined trajectory consisting of both the active
        waypoint buffer and remaining waypoints in the queue, limited by step_num.
        
        Args:
            step_num (int): Maximum number of waypoints to return from both
                           buffer and queue. Default is 10.
        
        Returns:
            list: Trajectory as a list of waypoints, ordered from current position
                  to future waypoints. Contains:
                  1. Up to step_num waypoints from the active buffer
                  2. Up to step_num waypoints from the remaining queue
        """
        trajectory = []
        for i in range(len(self.waypoints_queue)):
            if i < step_num:
                trajectory.append(self.waypoints_queue[-1-i])
            else:
                return trajectory

    def done(self):
        """
        Check if the local planner has completed all waypoints in the route.
        
        This method determines whether the vehicle has successfully navigated
        through all waypoints in both the active buffer and the remaining queue.
        The planner considers the route complete when both the waypoint buffer
        and the main queue are empty, indicating no more waypoints to follow.
        
        Returns:
            bool: True if both waypoint buffer and queue are empty, indicating
                  the route is complete. False otherwise.
        """
        return len(self.waypoints_queue) == 0 and len(self._waypoint_buffer) == 0

    def run_step(self, target_speed=None, debug=False, update_waypoints_queue=True):
        """
        Execute one complete step of local trajectory planning and control.
        
        This is the main execution method that performs a full planning cycle:
        1. Updates vehicle state (position, velocity)
        2. Refills waypoint buffer from queue if needed
        3. Runs PID controllers for lateral and longitudinal control
        4. Returns vehicle control commands
        
        Args:
            target_speed (float, optional): Desired vehicle speed in m/s. 
                                        If None, uses previously set target speed.
            debug (bool): Enable debug output for waypoint visualization.
                          Default is False.
            update_waypoints_queue (bool): Whether to refill waypoint buffer from queue.
                                         Default is True for normal operation.
        
        Returns:
            Control: Vehicle control commands containing:
                    - throttle: Acceleration command [0.0, 1.0]
                    - brake: Braking command [0.0, 1.0]  
                    - steer: Steering command [-1.0, 1.0]
                    - reverse: Reverse gear flag
                    - hand_brake: Emergency brake flag
                    - manual_gear_shift: Manual transmission flag
        """

        if target_speed is not None:
            self._target_speed = target_speed
        else:
            self._target_speed = self._speed_limit

        if len(self.waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            control = Control()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
            return control

        self._current_waypoint = Scene._agent_dict[self._vehicle_name].cur_waypoint

        vehicle_location = self._current_waypoint.transform.location
        max_index = -1

        for i, waypoint in enumerate(self._waypoint_buffer):
            location = waypoint.transform.location
            if calculate_distance(
                    location, vehicle_location) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self.waypoints_queue:
                    self._waypoint_buffer.append(
                        self.waypoints_queue.popleft())
                else:
                    break
        self.target_waypoint = self._waypoint_buffer[0]
        
        self.target_road_option = self.target_waypoint.road_option
        
        debug = False
        control = self._pid_controller.run_step(self._target_speed, self.target_waypoint,debug=debug)
        
        return control