#!/usr/bin/env python3
"""
Static Behavior Agent for SceneCrafter Simulation Framework
"""

import re
import random
import numpy as np
from sklearn import neighbors
from agents.navigation.controller import Control
from agents.navigation.local_planner_behavior import LocalPlanner, RoadOption
from agents.navigation.types_behavior import Cautious, Aggressive, Normal, ExtremeAggressive
import sys
from .waypoint import Waypoint
from .tools.misc import is_within_distance, calculate_distance, positive, calculate_rotation
import copy


class BehaviorAgent():
    """
    Static Behavior Agent for representing stationary vehicles in simulation.
    
    This agent class implements a minimal behavior model for vehicles that
    remain stationary throughout the simulation. It provides a consistent
    interface with dynamic behavior agents while maintaining zero velocity
    and fixed positioning.
    
    Key Characteristics:
    - Zero velocity maintenance: Always reports 0.0 km/h speed
    - Static positioning: Never moves from initial spawn location
    - Emergency stop behavior: Always returns full braking commands
    - Minimal computational load: No path planning or navigation calculations
    - Traffic integration: Serves as static obstacle in traffic simulations
    
    Typical Use Cases:
    - Parked vehicles in urban scenarios
    - Traffic obstacles for testing dynamic agents
    - Static environmental elements in scene generation
    - Baseline measurements for traffic flow analysis
    """
    
    def __init__(self, config):
        """
        Initialize static behavior agent with configuration parameters.
        
        This constructor sets up a static agent with fixed properties based
        on the provided configuration. The agent will maintain its initial
        position throughout the simulation lifecycle.
        
        Args:
            config (dict): Configuration dictionary containing:
                - name (str): Unique identifier for the agent
                - static_waypoint (Waypoint): Initial spawn location and orientation
                - vehicle_type (str): Type of vehicle (car, truck, etc.)
                - vehicle_bbox (list): Vehicle dimensions [length, width, height]
                
        Configuration Example:
            config = {
                'name': 'parked_sedan_01',
                'static_waypoint': Waypoint(x=100.5, y=200.3, yaw=1.57),
                'vehicle_type': 'sedan',
                'vehicle_bbox': [4.2, 1.8, 1.4]
            }
        """
        self.vehicle_name = config['name']
        self.look_ahead_steps = 0  # Static agents don't plan ahead
        self.end_route_flag = False  # Always False for static agents
        
        # Static positioning information
        self.cur_waypoint = config['static_waypoint']  # Current (fixed) position
        self.last_waypoint = config['static_waypoint']  # Previous (also fixed)
        
        # Vehicle state (always static)
        self.speed = 0  # km/h - always 0 for static agents
        self.velocity_xy = np.array([0, 0])  # km/h - always zero vector
        self.speed_limit = 0  # Not applicable for static agents
        self.min_speed = 0  # Always 0
        self.max_speed = 0  # Always 0
        
        # Vehicle properties
        self.vehicle_type = config['vehicle_type']
        self.bounding_box = config['vehicle_bbox']  # [length, width, height]
        
        # Behavior flags (always False for static agents)
        self.if_overtake = False  # No overtaking capability
        self.if_tailgate = False  # No tailgating behavior
        self.if_static = True  # Always True - identifies as static agent
        
        # Control state (always emergency stop)
        self.cur_control = Control()

    def get_next_lane_id(self):
        """
        Return the current lane ID for static positioning.
        
        Since static agents never move or change lanes, this method simply
        returns the lane ID of the current (fixed) waypoint. This provides
        consistency with dynamic agent interfaces.
        
        Returns:
            str: Lane ID of the static agent's current position
            
        Example:
            >>> lane_id = static_agent.get_next_lane_id()
            >>> print(f"Static vehicle is in lane: {lane_id}")
        """
        return self.cur_waypoint.lane_id

    def get_speed(self):
        """
        Return the current speed of the static vehicle.
        
        Static agents always maintain zero velocity, so this method
        consistently returns 0.0 km/h. This provides interface consistency
        with dynamic behavior agents while reflecting the static nature.
        
        Returns:
            float: Always 0.0 km/h for static agents
            
        Example:
            >>> speed = static_agent.get_speed()
            >>> assert speed == 0.0  # Always true for static agents
        """
        return 0.0

    def emergency_stop(self):
        """
        Generate emergency braking control for static positioning.
        
        This method creates a control command that applies maximum braking
        force with zero throttle and steering. For static agents, this
        effectively maintains the stationary position and serves as the
        default control output.
        
        Control Command Structure:
        - steer: 0.0 (neutral steering)
        - throttle: 0.0 (no acceleration)
        - brake: 1.0 (maximum braking force)
        - hand_brake: False (controlled braking)
        
        Returns:
            Control: Emergency braking control command for static positioning
            
        Example:
            >>> control = static_agent.emergency_stop()
            >>> print(f"Brake force: {control.brake}")  # Always 1.0
        """
        control = Control()
        control.steer = 0.0      # Neutral steering for stability
        control.throttle = 0.0   # Zero throttle (no acceleration)
        control.brake = 1.0      # Full brake force (maintains static position)
        control.hand_brake = False  # Use service brakes, not handbrake
        
        # Update internal control state
        self.cur_control = control
        return control

    def run_step(self, debug=False):
        """
        Execute one simulation step for static agent behavior.
        
        This method represents the main control loop for static agents.
        Since the agent remains stationary, it always returns an emergency
        stop command. The debug parameter is accepted for interface consistency
        but has no effect on static agent behavior.
        
        Args:
            debug (bool, optional): Debug flag for consistency with dynamic agents.
                                   Has no effect on static agent behavior.
                                   Defaults to False.
        
        Returns:
            Control: Emergency braking control command (always the same for static agents)
            
        Behavior Notes:
        - Always returns emergency stop command
        - No navigation or path planning performed
        - Position remains fixed throughout simulation
        - Debug parameter accepted but ignored
        
        Example:
            >>> for step in range(100):
            ...     control = static_agent.run_step(debug=True)
            ...     # control will always be emergency stop
        """
        return self.emergency_stop()