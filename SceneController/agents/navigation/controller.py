#!/usr/bin/env python3
"""
Vehicle Control System using PID Controllers
"""

from collections import deque
import math
import numpy as np
from SceneController.scene import Scene

class Control():
    """
    Vehicle Control Command Data Structure
    
    This class encapsulates all control commands that can be sent to a vehicle
    including throttle, steering, braking, and auxiliary controls.
    
    Attributes:
        throttle (float): Throttle control value [0.0, 1.0]
        steer (float): Steering control value [-1.0, 1.0] where -1 is full left, 1 is full right
        brake (float): Brake control value [0.0, 1.0]
        hand_brake (bool): Hand brake engagement (not used in autonomous mode)
        manual_gear_shift (bool): Manual gear shift control (not used in autonomous mode)
    """

    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, manual_gear_shift=False):
        """
        Initialize a new control command with specified parameters.
        
        Args:
            throttle (float): Initial throttle value (0.0-1.0)
            steer (float): Initial steering value (-1.0 to 1.0)
            brake (float): Initial brake value (0.0-1.0)
            hand_brake (bool): Hand brake state
            manual_gear_shift (bool): Manual gear shift state
        """
        self.throttle = throttle
        self.steer = steer
        self.brake = brake
        # not used
        self.hand_brake = hand_brake
        self.manual_gear_shift = manual_gear_shift

    def __str__(self):
        """
        Generate a human-readable string representation of the control command.
        
        Returns:
            str: Formatted string showing steering, throttle, and brake values
        """
        return 'steer: {:.2f}, throttle: {:.2f}, brake: {:.2f}'.format(self.steer, self.throttle, self.brake)

class VehiclePIDController():
    """
    Main Vehicle PID Controller for Autonomous Driving
    
    This class combines both lateral (steering) and longitudinal (speed) PID controllers
    to provide comprehensive low-level vehicle control from the client side. It coordinates
    the interaction between speed control and trajectory following.
    
    The controller implements safety limits and smoothing to ensure stable operation:
    - Steering rate limits to prevent abrupt changes
    - Throttle/brake saturation limits
    - Manual override prevention
    
    Attributes:
        max_brake (float): Maximum brake force [0.0, 1.0]
        max_throt (float): Maximum throttle [0.0, 1.0]
        max_steer (float): Maximum steering angle [-1.0, 1.0]
        _vehicle_name (str): Name identifier for the controlled vehicle
        _lon_controller (PIDLongitudinalController): Speed control PID
        _lat_controller (PIDLateralController): Steering control PID
    """

    def __init__(self, vehicle_name, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=1.0,
                 max_steering=0.8):
        """
        Initialize the main vehicle PID controller with specified parameters.
        
        Args:
            vehicle_name (str): Unique identifier for the vehicle to control
            args_lateral (dict): Configuration parameters for lateral PID controller
                Required keys: K_P (proportional), K_D (differential), K_I (integral)
            args_longitudinal (dict): Configuration parameters for longitudinal PID controller
                Required keys: K_P (proportional), K_D (differential), K_I (integral)
            offset (float, optional): Lateral offset from center line. Positive = right, negative = left
                Default: 0.0 (center line following)
            max_throttle (float, optional): Maximum throttle limit [0.0, 1.0]. Default: 0.75
            max_brake (float, optional): Maximum brake limit [0.0, 1.0]. Default: 1.0
            max_steering (float, optional): Maximum steering limit [0.0, 1.0]. Default: 0.8
                
        Note:
            Large offset values may cause the vehicle to drive in adjacent lanes
            or off-road, potentially breaking the controller stability.
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle_name = vehicle_name
        self.past_steering = Scene.get_vehicle_control(self._vehicle_name).steer
        self._lon_controller = PIDLongitudinalController(self._vehicle_name, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle_name, offset, **args_lateral)

    def run_step(self, target_speed, waypoint, debug=False):
        """
        Execute one complete control step by coordinating both lateral and longitudinal PID controllers.
        
        This method calculates the optimal control commands to reach the specified waypoint
        at the desired target speed. It combines speed control and trajectory following
        while applying safety limits and smoothing.
        
        Args:
            target_speed (float): Desired vehicle speed in km/h
            waypoint: Target location encoded as a waypoint object with transform information
            debug (bool, optional): Enable debug output. Default: False
            
        Returns:
            Control: Complete control command with throttle, brake, and steering values
            
        Process:
            1. Calculate longitudinal control (acceleration/braking) for speed matching
            2. Calculate lateral control (steering) for trajectory following
            3. Apply safety limits and rate constraints
            4. Return integrated control command
            
        Safety Features:
            - Steering rate limiting (max 0.2 rad/step)
            - Control saturation limits
            - Smooth control transitions
        """

        # Calculate longitudinal control (speed matching)
        acceleration = self._lon_controller.run_step(target_speed)
        if debug:
            print(f'Longitudinal acceleration: {acceleration:.3f}')
            
        # Calculate lateral control (trajectory following)
        current_steering = self._lat_controller.run_step(waypoint, debug=debug)
        
        # Create new control command
        control = Control()
        
        # Convert acceleration to throttle/brake commands
        # Positive acceleration -> throttle, negative -> brake
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Apply steering rate limiting for smooth operation
        # Prevents abrupt steering changes that could destabilize the vehicle
        steering_change_limit = 0.2  # Maximum steering change per step (radians)
        if current_steering > self.past_steering + steering_change_limit:
            current_steering = self.past_steering + steering_change_limit
        elif current_steering < self.past_steering - steering_change_limit:
            current_steering = self.past_steering - steering_change_limit

        # Apply absolute steering limits
        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        # Finalize control command
        control.steer = steering
        control.hand_brake = False  # Hand brake not used in autonomous mode
        control.manual_gear_shift = False  # Automatic transmission assumed
        
        # Update state for next iteration
        self.past_steering = steering

        return control


class PIDLongitudinalController():
    """
    PID Controller for Longitudinal (Speed) Control
    
    This class implements a Proportional-Integral-Derivative controller specifically
    designed for longitudinal vehicle control. It maintains vehicle speed by
    calculating appropriate acceleration/braking commands based on speed error.
    
    The controller uses the following PID equation:
    output = K_P * error + K_D * derivative_error + K_I * integral_error
    
    Attributes:
        _vehicle_name (str): Vehicle identifier for speed queries
        _k_p (float): Proportional gain - immediate error response
        _k_d (float): Derivative gain - damping oscillations
        _k_i (float): Integral gain - eliminating steady-state error
        _dt (float): Time step for discrete control [seconds]
        _error_buffer (deque): Circular buffer for error history (max 10 samples)
    """

    def __init__(self, vehicle_name, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Initialize the longitudinal PID controller with specified parameters.
        
        Args:
            vehicle_name (str): Unique identifier for the controlled vehicle
            K_P (float, optional): Proportional gain. Higher values increase response speed
                but may cause oscillation. Default: 1.0
            K_D (float, optional): Derivative gain. Provides damping to reduce overshoot
                and oscillation. Default: 0.0
            K_I (float, optional): Integral gain. Eliminates steady-state error but may
                cause windup if too high. Default: 0.0
            dt (float, optional): Control loop time step in seconds. Default: 0.03 (30ms)
                
        Note:
            Typical PID tuning process:
            1. Start with K_P only, increase until acceptable response
            2. Add K_D to reduce oscillation if needed
            3. Add K_I to eliminate steady-state error if present
        """
        self._vehicle_name = vehicle_name
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to maintain target speed.
        
        This method queries the current vehicle speed and calculates the appropriate
        acceleration/braking command using PID control to match the target speed.
        
        Args:
            target_speed (float): Desired speed in km/h
            debug (bool, optional): Enable debug output showing current speed
            
        Returns:
            float: Control output in range [-1.0, 1.0] where:
                Positive values: acceleration (throttle)
                Negative values: deceleration (brake)
                Zero: maintain current speed
        """
        # Get current vehicle speed from the simulation
        current_speed = Scene.get_vehicle_speed(self._vehicle_name)
        
        if debug:
            print(f'Current speed: {current_speed:.2f} km/h, Target: {target_speed:.2f} km/h')
            
        # Calculate PID control output
        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Calculate PID control output for speed regulation.
        
        Implements the discrete PID control equation to compute acceleration/braking
        commands based on speed error. The PID controller combines proportional,
        integral, and derivative terms for optimal control response.
        
        PID Equation:
        output = K_P * error + K_D * (d_error/dt) + K_I * integral(error*dt)
        
        Args:
            target_speed (float): Desired speed in km/h
            current_speed (float): Current vehicle speed in km/h
            
        Returns:
            float: Normalized control output [-1.0, 1.0]
                -1.0: Maximum braking
                0.0: No action (maintain speed)
                1.0: Maximum acceleration
        """
        # Calculate speed error
        error = target_speed - current_speed
        self._error_buffer.append(error)

        # Calculate derivative and integral terms if sufficient data available
        if len(self._error_buffer) >= 2:
            # Derivative term: rate of error change
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            # Integral term: accumulated error over time
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0  # No derivative available with single sample
            _ie = 0.0  # No integral accumulation yet

        # Calculate PID output and clamp to valid range
        pid_output = (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie)
        
        # Ensure output is within actuator limits [-1.0, 1.0]
        return np.clip(pid_output, -1.0, 1.0)

class PIDLateralController():
    """
    PID Controller for Lateral (Steering) Control
    
    This class implements a Proportional-Integral-Derivative controller specifically
    designed for lateral vehicle control. It calculates steering commands to maintain
    the vehicle on a desired trajectory by minimizing the heading error to target waypoints.
    
    The controller uses geometric error calculation based on the angle between the
    vehicle's current heading vector and the vector to the target waypoint.
    
    Attributes:
        _vehicle_name (str): Vehicle identifier for position queries
        _k_p (float): Proportional gain - immediate error response
        _k_d (float): Derivative gain - damping oscillations
        _k_i (float): Integral gain - eliminating steady-state error
        _dt (float): Time step for discrete control [seconds]
        _offset (float): Lateral offset from center line [meters]
        _e_buffer (deque): Circular buffer for error history (max 10 samples)
    """

    def __init__(self, vehicle_name, offset=0, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Initialize the lateral PID controller with specified parameters.
        
        Args:
            vehicle_name (str): Unique identifier for the vehicle to control
            offset (float, optional): Lateral offset from center line [meters]. 
                Positive = right offset, negative = left offset. Default: 0.0
            K_P (float, optional): Proportional gain. Controls immediate response to heading error.
                Higher values increase responsiveness but may cause oscillation. Default: 1.0
            K_D (float, optional): Derivative gain. Provides damping to reduce overshoot
                and oscillation. Default: 0.0
            K_I (float, optional): Integral gain. Eliminates steady-state tracking error.
                Use cautiously as it may cause instability. Default: 0.0
            dt (float, optional): Control loop time step in seconds. Default: 0.03 (30ms)
            
        Warning:
            Large offset values may cause the vehicle to drive in adjacent lanes
            or off-road, potentially leading to unstable control behavior.
        """
        self._vehicle_name = vehicle_name
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint, debug=False):
        """
        Execute one step of lateral control to steer the vehicle toward a target waypoint.
        
        This method queries the current vehicle position and calculates the appropriate
        steering command to minimize the heading error to the target waypoint.
        
        Args:
            waypoint: Target waypoint object with transform information (location and orientation)
            debug (bool, optional): Enable debug output for error calculation
            
        Returns:
            float: Steering control value in range [-1.0, 1.0] where:
                -1.0: Maximum steering to the left
                0.0: Straight ahead (no steering)
                1.0: Maximum steering to the right
        """
        # Get current vehicle waypoint and transform
        vehicle_waypoint = Scene.get_vehicle_waypoint(self._vehicle_name)
        vehicle_transform = vehicle_waypoint.transform
        
        # Calculate steering command using PID control
        return self._pid_control(waypoint, vehicle_transform, debug=debug)

    def _pid_control(self, waypoint, vehicle_transform, debug=False):
        """
        Calculate PID steering control based on heading error to target waypoint.
        
        This method computes the steering angle required to align the vehicle's
        heading with the direction vector to the target waypoint. It uses geometric
        calculations to determine the heading error and applies PID control to
        generate smooth steering commands.
        
        The algorithm:
        1. Calculate vehicle's current heading vector
        2. Calculate vector from vehicle to target waypoint
        3. Compute angle between these vectors (heading error)
        4. Apply PID control to the heading error
        5. Return normalized steering command
        
        Args:
            waypoint: Target waypoint with transform information
            vehicle_transform: Current vehicle transform with location and orientation
            debug (bool, optional): Enable debug output for error calculation
            
        Returns:
            float: Normalized steering command [-1.0, 1.0]
                -1.0: Maximum left steering
                0.0: Straight ahead
                1.0: Maximum right steering
        """
        # Extract current vehicle position and heading
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec[0], v_vec[1], 0.0])  # 2D heading vector

        # Calculate target position with optional lateral offset
        if self._offset != 0:
            # Apply lateral offset to waypoint position
            w_tran = waypoint.transform
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.rlocation + [self._offset * r_vec[0], self._offset * r_vec[1], 0.0]
        else:
            # Use waypoint position directly
            w_loc = waypoint.transform.location

        # Calculate vector from vehicle to target waypoint
        w_vec = np.array([w_loc[0] - ego_loc[0],
                          w_loc[1] - ego_loc[1],
                          0.0])
        
        # Calculate heading error angle using dot product
        # Add small epsilon to prevent division by zero
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 ((np.linalg.norm(w_vec) * np.linalg.norm(v_vec)) + 1e-8), -1.0, 1.0))
        
        # Debug mode: show alternative heading calculation
        if debug:
            w_vec_alt = waypoint.transform.get_forward_vector()
            w_vec_alt = np.array([w_vec_alt[0], w_vec_alt[1], 0.0])
            _dot_alt = math.acos(np.clip(np.dot(w_vec_alt, v_vec) /
                                     ((np.linalg.norm(w_vec_alt) * np.linalg.norm(v_vec)) + 1e-8), -1.0, 1.0))

        # Determine steering direction using cross product
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0  # Negative angle = left turn needed

        # Update error history for PID calculation
        self._e_buffer.append(_dot)
        
        # Calculate derivative and integral terms
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0  # No derivative available
            _ie = 0.0  # No integral accumulation

        # Calculate PID output and clamp to valid range
        pid_output = (self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie)
        return np.clip(pid_output, -1.0, 1.0)