#!/usr/bin/env python3
""" This module contains the different parameters sets for each behavior. """

scenario_trigger_flag = False
ignore_static_flag = False


class Cautious(object):
    max_speed = 20
    speed_lim_dist = 10
    speed_decrease = 12
    safety_time = 4
    min_proximity_threshold = 15
    braking_distance = 9
    overtake_counter = -1
    tailgate_counter = -1
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag

class Normal(object):
    max_speed = 25
    speed_lim_dist = 10
    speed_decrease = 10
    safety_time = 3
    min_proximity_threshold = 13
    braking_distance = 8
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag

class Aggressive(object):
    max_speed = 30
    speed_lim_dist = 10
    speed_decrease = 8
    safety_time = 2
    min_proximity_threshold = 11
    braking_distance = 7
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag



class ExtremeAggressive(object):
    max_speed = 35
    speed_lim_dist = 1
    speed_decrease = 1
    safety_time = 1
    min_proximity_threshold = 7
    braking_distance = 5
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag


class Cautious_fast(object):
    max_speed = 25
    speed_lim_dist = 10
    speed_decrease = 12
    safety_time = 4
    min_proximity_threshold = 12
    braking_distance = 4
    overtake_counter = -1
    tailgate_counter = -1
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag


class Normal_fast(object):
    max_speed = 30
    speed_lim_dist = 8
    speed_decrease = 10
    safety_time = 4
    min_proximity_threshold = 10
    braking_distance = 2
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag




class Aggressive_fast(object):
    max_speed = 35
    speed_lim_dist = 5
    speed_decrease = 8
    safety_time = 4
    min_proximity_threshold = 8
    braking_distance = 1
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag



class ExtremeAggressive_fast(object):
    max_speed = 40
    speed_lim_dist = 1
    speed_decrease = 1
    safety_time = 1
    min_proximity_threshold = 17
    braking_distance = 1
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag


class Cautious_highway(object):
    max_speed = 60
    speed_lim_dist = 10
    speed_decrease = 12
    safety_time = 4
    min_proximity_threshold = 17
    braking_distance = 14
    overtake_counter = -1
    tailgate_counter = -1
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag


class Normal_highway(object):
    max_speed = 55
    speed_lim_dist = 10
    speed_decrease = 10
    safety_time = 4
    min_proximity_threshold = 17
    braking_distance = 12
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag




class Aggressive_highway(object):
    max_speed = 65
    speed_lim_dist = 10
    speed_decrease = 8
    safety_time = 4
    min_proximity_threshold = 17
    braking_distance = 10
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag



class ExtremeAggressive_highway(object):
    max_speed = 90
    speed_lim_dist = 1
    speed_decrease = 1
    safety_time = 1
    min_proximity_threshold = 17
    braking_distance = 8
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag