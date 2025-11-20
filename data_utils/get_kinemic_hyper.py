import json
import os
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize


def load_json(file_path: str) -> dict:
    """Load a single JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) in radians to a 3x3 rotation matrix.
    Rotation order: X → Y → Z (intrinsic).
    """
    roll, pitch, yaw = euler_angles
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx  # Z-Y-X order


def fit_velocity_scaling_factor(
    velocities: np.ndarray,
    displacements: np.ndarray,
    dt: float = 0.1
) -> np.ndarray:
    """
    Fit linear combination coefficients [α, β] such that:
        ||displacement|| ≈ α * ||v_t|| + β * ||v_{t+1}||
    with constraint α + β = 1
    """
    v_norm_prev = velocities[:-1]          # ||v_t||
    v_norm_next = velocities[1:]           # ||v_{t+1}||
    X = np.column_stack([v_norm_prev, v_norm_next])
    y = displacements / dt                 # approximate velocity from position difference

    def loss(theta: np.ndarray) -> float:
        pred = X @ theta
        return np.sum((y - pred) ** 2)

    constraints = {'type': 'eq', 'fun': lambda theta: theta.sum() - 1}
    result = minimize(loss, x0=[0.5, 0.5], args=(), constraints=constraints, method='SLSQP')
    return result.x


def main():
    scene_id = '000'
    root_path = 'path_to_your_waymo_data/waymo_train'
    num_frames = 191
    dt = 0.1                
    f_len = 0.8             # assumed f_len length (m), common for passenger cars
    
    # ========================== Load all ego poses ==========================
    ego_states = []
    for frame_idx in range(num_frames):
        json_path = os.path.join(root_path, scene_id, str(frame_idx), 'ego_pose.json')
        data = load_json(json_path)

        state = np.zeros(11)
        state[:3] = data['loc']             
        state[3:6] = data['rot']             
        state[6] = data['omega']             
        state[7:9] = data['velocity_xy']     
        state[9:11] = data['acceleration_xy']
        ego_states.append(state)

    ego_states = np.array(ego_states)  

    # ========================== Extract basic signals ==========================
    position_xy = ego_states[:, :2]                  
    velocity_xy = ego_states[:, 7:9]                   
    velocity_norm = np.linalg.norm(velocity_xy, axis=1) 

    local_displacements = []
    yaw_rates = []  # d(yaw)/dt

    for i in range(num_frames - 1):
        R_world_to_vehicle = np.linalg.inv(euler_to_rotation_matrix(ego_states[i, 3:6]))
        delta_pos_world = ego_states[i + 1, :3] - ego_states[i, :3]
        delta_pos_local = R_world_to_vehicle @ delta_pos_world
        local_displacements.append(delta_pos_local[:2])  # only x,y

        yaw_diff = ego_states[i + 1, 5] - ego_states[i, 5]
        yaw_rates.append(yaw_diff / dt)

    local_displacements = np.array(local_displacements)     
    yaw_rates = np.array(yaw_rates)                        

    # ========================== Fit u1: velocity scaling factor ==========================
    displacement_norm = np.linalg.norm(local_displacements, axis=1)
    u1_coeffs = fit_velocity_scaling_factor(velocity_norm, displacement_norm, dt)
    print(f"u1 coefficients [α_prev, α_next] (α + β = 1): {u1_coeffs}")

    # ========================== Fit u2: slip angle related parameter ==========================
    with np.errstate(divide='ignore', invalid='ignore'):
        slip_angles_from_pos = np.arctan2(local_displacements[:, 1], local_displacements[:, 0] + 1e-12)

    v_next = velocity_norm[1:] + 1e-8
    theoretical_slip = np.arcsin(f_len * yaw_rates / v_next)

    # Filter out invalid/nan values (e.g., near-zero velocity)
    valid_mask = ~np.isnan(theoretical_slip) & ~np.isinf(theoretical_slip)

    # Least-squares: slip_from_pos = u2 * theoretical_slip
    u2 = np.dot(slip_angles_from_pos[valid_mask], theoretical_slip[valid_mask]) \
            / np.linalg.norm(theoretical_slip[valid_mask]) ** 2

    print(f"u1 = {u1_coeffs}")
    print(f"u2 (slip angle scaling) = {u2:.6f}")


if __name__ == "__main__":
    main()