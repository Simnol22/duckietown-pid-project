from typing import Tuple

import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    """
    Args:
        ticks: Current tick count from the encoders.
        prev_ticks: Previous tick count from the encoders.
        resolution: Number of ticks per full wheel rotation returned by the encoder.
    Return:
        dphi: Rotation of the wheel in radians.
        ticks: current number of ticks.
    """

    
    dticks = ticks-prev_ticks
    wheel_rot_per_tick = 2 * np.pi / resolution
    dphi = dticks* wheel_rot_per_tick

    return dphi, ticks


def pose_estimation(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:

    """
    Calculate the current Duckiebot pose using the dead-reckoning model.

    Args:
        R:                  radius of wheel (both wheels are assumed to have the same size) - this is fixed in simulation,
                            and will be imported from your saved calibration for the real robot
        baseline:           distance from wheel to wheel; 2L of the theory
        x_prev:             previous x estimate - assume given
        y_prev:             previous y estimate - assume given
        theta_prev:         previous orientation estimate - assume given
        delta_phi_left:     left wheel rotation (rad)
        delta_phi_right:    right wheel rotation (rad)

    Return:
        x_curr:                  estimated x coordinate
        y_curr:                  estimated y coordinate
        theta_curr:              estimated heading
    """

    d_left = R*delta_phi_left
    d_right = R*delta_phi_right

    distance_traveled = (d_left + d_right)/2 
    
    robot_rot = (d_right-d_left)/baseline # [radians]
    L = baseline/2
   
    dx = distance_traveled * np.cos(theta_prev)
    dy = distance_traveled * np.sin(theta_prev)
    dtheta = R / baseline * (delta_phi_right-delta_phi_left)
    
    x_curr = x_prev + dx
    y_curr = y_prev + dy
    theta_curr = theta_prev + dtheta
    # ---
    return x_curr, y_curr, theta_curr