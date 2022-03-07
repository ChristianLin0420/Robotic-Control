import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBicycle(Controller):
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, delta, v, l = info["x"], info["y"], info["yaw"], info["delta"], info["v"], info["l"]

        # Search Front Wheel Target
        front_x = x + l*np.cos(np.deg2rad(yaw))
        front_y = y + l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta))
        min_idx, min_dist = utils.search_nearest(self.path, (front_x,front_y))
        target = self.path[min_idx]

        # TODO: Stanley Control for Bicycle Kinematic Model
        delta = target[2]
        theta_e = (delta - yaw) % 360
        
        if theta_e > 180:
            theta_e -= 360
        
        e = [front_x - target[0], front_y - target[1]]
        p = [np.cos(np.deg2rad(theta_e + 90)), np.sin(np.deg2rad(theta_e + 90))]
        error = np.dot(e, p)

        next_delta = np.rad2deg(np.arctan2(-self.kp * error, vf)) + theta_e

        return next_delta, target
