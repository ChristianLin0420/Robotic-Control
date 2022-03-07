import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBicycle(Controller):
    def __init__(self, kp=0.4, ki=0.0001, kd=0.5):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
    
    def set_path(self, path):
        super().set_path(path)
        self.acc_ep = 0
        self.last_ep = 0
    
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State
        x, y, dt = info["x"], info["y"], info["dt"]

        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        target = self.path[min_idx]
        
        # TODO: PID Control for Bicycle Kinematic Model
        ang = np.arctan2(self.path[min_idx, 1] - y, self.path[min_idx, 0] - x)
        ep = min_dist * np.sin(ang)
        self.acc_ep += dt * ep
        diff_ep = (ep - self.last_ep) / dt
        next_delta = self.kp * ep + self.ki * self.acc_ep + self.kd * diff_ep
        self.last_ep = ep
        return next_delta, target
