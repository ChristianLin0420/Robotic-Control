import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBicycle(Controller):
    def __init__(self, kp=1, Lfc=25):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc

    # State: [x, y, yaw, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, v, l = info["x"], info["y"], info["yaw"], info["v"], info["l"]

        # Search Front Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        Ld = self.kp*v + self.Lfc
        target_idx = min_idx
        for i in range(min_idx,len(self.path)-1):
            dist = np.sqrt((self.path[i+1,0]-x)**2 + (self.path[i+1,1]-y)**2)
            if dist > Ld:
                target_idx = i
                break
        target = self.path[target_idx]

        # Control Algorithm
        alpha = np.arctan2(target[1]-y, target[0]-x) - np.deg2rad(yaw)
        next_delta = np.rad2deg(np.arctan2(2.0*l*np.sin(alpha)/Ld, 1))
        return next_delta, target
