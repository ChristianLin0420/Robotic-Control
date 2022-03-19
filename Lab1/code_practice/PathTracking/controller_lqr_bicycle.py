import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBicycle(Controller):
    def __init__(self, Q=np.eye(4), R=np.eye(1)):
        self.path = None
        self.Q = Q
        self.Q[0,0] = 1
        self.Q[1,1] = 1
        self.Q[2,2] = 1
        self.Q[3,3] = 1
        self.R = R*5000
        self.pe = 0
        self.pth_e = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0

    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01): # Discrete-time Algebra Riccati Equation (DARE)
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v, l, dt]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, delta, v, l, dt = info["x"], info["y"], info["yaw"], info["delta"], info["v"], info["l"], info["dt"]
        yaw = utils.angle_norm(yaw)
        
        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        target = self.path[min_idx]
        target[2] = utils.angle_norm(target[2])

        def normalize(rad):
            return (rad + np.pi) % (2 * np.pi) - np.pi
        
        # TODO: LQR Control for Bicycle Kinematic Model
        # yaw = normalize(yaw)
        print("yaw: {}".format(yaw))
        front_x = x + l * np.cos(np.deg2rad(yaw))
        front_y = y + l * np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta))

        theta_p = target[2]
        theta_e = (theta_p - yaw) % 360
        
        if theta_e > 180:
            theta_e -= 360
        
        e = [front_x - target[0], front_y - target[1]]
        p = [np.cos(np.deg2rad(theta_e + 90)), np.sin(np.deg2rad(theta_e + 90))]
        error = np.dot(e, p)
        e_dot = vf * np.sin(np.deg2rad(delta - theta_e))
        theta_dot = v * np.tan(np.deg2rad(delta)) / l

        A = np.array([[1, dt, 0, 0], [0, 0, v, 0], [0, 0, 1, dt], [0, 0, 0, 0]])
        B = np.array([[0], [0], [0], [v / info["l"]]])
        x = np.array([[error], [e_dot], [yaw], [theta_dot]])

        P = self._solve_DARE(A, B, self.Q, self.R)
        tmp = -np.linalg.inv(self.R + B.T @ P @ B)
        next_delta = np.rad2deg(tmp @ B.T @ P @ A @ x)

        print("next_delta: {}".format(next_delta))

        return next_delta, target
