import sys
import numpy as np
import cv2

sys.path.append("..")
from Simulation.simulator import Simulator
import Simulation.utils as utils
from Simulation.utils import State, ControlState
from Simulation.kinematic_basic import KinematicModelBasic as KinematicModel

#wheeled mobile robotics
class SimulatorBasic(Simulator):
    def __init__(self,
            v_range = 20,
            w_range = 45,
            # Wheel Distance
            l = 14,
            # Wheel Size
            wu = 10,
            wv = 4,
            # Car Size
            car_w = 24,
            car_f = 20,
            car_r = 10,
            dt = 0.1
        ):
        self.control_type = "basic"
        # Control Constrain
        self.v_range = v_range
        self.w_range = w_range
        # Wheel Distance
        self.l = l
        # Wheel size
        self.wu = wu
        self.wv = wv
        # Car size
        self.car_w = car_w
        self.car_f = car_f
        self.car_r = car_r
        # Simulation delta time
        self.dt = dt
        self.model = KinematicModel(dt)

        # Initialize State
        self.state = State()
        self.cstate = ControlState(self.control_type, 0.0, 0.0)
        self.car_box = utils.compute_car_box(self.car_w, self.car_f, self.car_r, self.state.pose())
    
    def init_pose(self, pose):
        self.state.update(pose[0], pose[1], pose[2])
        self.cstate = ControlState(self.control_type, 0.0, 0.0)
        self.car_box = utils.compute_car_box(self.car_w, self.car_f, self.car_r, self.state.pose())
        self.record = []
        return self.state, {}

    def step(self, command, update_state=True):
        if command is not None:
            # Check Control Command
            self.cstate.v = command.v if command.v is not None else self.cstate.v 
            self.cstate.w = command.w if command.w is not None else self.cstate.w

        # Control Constrain
        if self.cstate.v > self.v_range:
            self.cstate.v = self.v_range
        elif self.cstate.v < -self.v_range:
            self.cstate.v = -self.v_range
        if self.cstate.w > self.w_range:
            self.cstate.w = self.w_range
        elif self.cstate.w < -self.w_range:
            self.cstate.w = -self.w_range

        # Motion
        state_next = self.model.step(self.state, self.cstate)
        if update_state:
            self.state = state_next
            self.record.append((self.state.x, self.state.y, self.state.yaw))
            self.car_box = utils.compute_car_box(self.car_w, self.car_f, self.car_r, self.state.pose())
        return state_next, {}
    
    def __str__(self):
        return self.state.__str__() + " " + self.cstate.__str__()

    def render(self, img=None):
        if img is None:
            img = np.ones((600,600,3))
        ########## Draw History ##########
        rec_max = 1000
        start = 0 if len(self.record)<rec_max else len(self.record)-rec_max
        # Draw Trajectory
        color = (0/255,97/255,255/255)
        for i in range(start,len(self.record)-1):
            cv2.line(img,(int(self.record[i][0]),int(self.record[i][1])), (int(self.record[i+1][0]),int(self.record[i+1][1])), color, 1)

        ########## Draw Car ##########
        # Car box
        pts1, pts2, pts3, pts4 = self.car_box
        color = (0,0,0)
        size = 1
        cv2.line(img, tuple(pts1.astype(int).tolist()), tuple(pts2.astype(int).tolist()), color, size)
        cv2.line(img, tuple(pts1.astype(int).tolist()), tuple(pts3.astype(int).tolist()), color, size)
        cv2.line(img, tuple(pts3.astype(int).tolist()), tuple(pts4.astype(int).tolist()), color, size)
        cv2.line(img, tuple(pts2.astype(int).tolist()), tuple(pts4.astype(int).tolist()), color, size)
        # Car center & direction
        t1 = utils.rot_pos( 6, 0, -self.state.yaw) + np.array((self.state.x,self.state.y))
        t2 = utils.rot_pos( 0, 4, -self.state.yaw) + np.array((self.state.x,self.state.y))
        t3 = utils.rot_pos( 0, -4, -self.state.yaw) + np.array((self.state.x,self.state.y))
        cv2.line(img, (int(self.state.x),int(self.state.y)), (int(t1[0]), int(t1[1])), (0,0,1), 2)
        cv2.line(img, (int(t2[0]), int(t2[1])), (int(t3[0]), int(t3[1])), (1,0,0), 2)
        
        ########## Draw Wheels ##########
        w1 = utils.rot_pos( 0, self.l, -self.state.yaw) + np.array((self.state.x,self.state.y))
        w2 = utils.rot_pos( 0,-self.l, -self.state.yaw) + np.array((self.state.x,self.state.y))
        # 4 Wheels
        img = utils.draw_rectangle(img,int(w1[0]),int(w1[1]),self.wu,self.wv,-self.state.yaw)
        img = utils.draw_rectangle(img,int(w2[0]),int(w2[1]),self.wu,self.wv,-self.state.yaw)
        # Axle
        img = cv2.line(img, tuple(w1.astype(int).tolist()), tuple(w2.astype(int).tolist()), (0,0,0), 1)
        return img
