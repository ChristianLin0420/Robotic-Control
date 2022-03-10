import sys
import numpy as np
import cv2

sys.path.append("..")
from Simulation.simulator_basic import SimulatorBasic 
from Simulation.simulator_differential_drive import SimulatorDifferentialDrive
from Simulation.simulator_bicycle import SimulatorBicycle
from Simulation.utils import Bresenham, compute_car_box, EndPoint, ControlState
from Simulation.sensor_lidar import LidarModel

class SimulatorMap(SimulatorBasic, SimulatorDifferentialDrive, SimulatorBicycle):
    def __init__(self, simulator_class, m, **kargs):
        simulator_class.__init__(self, **kargs)
        self.simulator_class = simulator_class
        self.m = m

    def init_pose(self, pose):
        state, info = self.simulator_class.init_pose(self, pose)
        collision = self.collision_detect(self.m, self.car_box)
        info["collision"] = collision
        return state, info

    def collision_detect(self, m, car_box):
        p1,p2,p3,p4 = car_box
        l1 = Bresenham(p1[0], p2[0], p1[1], p2[1])
        l2 = Bresenham(p2[0], p3[0], p2[1], p3[1])
        l3 = Bresenham(p3[0], p4[0], p3[1], p4[1])
        l4 = Bresenham(p4[0], p1[0], p4[1], p1[1])
        check = l1+l2+l3+l4
        collision = False
        for pts in check:
            for i in range(-1,2):
                for j in range(-1,2):
                    if m[int(pts[1]+i),int(pts[0])+j]<0.5:
                        collision = True
                        break
        return collision
        
    def step(self, command):
        state_next, _ = self.simulator_class.step(self, command, update_state=False)
        car_box_next = compute_car_box(self.car_w, self.car_f, self.car_r, state_next.pose())
        collision = self.collision_detect(self.m, car_box_next)
        if collision:
            self.state.w = 0.0
            self.state.v = -0.5*self.state.v
            state_next = self.simulator_class.step(self, ControlState(self.control_type, 0, 0))
        else:
            self.state = state_next
            self.record.append((self.state.x, self.state.y, self.state.yaw))
            self.car_box = compute_car_box(self.car_w, self.car_f, self.car_r, self.state.pose())
        return self.state, {"collision":collision}

    def render(self):
        img = np.repeat(self.m[...,np.newaxis],3,2)
        img = self.simulator_class.render(self, img)
        return img

class SimulatorMapLidar(SimulatorMap):
    def __init__(self, simulator_class, m, lidar_params=[31,-120.0,120.0,250], **kargs):
        SimulatorMap.__init__(self, simulator_class, m, **kargs)
        self.simulator_class = simulator_class
        self.lidar_param = lidar_params
        self.lidar = LidarModel(*lidar_params)
        self.sense_data = self.lidar.measure(self.m, self.state.pose())
    
    def init_pose(self, pose):
        state, info = SimulatorMap.init_pose(self, pose)
        self.sense_data = self.lidar.measure(self.m, self.state.pose())
        info["lidar"] = self.sense_data
        return state, info

    def step(self, command):
        state_next, info = SimulatorMap.step(self, command)
        self.sense_data = self.lidar.measure(self.m, self.state.pose())
        info["lidar"] = self.sense_data
        return state_next, info
    
    def render(self):
        img = np.repeat(self.m[...,np.newaxis],3,2)
        # Draw rays
        pose = self.state.pose()
        plist = EndPoint(pose, self.lidar_param, self.sense_data)
        for i, pts in enumerate(plist):
            if self.sense_data[i] < self.lidar_param[3]:
                color = (0.0,0.9,0.0)
            else:
                color = (0.7,1.0,0.7)
            cv2.line(
            img, 
            (int(1*pose[0]), int(1*pose[1])), 
            (int(1*pts[0]), int(1*pts[1])),
            color, 1)
        img = self.simulator_class.render(self, img)
        return img
        