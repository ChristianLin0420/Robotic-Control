import sys
import numpy as np

sys.path.append("..")
from Simulation.utils import Bresenham, compute_car_box
from Simulation.sensor_lidar import LidarModel

def SimulatorMap(simulator_class):
    class SimulatorMapClass(simulator_class):
        def __init__(self, m, **kargs):
            simulator_class.__init__(self, **kargs)
            self.simulator_class = simulator_class
            self.m = m

        def collision_detect(self, m, car_box):
            p1,p2,p3,p4 = car_box
            l1 = Bresenham(p1[0], p2[0], p1[1], p2[1])
            l2 = Bresenham(p2[0], p3[0], p2[1], p3[1])
            l3 = Bresenham(p3[0], p4[0], p3[1], p4[1])
            l4 = Bresenham(p4[0], p1[0], p4[1], p1[1])
            check = l1+l2+l3+l4
            collision = False
            for pts in check:
                if m[int(pts[1]),int(pts[0])]<0.5:
                    collision = True
                    break
            return collision
            
        def step(self, command):
            state_next, info = self.simulator_class.step(self, command, update_state=False)
            car_box_next = compute_car_box(self.car_w, self.car_f, self.car_r, state_next.pose())
            collision = self.collision_detect(self.m, car_box_next)
            if collision:
                self.state.v = -0.5*self.state.v
                state_next = self.simulator_class.step(self, command)
            else:
                self.state = state_next
                self.record.append((self.state.x, self.state.y, self.state.yaw))
                self.car_box = compute_car_box(self.car_w, self.car_f, self.car_r, self.state.pose())
            info["collision"] = collision
            return self.state, info

        def render(self):
            img = np.repeat(self.m[...,np.newaxis],3,2)
            img = self.simulator_class.render(self, img)
            return img

    return SimulatorMapClass

def SimulatorMapLidar(simulator_class):
    simulator_class2 = SimulatorMap(simulator_class)
    class SimulatorMapLidarClass(simulator_class2):
        def __init__(self, m, lidar_param, **kwargs):
            simulator_class2.init()

        def step(self):
            pass

        def render(self):
            pass
    
    return SimulatorMapLidarClass