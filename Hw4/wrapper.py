SIMULATOR_TYPE = "basic" # basic / diff_drive / bicycle
try:
    if SIMULATOR_TYPE == "basic":
        from Simulation.simulator_basic import SimulatorBasic as Simulator
    elif SIMULATOR_TYPE == "diff_drive":
        from Simulation.simulator_differential_drive import SimulatorDifferentialDrive as Simulator
    elif SIMULATOR_TYPE == "bicycle":
        from Simulation.simulator_bicycle import SimulatorBicycle as Simulator
    else:
        raise NameError("Unknown simulator!!")
except:
    raise

import numpy as np
import cv2
from Simulation.utils import ControlState
import PathTracking.utils
import cubic_spline

#Path tracking environment class
class PathTrackingEnv():
    #Constructor
    def __init__(self, init_range=20, max_step=400):
        self.max_step = max_step
        self.init_range = init_range
        self.simulator = Simulator(w_range=60)
        self.reset()
    
    #Reset
    def reset(self):
        self.path = self.gen_path()
        self.img_path = np.ones((600,600,3))
        for i in range(self.path.shape[0]-1):
            p1 = (int(self.path[i,0]), int(self.path[i,1]))
            p2 = (int(self.path[i+1,0]), int(self.path[i+1,1]))
            cv2.line(self.img_path, p1, p2, (1.0,0.5,0.5), 1)

        start = (
            200 + np.random.randint(-self.init_range,self.init_range),
            50 + np.random.randint(-self.init_range,self.init_range),
            90
        )
        self.simulator.init_pose(start)
        # Initial State
        position = (self.simulator.state.x, self.simulator.state.y)
        min_idx, min_dist = PathTracking.utils.search_nearest(self.path, position)
        self.target = self.path[min_idx]
        self.last_idx = min_idx
        self.n_step = 0

        record_path = np.concatenate((self.get_record_path(-1), self.get_record_path(-1)))
        future_path = self.get_future_path(min_idx)
        state = np.concatenate((record_path, future_path))
        info = {"min_idx": min_idx, "pose":self.simulator.state.pose(), "record": self.simulator.record}
        return state, info

    #Step
    def step(self, action):
        if action[0] > 1: action[0] = 1
        elif action[0] < -1: action[0] = -1

        # Action Range 
        if SIMULATOR_TYPE == "basic":
            action0 = self.simulator.v_range * 0.6
            action1 = self.simulator.w_range * action[0]
        elif SIMULATOR_TYPE == "diff_drive":
            action0 = self.simulator.lw_range * 1.0
            action1 = self.simulator.rw_range * action[0]
        elif SIMULATOR_TYPE == "bicycle":
            action0 = self.simulator.a_range * 1.0
            action1 = self.simulator.delta_range * action[0]
        else:
            raise NameError("Unknown simulator!!")

        # Controlling
        command = ControlState(SIMULATOR_TYPE, action0, action1)
        self.simulator.step(command)
        position = (self.simulator.state.x, self.simulator.state.y)
        min_idx, min_dist = PathTracking.utils.search_nearest(self.path, position)

        record_path = np.concatenate((self.get_record_path(-2), self.get_record_path(-1)))
        future_path = self.get_future_path(min_idx)
        state_next = np.concatenate((record_path, future_path))
        info = {"min_idx": min_idx, "pose":self.simulator.state.pose(), "record": self.simulator.record}
        self.n_step += 1

        # Compute Reward
        reward = 0
        self.target = self.path[min_idx]
        error_yaw = (self.target[2] - self.simulator.state.yaw) % 360
        if error_yaw > 180:
            error_yaw = 360 - error_yaw

        idx_diff = min_idx - self.last_idx
        if idx_diff > 0:
            progress_reward = 0.1
        elif idx_diff == 0:
            progress_reward = 0
        else:
            progress_reward = -1.0

        reward = 0.8 * np.exp(-0.1 * min_dist) + 0.2 * np.exp(-0.1 * error_yaw*error_yaw) + progress_reward
        self.last_idx = min_idx

        # Done
        done = False
        goal_dist = np.sqrt((position[0] - self.path[-1,0])**2 + (position[1] - self.path[-1,1])**2)
        if min_idx == len(self.path)-1 or goal_dist < 10 or self.n_step >= self.max_step:
            done = True

        return state_next, reward, done, info

    #Render
    def render(self, img=None):
        img = self.img_path.copy()
        cv2.circle(img,(int(self.target[0]),int(self.target[1])),3,(1,0.3,0.7),2) # target points
        img = self.simulator.render(img)
        img = cv2.flip(img, 0)
        return img

    #Get future path
    def get_future_path(self, start_idx, interval=16):
        future_path = []

        for i in range(4):
            idx = start_idx + i*interval
            if idx < len(self.path):
                future_path.append(self.path[idx, :2])
            else:
                 future_path.append(self.path[-1, :2])

        return np.concatenate(future_path) / 600.0

    #Get record path
    def get_record_path(self, idx):
        record_path = np.array(self.simulator.record[idx], dtype=np.float32)
        record_path[0] /= 600.0
        record_path[1] /= 600.0
        record_path[2] = np.radians(record_path[2])
        return record_path

    #Generate a path randomly
    def gen_path(self):
        path = [[200+np.random.randint(-30,30),  50+np.random.randint(-30,30)], 
                [200+np.random.randint(-30,30), 200+np.random.randint(-30,30)], 
                [150+np.random.randint(-30,30), 250+np.random.randint(-30,30)], 
                [ 50+np.random.randint(-30,30), 350+np.random.randint(-30,30)]]
        if np.random.random() > 0.5:
            path[0][0] = 400 - path[0][0]
            path[1][0] = 400 - path[1][0]
            path[2][0] = 400 - path[2][0]
            path[3][0] = 400 - path[3][0]
        path_smooth = cubic_spline.cubic_spline_2d(path, interval=3)
        return np.array(path_smooth)
