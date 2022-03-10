import argparse
import numpy as np
import cv2
from Simulation.utils import ControlState
from Simulation.simulator_map import SimulatorMap
from PathPlanning.cubic_spline import *

##############################
# Global Variables
##############################
pose = None
nav_pos = None
way_points = None
path = None
m_cspace = None
set_controller_path = False

##############################
# Navigation
##############################
# Mouse Click Callback
def mouse_click(event, x, y, flags, param):
    global pose, nav_pos, way_points, path, m_cspace, set_controller_path
    if event == cv2.EVENT_LBUTTONUP:
        nav_pos_new = (x, m.shape[0]-y)
        if m_cspace[nav_pos_new[1], nav_pos_new[0]] > 0.5:
            way_points = planner.planning((pose[0],pose[1]), nav_pos_new, 20)
            if len(way_points) > 1:
                nav_pos = nav_pos_new
                path = np.array(cubic_spline_2d(way_points, interval=4))
                set_controller_path = True

def pos_int(p):
    return (int(p[0]), int(p[1]))

def render_path(img, nav_pos, way_points, path):
    cv2.circle(img,nav_pos,5,(0.5,0.5,1.0),3)
    for i in range(len(way_points)):    # Draw Way Points
        cv2.circle(img, pos_int(way_points[i]), 3, (1.0,0.4,0.4), 1)
    for i in range(len(path)-1):    # Draw Interpolating Curve
        cv2.line(img, pos_int(path[i]), pos_int(path[i+1]), (1.0,0.4,0.4), 1)
    return img

def navigation(args, simulator, controller, planner, start_pose=(100,200,0)):
    global pose, nav_pos, way_points, path, set_controller_path
    # Initialize
    window_name = "Known Map Navigation Demo"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)
    simulator.init_pose(start_pose)
    command = ControlState(args.simulator, None, None)
    pose = start_pose
    collision_count = 0
    # Main Loop
    while(True):
        # Update State
        simulator.step(command)
        pose = (simulator.state.x, simulator.state.y, simulator.state.yaw)
        print("\r", simulator, "| Goal:", nav_pos, end="\t")
        
        if set_controller_path:
            controller.set_path(path)
            set_controller_path = False

        if path is not None and collision_count == 0:
            # TODO: Planning and Controlling
            if args.simulator == "basic":
                next_v = 0
                next_w = 0
                command = ControlState("basic", next_v, next_w)
            elif args.simulator == "diff_drive":
                next_lw = 0
                next_rw = 0
                command = ControlState("diff_drive", next_lw, next_rw)
            elif args.simulator == "bicycle":
                next_a = 0
                next_delta = 0
                command = ControlState("bicycle", next_a, next_delta)
            else:
                exit()            
        else:
            command = None

        _, info = simulator.step(command)
        # Collision Handling
        if info["collision"]:
            collision_count = 1
        if collision_count > 0:
            # TODO: Collision Handling
            pass
        
        # Render Path
        img = simulator.render()
        if nav_pos is not None and way_points is not None:
            img = render_path(img, nav_pos, way_points, path)

        img = cv2.flip(img, 0)
        cv2.imshow(window_name, img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            simulator.init_state(start_pose)
        if k == 27:
            print()
            break

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str, default="diff_drive", help="diff_drive/bicycle")
    parser.add_argument("-c", "--controller", type=str, default="pure_pursuit", help="pid/pure_pursuit/stanley/lqr")
    parser.add_argument("-p", "--planner", type=str, default="a_star", help="a_star/rrt/rrt_star")
    parser.add_argument("-m", "--map", type=str, default="Maps/map1.png", help="image file name")
    args = parser.parse_args()

    # Read Map
    img = cv2.flip(cv2.imread(args.map),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    m_cspace = 1-cv2.dilate(1-m, np.ones((40,40))) # Configuration-Space

    # Select Simulator, Controller, and Planner
    try:
        # Simulator / Controller
        if args.simulator == "basic":
            from Simulation.simulator_basic import SimulatorBasic
            simulator = SimulatorMap(SimulatorBasic, m=m, l=9, wu=7, wv=3, car_w=16, car_f=13, car_r=7)
            if args.controller == "pid":
                from PathTracking.controller_pid_basic import ControllerPIDBasic as Controller
                controller = Controller()
            elif args.controller == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_basic import ControllerPurePursuitBasic as Controller
                controller = Controller(Lfc=1)
            elif args.controller == "lqr":
                from PathTracking.controller_lqr_basic import ControllerLQRBasic as Controller
                controller = Controller()
            else:
                raise NameError("Unknown controller!!")
        elif args.simulator == "diff_drive":
            from Simulation.simulator_differential_drive import SimulatorDifferentialDrive
            simulator = SimulatorMap(SimulatorDifferentialDrive, m=m, l=9, wu=7, wv=3, car_w=16, car_f=13, car_r=7)
            if args.controller == "pid":
                from PathTracking.controller_pid_basic import ControllerPIDBasic as Controller
                controller = Controller()
            elif args.controller == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_basic import ControllerPurePursuitBasic as Controller
                controller = Controller(Lfc=1)
            elif args.controller == "lqr":
                from PathTracking.controller_lqr_basic import ControllerLQRBasic as Controller
                controller = Controller()
            else:
                raise NameError("Unknown controller!!")
        elif args.simulator == "bicycle":
            from Simulation.simulator_bicycle import SimulatorBicycle 
            simulator = SimulatorMap(SimulatorBicycle, m=m, l=20, d=5, wu=5, wv=2, car_w=14, car_f=25, car_r=5)
            if args.controller == "pid":
                from PathTracking.controller_pid_bicycle import ControllerPIDBicycle as Controller
                controller = Controller()
            elif args.controller == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_bicycle import ControllerPurePursuitBicycle as Controller
                controller = Controller(Lfc=1)
            elif args.controller == "stanley":
                from PathTracking.controller_stanley_bicycle import ControllerStanleyBicycle as Controller
                controller = Controller()
            elif args.controller == "lqr":
                from PathTracking.controller_lqr_bicycle import ControllerLQRBicycle as Controller
                controller = Controller()
            else:
                raise NameError("Unknown controller!!")
        else:
            raise NameError("Unknown simulator!!")
        # Planner
        if args.planner == "a_star":
            from PathPlanning.planner_a_star import PlannerAStar as Planner
        elif args.planner == "rrt":
            from PathPlanning.planner_rrt import PlannerRRT as Planner
        elif args.planner == "rrt_star":
            from PathPlanning.planner_rrt_star import PlannerRRTStar as Planner
        else:
            print("Unknown planner !!")
            exit(0)
        planner = Planner(m_cspace)
    except:
        raise
    
    navigation(args, simulator, controller, planner)