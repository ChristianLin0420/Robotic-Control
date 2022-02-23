import numpy as np
import cv2
import argparse
from Simulation.utils import ControlState
import PathTracking.utils

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str, default="diff_drive", help="basic/diff_drive/bicycle")
    parser.add_argument("-c", "--controller", type=str, default="pure_pursuit", help="pid/pure_pursuit/stanley/lqr")
    parser.add_argument("-t", "--path_type", type=int, default=2, help="1/2")
    args = parser.parse_args()

    # Select Simulator and Controller
    try:
        # Basic Kinematic Model 
        if args.simulator == "basic":
            from Simulation.simulator_basic import SimulatorBasic as Simulator
            if args.controller == "pid":
                from PathTracking.controller_pid_basic import ControllerPIDBasic as Controller
            elif args.controller == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_basic import ControllerPurePursuitBasic as Controller
            elif args.controller == "stanley":
                from PathTracking.controller_stanley_basic import ControllerStanleyBasic as Controller
            elif args.controller == "lqr":
                from PathTracking.controller_lqr_basic import ControllerLQRBasic as Controller
            else:
                raise NameError("Unknown controller!!")
        # Differential Drive Kinematic Model
        elif args.simulator == "diff_drive":
            from Simulation.simulator_differential_drive import SimulatorDifferentialDrive as Simulator
            if args.controller == "pid":
                from PathTracking.controller_pid_basic import ControllerPIDBasic as Controller
            elif args.controller == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_basic import ControllerPurePursuitBasic as Controller
            elif args.controller == "stanley":
                from PathTracking.controller_stanley_basic import ControllerStanleyBasic as Controller
            elif args.controller == "lqr":
                from PathTracking.controller_lqr_basic import ControllerLQRBasic as Controller
            else:
                raise NameError("Unknown controller!!")
        # Bicycle Model
        elif args.simulator == "bicycle":
            from Simulation.simulator_bicycle import SimulatorBicycle as Simulator
            if args.controller == "pid":
                from PathTracking.controller_pid_bicycle import ControllerPIDBicycle as Controller
            elif args.controller == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_bicycle import ControllerPurePursuitBicycle as Controller
            elif args.controller == "stanley":
                from PathTracking.controller_stanley_bicycle import ControllerStanleyBicycle as Controller
            elif args.controller == "lqr":
                from PathTracking.controller_lqr_bicycle import ControllerLQRBicycle as Controller
            else:
                raise NameError("Unknown controller!!")
        else:
            raise NameError("Unknown simulator!!")
    except:
        raise

    print("Simulator:", args.simulator, "| Controller:", args.controller)

    # Create Path
    if args.path_type == 1:
        path = PathTracking.utils.path1()
    elif args.path_type == 2:
        path = PathTracking.utils.path2()
    else:
        print("Unknown path type !!")
        exit(0)

    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1): # Draw Path
        p1 = (int(path[i,0]), int(path[i,1]))
        p2 = (int(path[i+1,0]), int(path[i+1,1]))
        cv2.line(img_path, p1, p2, (1.0,0.5,0.5), 1)
    
    # Initialize Car
    simulator = Simulator()
    start = (50,300,0)
    simulator.init_pose(start)
    controller = Controller()
    controller.set_path(path)

    while(True):
        print("\r", simulator, end="\t")
        # Control
        end_dist = np.hypot(path[-1,0]-simulator.state.x, path[-1,1]-simulator.state.y)
        if args.simulator == "basic":
            if end_dist > 10:
                next_v = 20
            else:
                next_v = 0
            # Lateral
            info = {
                "x":simulator.state.x, 
                "y":simulator.state.y, 
                "yaw":simulator.state.yaw, 
                "v":simulator.state.v, 
                "dt":simulator.dt
            }
            next_w, target = controller.feedback(info)
            command = ControlState("basic", next_v, next_w)
        elif args.simulator == "diff_drive":
            # Longitude
            if end_dist > 10:
                next_v = 20
            else:
                next_v = 0
            # Lateral
            info = {
                "x":simulator.state.x, 
                "y":simulator.state.y, 
                "yaw":simulator.state.yaw, 
                "v":simulator.state.v, 
                "dt":simulator.dt
            }
            next_w, target = controller.feedback(info)
            # : v,w to motor control
            r = simulator.wu/2
            next_lw = 0
            next_rw = 0
            command = ControlState("diff_drive", next_lw, next_rw)
        elif args.simulator == "bicycle":
            # Longitude (P Control)
            if end_dist > 40:
                target_v = 20
            else:
                target_v = 0
            next_a = (target_v - simulator.state.v)*0.5
            # Lateral
            info = {
                "x":simulator.state.x, 
                "y":simulator.state.y, 
                "yaw":simulator.state.yaw, 
                "v":simulator.state.v,
                "delta":simulator.cstate.delta,
                "l":simulator.l, 
                "dt":simulator.dt
            }
            next_delta, target = controller.feedback(info)
            command = ControlState("bicycle", next_a, next_delta)
 
        # Update State & Render
        simulator.step(command)
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = simulator.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("Path Tracking Test", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            simulator.init_state(start)
        if k == 27:
            print()
            break
