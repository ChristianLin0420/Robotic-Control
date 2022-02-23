import argparse
import numpy as np
import cv2
from Simulation.utils import ControlState

# Basic Kinematic Model
def run_basic():
    from Simulation.simulator_basic import SimulatorBasic as Simulator
    print("Control Hint:")
    print("[W] Increase velocity.")
    print("[S] Decrease velocity.")
    print("[A] Increase angular velocity. (Anti-Clockwise)")
    print("[D] Decrease angular velocity. (Clockwise)")
    print("====================")
    simulator = Simulator()
    simulator.init_pose((300,300,0))
    while(True):
        k = cv2.waitKey(1)
        if k == ord("a"):
            command = ControlState(args.simulator, None, simulator.cstate.w+5)
        elif k == ord("d"):
            command = ControlState(args.simulator, None, simulator.cstate.w-5)
        elif k == ord("w"):
            command = ControlState(args.simulator, simulator.cstate.v+4, None)
        elif k == ord("s"):
            command = ControlState(args.simulator, simulator.cstate.v-4, None)
        elif k == 27:
            print()
            break
        else:
            command = ControlState(args.simulator, None, None)
        simulator.step(command)
        print("\r", simulator, end="\t")
        img = simulator.render()
        img = cv2.flip(img, 0)
        cv2.imshow("Motion Model", img)
        
# Diferential-Drive Kinematic Model
def run_diff_drive():
    from Simulation.simulator_differential_drive import SimulatorDifferentialDrive as Simulator
    print("Control Hint:")
    print("[A] Decrease angular velocity of left wheel.")
    print("[Q] Increase angular velocity of left wheel.")
    print("[D] Decrease angular velocity of right wheel.")
    print("[E] Increase angular velocity of right wheel.")
    print("====================")
    simulator = Simulator()
    simulator.init_pose((300,300,0))
    command = ControlState(args.simulator, None, None)
    while(True):
        k = cv2.waitKey(1)
        if k == ord("a"):
            command = ControlState(args.simulator, simulator.cstate.lw-30, None)
        elif k == ord("d"):
            command = ControlState(args.simulator, None, simulator.cstate.rw-30)
        elif k == ord("q"):
            command = ControlState(args.simulator, simulator.cstate.lw+30, None)
        elif k == ord("e"):
            command = ControlState(args.simulator, None, simulator.cstate.rw+30)
        elif k == 27:
            print()
            break
        else:
            command = ControlState(args.simulator, None, None)
        simulator.step(command)
        print("\r", simulator, end="\t")
        img = simulator.render()
        img = cv2.flip(img, 0)
        cv2.imshow("Motion Model", img)
        
# Bicycle Kinematic Model
def run_bicycle():
    from Simulation.simulator_bicycle import SimulatorBicycle as Simulator
    print("Control Hint:")
    print("[W] Increase velocity.")
    print("[S] Decrease velocity.")
    print("[A] Wheel turn anti-clockwise.")
    print("[D] Wheel turn clockwise.")
    print("====================")
    simulator = Simulator()
    simulator.init_pose((300,300,0))
    command = ControlState(args.simulator, None, None)
    while(True):
        k = cv2.waitKey(1)
        if k == ord("a"):
            command = ControlState(args.simulator, 0, simulator.cstate.delta+5)
        elif k == ord("d"):
            command = ControlState(args.simulator, 0, simulator.cstate.delta-5)
        elif k == ord("w"):
            command = ControlState(args.simulator, simulator.cstate.a+10, None)
        elif k == ord("s"):
            command = ControlState(args.simulator, simulator.cstate.a-10, None)
        elif k == 27:
            print()
            break
        else:
            command = ControlState(args.simulator, 0, None)
        simulator.step(command)
        print("\r", simulator, end="\t")
        img = simulator.render()
        img = cv2.flip(img, 0)
        cv2.imshow("Motion Model", img)
        
if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str, default="basic", help="basic/diff_drive/bicycle")
    args = parser.parse_args()
    try:
        if args.simulator == "basic":
            run_basic()
        elif args.simulator == "diff_drive":
            run_diff_drive()
        elif args.simulator == "bicycle":
            run_bicycle()
        else:
            raise NameError("Unknown simulator!!")
    except NameError:
        raise
