import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlState
from Simulation.kinematic import KinematicModel

class KinematicModelDifferentialDrive(KinematicModel):
    def __init__(self, r, l, dt):
        # Simulation delta time
        self.r = r
        self.l = l
        self.dt = dt
    
    def step(self, state:State, cstate:ControlState) -> State:
        # TODO: Differential Drive Kinematic Model
        state_next = state
        return state_next