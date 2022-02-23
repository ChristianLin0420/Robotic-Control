import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlState
from Simulation.kinematic import KinematicModel

class KinematicModelBasic(KinematicModel):
    def __init__(self, dt):
        # Simulation delta time
        self.dt = dt

    def step(self, state:State, cstate:ControlState) -> State:
        # TODO: Basic Kinematic Model
        state_next = state
        return state_next
