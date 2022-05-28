import abc
import sys
sys.path.append("..")
from Simulation.utils import State, ControlState

class KinematicModel:
    @abc.abstractmethod
    def step(self, state:State, cstate:ControlState) -> State:
        return NotImplementedError