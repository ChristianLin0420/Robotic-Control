import abc

class Planner:
    def __init__(self, m):
        self.map = m

    @abc.abstractmethod
    def planning(self, start, goal):
        return NotImplementedError

