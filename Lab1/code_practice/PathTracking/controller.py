import abc

class Controller:
    def set_path(self, path):
        self.path = path

    @abc.abstractmethod
    def feedback(self, info):
        return NotImplementedError
