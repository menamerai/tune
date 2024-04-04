from abc import ABC

class BaseTask(ABC):
    def __init__(self) -> None:
        raise NotImplementedError("BaseTask is an abstract class and cannot be instantiated.")

    def train(self) -> None:
        raise NotImplementedError("train() must be implemented in the subclass.")
    
    def infer(self) -> None:
        raise NotImplementedError("infer() must be implemented in the subclass.")
    