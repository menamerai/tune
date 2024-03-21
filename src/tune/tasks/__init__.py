from abc import ABC
from datasets import Dataset

class BaseTask(ABC):
    def __init__(self) -> None:
        raise NotImplementedError("BaseTask is an abstract class and cannot be instantiated.")
    
    def get_prompt(self) -> str:
        raise NotImplementedError("get_prompt() must be implemented in the subclass.")

    def get_train_data(self) -> Dataset:
        raise NotImplementedError("get_train_data() must be implemented in the subclass.")