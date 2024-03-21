import datasets

from tune.tasks import BaseTask

class WinograndeTask(BaseTask):
    def __init__(self, dataset_name: str = "winogrande") -> None:
        self.dataset = datasets.load_dataset(dataset_name)

    def get_train_data(self) -> datasets.Dataset:
        return self.dataset["train"]
    
    def get_val_data(self) -> datasets.Dataset:
        return self.dataset["validation"]