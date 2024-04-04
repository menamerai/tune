import datasets

from dataclasses import dataclass
from enum import Enum
from tune.tasks.base import BaseTask
from tune.models import BaseModel
from tune.prompt.winogrande import WinograndePrompt, winogrande_example_factory


class WinograndeTask(BaseTask):

    WG_PREDS = Enum("WG_PREDS", "1 2")
    PROMPT_MODES = Enum("PROMPT_MODES", "baseline")

    def __init__(
        self,
        model: BaseModel,
        dataset_name: str = "winogrande",
        size: str = "winogrande_xs",
        examples: dict[str, list[str]] = winogrande_example_factory(),
    ) -> None:
        self.model = model
        self.prompt_generator = WinograndePrompt(
            examples=examples, stop_words=["</ANSWER>"]
        )
        self.dataset = datasets.load_dataset(dataset_name, size)

    def infer_single(self, doc: dict[str, str]) -> WG_PREDS:
        """
        Predict the correct option for a single Winogrande example.

        Args:
            doc (dict): A dictionary containing the sentence and two options to predict.

        Returns:
            str: The predicted option.
        """
        prompt = self.prompt_generator.build_baseline_prompt(doc)
        return self.model.predict(prompt)

    def infer_all(self, docs: dict[str, list[str]]) -> list[WG_PREDS]:
        """
        Predict the correct option for a list of Winogrande examples.

        Args:
            docs (dict): A dictionary containing the sentences and two options to predict.

        Returns:
            list: A list of predicted options.
        """
        results = []
        for i in range(len(docs["sentence"])):
            doc = {
                "sentence": docs["sentence"][i],
                "option1": docs["option1"][i],
                "option2": docs["option2"][i],
            }
            pred = self.infer_single(doc)
            # check if prediction is valid
            if pred not in self.WG_PREDS.__members__:
                print(f"Invalid prediction for example {doc}: {pred}")
                results.append("1")  # default to 1
            else:
                results.append(pred)

        return results
