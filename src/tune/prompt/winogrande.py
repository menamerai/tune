from dataclasses import dataclass
from tune.prompt import BasePrompt


def winogrande_example_factory() -> dict[str, list[str]]:
    return {
        "setence": [
            "The city councilmen refused the demonstrators a permit because _ feared violence.",
            "The trophy doesn't fit into the brown suitcase because _ is too large.",
            "Paul tried to call George on the phone, but _ wasn't successful.",
        ],
        "option1": [
            "The city councilmen",
            "The brown suitcase",
            "Paul",
        ],
        "option2": [
            "The demonstrators",
            "The trophy",
            "George",
        ],
        "answer": [0, 1, 0],
    }


def winogrande_instructions_factory() -> str:
    instruct = "This is a problem that involve choosing the right option to complete a sentence.\n"
    instruct += "You will be given a sentence with two options. Choose the option that best completes the sentence.\n"
    return instruct


@dataclass(kw_only=True)
class WinograndePrompt(BasePrompt):
    instructions: str = winogrande_instructions_factory()
    examples: dict[str, list[str]]
    stop_words: list[str]

    def __post_init__(self):
        super().__post_init__()
        # check if examples are valid
        if not all([key in self.examples for key in ["setence", "option1", "option2", "answer"]]):
            raise ValueError("Invalid examples dictionary.")
        if len(self.examples["setence"]) != len(self.examples["option1"]) != len(self.examples["option2"]) != len(self.examples["answer"]):
            raise ValueError("Invalid examples dictionary.")

    def build_baseline_prompt(self, doc: dict[str, str]) -> str:
        """
        Build a prompt for the Winogrande task with the baseline instructions.

        Args:
            doc (dict): A dictionary containing the sentence and two options.

        Returns:
            str: The prompt for the Winogrande task.
        """
        example_text = "Here are some examples:\n\n"
        for i in range(len(self.examples["setence"])):
            example_text += f"<SENTENCE>\n{self.examples['setence'][i]}\n</SENTENCE>\n"
            example_text += f"<OPTIONS>\n1: {self.examples['option1'][i]}\n2: {self.examples['option2'][i]}\n</OPTIONS>\n"
            example_text += f"<ANSWER>\n{self.examples['answer'][i] + 1}\n</ANSWER>\n\n"

        query_text = "Here is the question:\n\n"
        query_text += f"<SENTENCE>\n{doc['sentence']}\n</SENTENCE>\n"
        query_text += f"<OPTIONS>\n1: {doc['option1']}\n2: {doc['option2']}\n</OPTIONS>\n"
        query_text += f"<ANSWER>"

        return self.instructions + example_text + query_text
    
    
