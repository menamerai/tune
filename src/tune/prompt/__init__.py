from dataclasses import dataclass
from abc import ABC


@dataclass
class BasePrompt(ABC):
    instructions: str

    def __post_init__(self):
        if not isinstance(self.instructions, str):
            raise ValueError("Prompt must be a string.")

    def __str__(self):
        return self.instructions

    def __repr__(self):
        return self.instructions
