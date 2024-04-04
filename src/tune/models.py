import torch
from abc import ABC
from typing import Any
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    StoppingCriteriaList,
)
from tune.utils import StopOnWords


class BaseModel(ABC):
    def __init__(self) -> None:
        raise NotImplementedError(
            "BaseModel is an abstract class and cannot be instantiated."
        )

    def predict(self) -> Any:
        raise NotImplementedError("predict() must be implemented in the subclass.")

    def train(self) -> None:
        raise NotImplementedError("train() must be implemented in the subclass.")

    def save(self) -> None:
        raise NotImplementedError("save() must be implemented in the subclass.")


class CausalLMModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        conf: PretrainedConfig | None = None,
        q_conf: BitsAndBytesConfig | None = None,
        stop_words: list[str] | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stop_words = stop_words
        self.model_name = model_name
        if conf is None:
            self.config = AutoConfig.from_pretrained(model_name)
        else:
            self.config = conf
        if q_conf is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, token=hf_token
            )
        else:
            self.model = AutoModelForCausalLM.from_config(
                self.config, quantization_config=q_conf, token=hf_token
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            stopping_criteria=(
                StoppingCriteriaList(
                    [StopOnWords(self.stop_words, self.tokenizer, self.device)]
                )
                if self.stop_words
                else None
            ),
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=250,
            return_full_text=False,
        )

    def predict(self, prompt: str) -> str:
        pred = self.generator(prompt)[0]["generated_text"]
        if self.stop_words is None:  # might need a parsing function here
            return pred
        else:
            for stop_word in self.stop_words:
                pred = pred.replace(stop_word, "")

            return pred.strip()

    def train(self) -> None:
        pass

    def save(self) -> None:
        pass
