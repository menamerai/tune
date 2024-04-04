import torch
from transformers import StoppingCriteria, PreTrainedTokenizerBase


class StopOnWords(StoppingCriteria):
    def __init__(
        self,
        stop_words: list[str],
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cpu",
    ):
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.device = device

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ):
        stop_token_ids = [
            self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt")[
                0
            ].to(self.device)
            for word in self.stop_words
        ]
        for stop_token_id in stop_token_ids:
            # check last n tokens with n = len(token_id)
            # print(f"test {self.tokenizer.batch_decode(input_ids[0][-len(stop_token_id):])} against {self.tokenizer.batch_decode(stop_token_ids)}")
            if torch.equal(input_ids[0][-len(stop_token_id) :], stop_token_id):
                return True
        return False

