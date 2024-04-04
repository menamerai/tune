import os

import torch
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig

from tune.eval import evaluate_classification
from tune.models import CausalLMModel
from tune.tasks.winogrande import WinograndeTask

if __name__ == "__main__":
    load_dotenv()
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = CausalLMModel(
        model_name="meta-llama/Llama-2-7b-hf",
        hf_token=os.getenv("HF_TOKEN"),
        q_conf=q_config,
        stop_words=["</ANSWER>"],
    )
    task = WinograndeTask(model=model)

    eval_dataset = task.dataset["validation"][:5]

    preds = task.infer_all(eval_dataset)
    evaluate_classification(
        eval_dataset["answer"], preds, ["1", "2"], save_folder="./output"
    )
    print("Prediction complete.")
