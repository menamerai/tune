from tune.models import CausalLMModel
from tune.tasks.winogrande import WinograndeTask
from tune.eval import evaluate_classification
# from transformers import BitsAndBytesConfig


# q_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

print("Loading model...")
model = CausalLMModel(model_name="microsoft/phi-1_5", stop_words=["</ANSWER>"])
print("Model loaded.")

print("Loading task...")
task = WinograndeTask(model=model)
print("Task loaded.")

eval_dataset = task.dataset["validation"][:5]

print("Predicting...")
preds = task.infer_all(eval_dataset)
evaluate_classification(eval_dataset["answer"], preds, ["1", "2"], save_folder="./output")
print("Prediction complete.")
