import torch

from transformers import AutoTokenizer
from modeling_mistral import MistralForCausalLM


# Saved model path
CHECKPOINT = "./code_generation"

# Model name for tokenizer
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Max New Tokens for generating
MAX_NEW_TOKENS = 1024

# Load trained model
model = MistralForCausalLM.from_pretrained(CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.cuda()
model.eval()

# Prepare test prompt
input_text = f"[INST] Write a python function to find the volume of a triangular prism. [/INST]"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
generated_text_ids = input_ids.cuda()

# Generate python function
output = model.generate(generated_text_ids, max_new_tokens=MAX_NEW_TOKENS)

# Decode generated tokens
generated_text = tokenizer.decode(output, skip_special_tokens=True)
print(f"Mistral: {generated_text}")
