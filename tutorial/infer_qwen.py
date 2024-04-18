import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'model'))

from transformers import AutoTokenizer, AutoModelForCausalLM


# Saved model path
CHECKPOINT = "./qwen_code_generation"

# Model name for tokenizer
MODEL_NAME = "Qwen/Qwen1.5-7B"

# Max New Tokens for generating
MAX_NEW_TOKENS = 512

# Load trained model
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
model.cuda()

# Prepare test prompt
input_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a function to join given list of strings with space.\n\n### Input:\n['I', 'love', 'you']\n\n### Output:\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
generated_text_ids = input_ids.cuda()

with torch.no_grad():
    # Generate python function
    output = model.generate(generated_text_ids, max_new_tokens=MAX_NEW_TOKENS)

    # Decode generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Mistral: {generated_text}")
