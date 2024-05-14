import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'model'))

from transformers import AutoTokenizer
from modeling_baichuan import BaichuanForCausalLM


# Saved model path
CHECKPOINT = "./baichuan_code_generation"

# Model name for tokenizer
MODEL_NAME = "baichuan-inc/Baichuan-13B-Base"

# Max New Tokens for generating
MAX_NEW_TOKENS = 512

# Load trained model
model = BaichuanForCausalLM.from_pretrained(CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.cuda()
model.eval()

# Prepare test prompt
input_text = f"[INST] I can no longer afford order 11234, cancel it [/INST]"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
generated_text_ids = input_ids.cuda()

with torch.no_grad():
    # Generate python function
    output = model.generate(generated_text_ids, max_new_tokens=MAX_NEW_TOKENS)

    # Decode generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Baichuan: {generated_text}")
