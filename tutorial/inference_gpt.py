import torch
import sys, os
import argparse
from transformers import AutoModelForCausalLM, AdamW, AutoTokenizer

QUERY = ""

def inference(args):
    # Saved model path
    # Max New Tokens for generating
    
    # Load trained model
    model = AutoModelForCausalLM.from_pretrained(args.model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.cuda()
    model.eval()

    # Prepare test prompt
    input_text = f"[INST] {QUERY} [/INST]"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    generated_text_ids = input_ids.cuda()

    with torch.no_grad():
        # Generate python function
        output = model.generate(generated_text_ids, max_new_tokens=args.max_length)

        # Decode generated tokens
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"{generated_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "cerebras/Cerebras-GPT-13B")
    parser.add_argument("--max-length", type = int, default = 512)
    parser.add_argument("--model-save-path", type =str, default="./gpt_checkpoint")
    args = parser.parse_args()
    inference(args)