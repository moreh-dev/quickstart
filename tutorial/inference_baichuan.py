import torch
import sys, os
from argparse import ArgumentParser
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'model'))

from transformers import AutoTokenizer
from modeling_baichuan import BaichuanForCausalLM


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--max-length", type = int, default = 512)
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="./baichuan_code_generation"
    )
    parser.add_argument("--use-lora", action = "store_true")
    return parser.parse_args()

# Saved model path
def main(args):

    # Max New Tokens for generating
    MAX_NEW_TOKENS = 512

    # Load trained model
    if not args.use_lora:
        model = BaichuanForCausalLM.from_pretrained(args.model_save_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_save_path, trust_remote_code=True)
    else:
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(args.model_save_path)
        model = BaichuanForCausalLM.from_pretrained(config.base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, args.model_save_path)
        model = model.merge_and_unload()
    model.cuda()
    model.eval()

    # Prepare test prompt
    input_text = f"[INST] I can no longer afford order 11234, cancel it [/INST]"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    generated_text_ids = input_ids.cuda()

    with torch.no_grad():
        # Generate python function
        output = model.generate(generated_text_ids,  max_new_tokens=args.max_length)

        # Decode generated tokens
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Baichuan: {generated_text}")

if __name__ == "__main__":
    args = parse_args()
    main(args)