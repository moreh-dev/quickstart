import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from argparse import ArgumentParser

QUERY = """Write a python program that counts all 'a's in a string. For example, if the string "Banana" is given, the program should return 3.
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--max-length", type = int, default = 512)
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="./opt_checkpoint"
    )
    parser.add_argument("--use-lora", action = "store_true")
    return parser.parse_args()

def main(args): 
    # Load trained model    
    if not args.use_lora:
        model = AutoModelForCausalLM.from_pretrained(args.model_save_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_save_path)
    else:
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(args.model_save_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, args.model_save_path)
        model = model.merge_and_unload()

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
        answer = re.findall(r"\[INST\][\S\s]*\[\/INST\]([\S\s]*)", generated_text)[0]
        print(f"{answer}")


if __name__ == "__main__":
    args = parse_args()
    main(args)