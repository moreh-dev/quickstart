import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="./mistral_code_generation"
    )
    parser.add_argument("--use-lora", action = "store_true")
    return parser.parse_args()
# Saved model path

def main(args):

    # Max New Tokens for generating
    MAX_NEW_TOKENS = 512

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
    input_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a function to join given list of strings with space.\n\n### Input:\n['I', 'love', 'you']\n\n### Output:\n"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    generated_text_ids = input_ids.cuda()

    with torch.no_grad():
        # Generate python function
        output = model.generate(generated_text_ids, max_new_tokens=MAX_NEW_TOKENS)

        # Decode generated tokens
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Mistral: {generated_text}")

if __name__ == "__main__":
    args = parse_args()
    main(args)