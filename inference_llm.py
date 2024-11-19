from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--model-name-or-path",
                        type=str,
                        default="llama3-finetuned")
    parser.add_argument("--use-lora", action="store_true")
    return parser.parse_args()


def main(args):
    # Load trained model
    if not args.use_lora:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        from peft import PeftConfig
        from peft import PeftModel
        config = PeftConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, args.model_name_or_path)
        model = model.merge_and_unload()
    model.eval()
    model.cuda()

    # Prepare test prompt
    input_text = "##INSTRUCTION What is the status of my return for {{Order Number}}? \n\n ##RESPONSE "
    tokenized_input = tokenizer(input_text, return_tensors="pt")
    input_ids = tokenized_input['input_ids'].to('cuda')
    attention_mask = tokenized_input['attention_mask'].to('cuda')

    with torch.no_grad():
        # Generate python function
        output = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=args.max_length,
                                pad_token_id=tokenizer.eos_token_id)

        # Decode generated tokens
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"{generated_text}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
