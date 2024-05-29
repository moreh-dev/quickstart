import torch
from datasets import load_dataset
from argparse import ArgumentParser
from transformers import AutoTokenizer

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        help="dataset name or path",
        default='bitext/Bitext-customer-support-llm-chatbot-training-dataset'
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="model name or path",
        default='baichuan-inc/Baichuan-13B-Base'
    )
    parser.add_argument(
        "--block-size", 
        type=int, 
        default=1024, 
        help="max input token length"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default='./baichuan_dataset.pt'
    )
    args = parser.parse_args()
    return args


# Construct a formatted prompt
def create_prompt(prompt):
    full_prompt = f"[INST] {prompt['instruction']} \n Category is {prompt['category']} \n Intent is {prompt['intent']} [/INST]\n{prompt['response']}"
    return full_prompt


def main(args):
    dataset = load_dataset(args.dataset_name_or_path).with_format("torch")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)


    def preprocess(prompt):
        tokenized = tokenizer(
            create_prompt(prompt),
            padding="max_length",
            truncation=True,
            max_length=args.block_size,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    dataset = dataset.map(preprocess, num_proc=16, load_from_cache_file=True)
    torch.save(dataset, args.save_path)



if __name__ == '__main__':
    args = parse_args()
    main(args)
