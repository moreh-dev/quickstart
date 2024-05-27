from transformers import AutoTokenizer, AutoModelForCausalLM


def main(args):

    # Load Llama-8b-base model and tokenizer
    print(f"Load {args.model_name} model checkpoint and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=args.token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token)

    # Save it to ./llama3-8b- folder
    print(f"Save pretrained {args.model_name} model checkpoint and tokenizer to {args.save_dir} folder...")
    model.save_pretrained(args.save_dir, use_safetensors=True)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace Token",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model name",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./llama3-8b",
        help="Save directory",
    )
    args = parser.parse_args()

    main(args)
