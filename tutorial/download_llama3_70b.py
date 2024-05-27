from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def main(args):

    print(f"Load {args.model_name} model checkpoint and tokenizer...")
    model = pipeline(
    "text-generation", model=args.model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token)

    # Save it to ./llama3-70b folder
    print(f"Save pretrained {args.model_name    } model checkpoint and tokenizer to {args.save_dir} folder...")
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
        default="meta-llama/Meta-Llama-3-70B",
        help="Model name",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./llama3-70b",
        help="Save directory",
    )
    args = parser.parse_args()

    main(args)
