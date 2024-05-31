import argparse
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers import DDPMScheduler
import torch

PROMPT = "Bill Gates with a hoodie"

def parse_arguments():
    parser = argparse.ArgumentParser(description="SDXL Training Script")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="./sdxl-finetuned",
    )

    parser.add_argument(
        "--lora-weight",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=82,
    )
    
    return parser.parse_args()

def main(args):
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model_name_or_path)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    if args.lora_weight is not None:
        pipe.load_lora_weights(args.lora_weight)

    pipe = pipe.to("cuda")
    generator = torch.Generator().manual_seed(args.seed) 
    with torch.no_grad():
        img = pipe(PROMPT, num_inference_steps=25, generator = generator )
    img.images[0].save('sdxl_result-1.png')

if __name__=="__main__":
    args = parse_arguments()
    main(args)