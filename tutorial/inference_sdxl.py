import argparse
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers import DDPMScheduler
import torch

PROMPT = "a man in a green jacket with a sword"

def parse_arguments():
    parser = argparse.ArgumentParser(description="SDXL Training Script")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="./sdxl-finetuned",
    )
    
    return parser.parse_args()

def main(args):
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model_name_or_path)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to("cuda")
    generator = torch.Generator().manual_seed(78) 
    with torch.no_grad():
        img = pipe(PROMPT, num_inference_steps=25, generator = generator)
    img.images[0].save('sdxl_result.png')

if __name__=="__main__":
    args = parse_arguments()
    main(args)