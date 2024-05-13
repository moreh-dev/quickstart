from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


model_id = "meta-llama/Meta-Llama-3-8B"
pipeline = pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)


print("Model save start")
pipeline.save_pretrained('./llama3-8b', use_safetensors=True)
print("Model save end")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.save_pretrained("./llama3-8b")

