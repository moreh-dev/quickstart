### List of Contents 

- [Quickstart](#quickstart)
- [Getting Started](#getting-started)
  - [Dependency Installation](#dependency-installation)
  - [Model Preparation](#model-preparation)
- [Fine-tuning](#fine-tuning)
  - [Activate Moreh Advanced Parallelization(AP)](#activate-moreh-advanced-parallelizationap)
  - [LLM Information](#llm-information)
  - [Training](#training)
    - [Full Fine-tuning](#full-fine-tuning)
    - [LoRA Fine-tuning](#lora-fine-tuning)
    - [Inference](#inference)
  - [Stable Diffusion XL](#stable-diffusion-xl)
    - [Training](#training-1)
    - [Inference](#inference-1)
# Quickstart

This repository provides code to experiment with training large models on [Moreh's MoAI Platform](https://moreh.io/product).
With the MoAI platform you can scale to thousands of GPU/NPUs by automatic parallelization and optimization, without any code changes.

We currently provide four LLMs; Llama3, Qwen2.5, Mistral, and Baichuan2, as well as SDXL.

    
# Getting Started
This repository contains examples of PyTorch training codes that can be executed on the MoAI Platform. Users using Pytorch on the MoAI Platform can easily train large models without extensive effort. For more information about the MoAI Platform and detailed tutorials, please visit the [Moreh Docs](https://docs.moreh.io).

## Dependency Installation

First, clone this repository and navigate to the repo directory.
```bash
git clone https://github.com/moreh-dev/quickstart
cd quickstart
```
After you are in the `quickstart` directory, install the dependency packages by following commands :

```bash
pip install -r requirements/requirements_llm.txt
```

## Model Preparation
If you want to fine-tune the Llama2, Llama3, or Mistral models, you need access to their respective Hugging Face repositories. Please ensure you have the necessary acess before starting model training.
- Llama3 : https://huggingface.co/meta-llama/Meta-Llama-3-8B or https://huggingface.co/meta-llama/Meta-Llama-3-70B
- Mistral : https://huggingface.co/mistralai/Mistral-7B-v0.3

After obtaining access, authenticate your token with the following command:
```
huggingface-cli login
```

# Fine-tuning

## Activate Moreh Advanced Parallelization(AP)
The following line is added in the each code to enable AP on the MoAI Platform.
```python
...

torch.moreh.option.enable_advanced_parallelization()

...
```
## LLM Information

Information about the models currently supported by this repository are as follows:

| **Baseline Model**                                                      | **Model Card Name**             |
|-------------------------------------------------------------------------|---------------------------------|
| [Llama3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)          | meta-llama/Meta-Llama-3-8B      |
| [Llama3 70B  ](https://huggingface.co/meta-llama/Meta-Llama-3-70B)      | meta-llama/Meta-Llama-3-70B     |
| [Qwen2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B)                    | Qwen/Qwen2.5-7B                 |
| [Mistral v0.3 7B ](https://huggingface.co/mistralai/Mistral-7B-v0.3)    | mistralai/Mistral-7B-v0.3       |
| [Baichuan2 13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base) | baichuan-inc/Baichuan2-13B-Base |


## Training



### Full Fine-tuning

 Run the training script to fully fine-tune the model. For example, if you want to fine-tune the llama-3 8B model:
```bash 
TOKENIZERS_PARALLELISM=true accelerate launch --config_file config.yaml train_llm.py \
     --lr 0.000001 \
     --model meta-llama/Meta-Llama-3-8B \
     --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
     --train-batch-size 64 \
     --eval-batch-size 64 \
     --sequence-length 1024 \
     --log-interval 10 \
     --num-epochs 5 \
     --output-dir llama3-finetuned
```

### LoRA Fine-tuning

To train the LoRA adapter only, you can give a `--lora` argument with LoRA config parameters. 

```bash 
TOKENIZERS_PARALLELISM=true accelerate launch --config_file config.yaml train_llm.py \
     --lr 0.0001 \
     --model meta-llama/Meta-Llama-3-8B \
     --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
     --train-batch-size 64 \
     --eval-batch-size 64 \
     --sequence-length 1024 \
     --log-interval 10 \
     --num-epochs 5 \
     --lora \
     --lora-r 64 \
     --lora-alpha 16 \
     --lora-dropout 0.1 \
     --output-dir llama3-finetuned-lora
```
You can change model name in `--model` arguments to fine-tune your desired model.

If you want to fine-tune your model with the other dataset, you can fix `__call__` method of the `Preprocessor` class which is defined in `train_utils.py` to the desired format.

### Inference

Perform inference by running the inference script for each model.

```bash 
python inference_llm.py \ 
  --model-name-or-path ${SAVE_DIR_PATH}
```

If you want to perform inference with LoRA weights, add `--use-lora` argument to the inference script/

```bash 
python inference_llm.py \ 
  --model-name-or-path ${SAVE_DIR_PATH} \ 
  --use-lora
```

```
# output example
##INSTRUCTION What is the status of my return for {{Order Number}}?

##RESPONSE Thank you for contacting us regarding the status of your return for order number {{Order Number}}. To provide you with accurate information, I kindly request you to visit the '{{Order Status}}' section on our website. There, you will find the most up-to-date details on the progress of your return. If you have any further questions or need additional assistance, please don't hesitate to let me know. I'm here to help you every step of the way!
```


## Stable Diffusion XL

We provide fine-tuning example code for the Stable Diffusion XL model.

### Training
| Baseline Model                                                                         | Task                     | Training Script          | Dataset                                                                                            |
| -------------------------------------------------------------------------------------- | ------------------------ | ------------------------ | -------------------------------------------------------------------------------------------------- |
| [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Text-to-Image Generation | `tutorial/train_sdxl.py` | [lambdalabs/naruto-blip-captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) |


Run the training script for Stable Diffusion XL:

```bash
pip install -r requirements/requirements_sdxl.txt
```

```bash 
python train_sdxl.py \
  --epochs 20 \
  --batch-size 16 \
  --num-workers 8 \
  --lr=1e-05 \
  --save-dir=${SAVE_DIR_PATH} \
  --log-interval 1 \
  --lr-scheduler linear 
```

### Inference 
After training, you can proceed inference with your fine-tuned model using the following command:
```bash
python inference_sdxl.py \
  --model-name-or-path=${SAVE_DIR_PATH}
```

Adjust the prompt by editing the PROMPT variable in the inference script:
```python
...
PROMPT = "Bill Gates with a hoodie"
...
```


The resulting image will be saved as `sdxl_result.jpg`.  

The image on the left shows the inference results of the model before fine-tuning, while the image on the right shows the inference results of the fine-tuned model.
<p align="center">
<img src="assets/sdxl_nofinetune.png" alt="drawing" style="width:200px;"/> 
<img src="assets/sdxl_withfinetune.png" alt="drawing" style="width:200px;"/>
</p>
