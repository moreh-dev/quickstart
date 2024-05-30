### List of Contents 

- [Quickstart](#quickstart)
- [Getting Started](#getting-started)
  - [Dependency Installation](#dependency-installation)
  - [Model Prepare](#model-prepare)
- [Fine-tuning](#fine-tuning)
  - [Activate Moreh Advanced Parallelization(AP)](#activate-moreh-advanced-parallelizationap)
  - [LLM](#llm)
    - [Training](#training)
    - [Inference](#inference)
  - [Stable Diffusion XL](#stable-diffusion-xl)
    - [Training](#training-1)
    - [Inference](#inference-1)
# Quickstart

This repository provides code to experiment with training large models on [Moreh's MoAI Platform](https://moreh.io/product).
With the MoAI platform you can scale to thousands of GPU/NPUs by automatic parallelization and optimization, without any code changes.

We currently provide five LLMs; Qwen1.5, Llama2, Mistral, GPT and Baichuan2, as well as SDXL.

    
# Getting Started
This repository contains examples of PyTorch training codes that can be executed on the MoAI Platform. Users using Pytorch on the MoAI Platform can easily train large models without extensive effort. For more information about the MoAI Platform and detailed tutorials, please visit the [Moreh Docs](https://docs.moreh.io).

## Dependency Installation

First, clone this repository and navigate to the repo directory.
```bash
git clone https://github.com/moreh-dev/quickstart
cd quickstart
```
After you are in the `quickstart` directory, install the dependency packages for the model you want to fine-tune. The requirements files for each model are located in the `requirements` directory. For example, to install the dependencies for the Llama2 model, use the following command:

```bash
pip install -r requirements/requirements_llama2.txt
```

## Model Preparation
If you want to fine-tune the Llama2, Llama3, or Mistral models, you need access to their respective Hugging Face repositories. Please ensure you have the necessary acess before starting model training.
- Llama3 : https://huggingface.co/meta-llama/Meta-Llama-3-8B or https://huggingface.co/meta-llama/Meta-Llama-3-70B
- Llama2 : https://huggingface.co/meta-llama/Llama-2-7b-hf
- Mistral : https://huggingface.co/mistralai/Mistral-7B-v0.1

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

Information about the models currently supported by this repository, along with their target tasks and training scripts, are as follows:

| **Baseline Model**    | **Task**           | **Training Script**                      | **Dataset**                                                                                                                                                |
| ------------ | ------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Llama3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)  | Text Summarization | `tutorial/train_llama3.py`     | [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)                                                                                      |
| [Llama3 70B  ](https://huggingface.co/meta-llama/Meta-Llama-3-70B) | Text Summarization | `tutorial/train_llama3_70b.py` | [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)                                                                                      |
| [Llama2 7B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)       | Text Summarization | `tutorial/train_llama2.py`     | [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)                                                                                      |
| [Qwen1.5 7B](https://huggingface.co/Qwen/Qwen1.5-7B)     | Code Generation    | `tutorial/train_qwen.py`       | [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)                               |
| [Mistral v0.1 7B ](https://huggingface.co/mistralai/Mistral-7B-v0.1)      | Code Generation    | `tutorial/train_mistral.py`    | [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)                               |
| [Cerebras GPT 13B](https://huggingface.co/cerebras/Cerebras-GPT-13B) | Code Generation    | `tutorial/train_gpt.py`        | [mlabonne/Evol-Instruct-Python-26k](https://huggingface.co/datasets/mlabonne/Evol-Instruct-Python-26k)                                                     |
| [Baichuan2 13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)    | Chatbot            | `tutorial/train_baichuan2_13b.py`   | [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) |

### Training

Run the training script to fine-tune the model. For example, to train the Llama3 8B model:

```bash 
python tutorial/train_llama3.py \
  --epochs 1  \
  --batch-size 256 \
  --block-size 1024 \
  --max_train_steps=15000 \
  --lr=0.00001 \
  --save-dir=${SAVE_DIR_PATH} \
  --ignore-index -100 \
  --log-interval 1
```


### Inference
| **Baseline Model**       | **Infernece Script**                    |
| --------------- | -------------------------------- |
| Llama3 8B & 70B | `tutorial/inference_llama3.py`   |
| Llama2 7B         | `tutorial/inference_llama2.py`   |
| Qwen1.5 7B        | `tutorial/inference_qwen.py`     |
| Mistral 7B        | `tutorial/inference_mistral.py`  |
| Cerebras GPT 13B   | `tutorial/inference_gpt.py`      |
| Baichuan2 13B       | `tutorial/inference_baichuan.py` |

Perform inference by running the inference script for each model. For example, to proceed with inference on fine-tuned Llama3 models:

```bash 
python tutorial/train_llama3.py \
  --model-name-or-path=${SAVE_DIR_PATH}
```

```bash
# output example
Llama3: [SUMMARIZE] (CNN)Arsenal kept their slim hopes of winning this season's English Premier League title alive by beating relegation threatened Burnley 1-0 at Turf Moor. A first half goal from Welsh international Aaron Ramsey was enough to separate the two sides and secure Arsenal's hold on second place. More importantly it took the north London club to within four points of first placed Chelsea, with the two clubs to play next week. 
...
```


## Stable Diffusion XL

We provide fine-tuning example code for the Stable Diffusion XL model.

### Training
| Baseline Model                                                                         | Task                     | Training Script          | Dataset                                                                                            |
| -------------------------------------------------------------------------------------- | ------------------------ | ------------------------ | -------------------------------------------------------------------------------------------------- |
| [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Test-to-Image Generation | `tutorial/train_sdxl.py` | [lambdalabs/naruto-blip-captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) |


Run the training script for Stable Diffusion XL:

```bash 
python tutorial/train_sdxl.py \
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
python tutorial/inference_sdxl.py \
  --model-name-or-path=${SAVE_DIR_PATH}
```

Adjust the prompt by editing the PROMPT variable in the inference script:
```python
...
PROMPT = "a man with grey hair and a red jacket"
...
```


The resulting image will be saved as `sdxl_result.jpg`.

