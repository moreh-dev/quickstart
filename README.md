# Moreh Quickstart
Quickstart code for moreh docs

-----
## Prerequisites

This repository contains simple example with smaller model sizes.
To train large scale models, you need to use MoAI platform of **[Moreh](https://moreh.io/)**, 
and you don't need to modify any model code or train script in this repository. 


## Installation

### 1. (Optional) create conda environment
```bash
conda create -n {your-env-name} python=3.8 -y
```

### 2. (Optional) Install the latest version of MoAI Platform
```bash
update-moreh
```

### 3. Clone moreh hub models repository
``` bash
git clone https://github.com/moreh-dev/moreh-hub-models.git
```

### 4. install dependencies
```bash
cd moreh-quickstart
pip install -e .
```


## Quick tour

To immediately train the model, we provide example scripts. 
Here is how to quickly train a gpt model with an example script.

```bash
bash train_gpt_small.sh
```

