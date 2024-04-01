#!/bin/bash

python train.py \
 --model-type gpt \
 --model-config-path ./model/configs/gpt-2.3b.json \
 --tokenizer-name gpt2 \
 --save-model ./checkpoint/gpt \
 --checkpoint-path ./checkpoint/gpt \
 --checkpoint-step-interval 10000 \
 --update-tokenizer \
 --warmup 2000 \
 --train-batch-size 4 \
 --val-batch-size 4 \
 --grad-accumulation-steps 1 \
 --lr 0.0025 \
 --lr-scheduler LinearWarmup \
 --optimizer AdamW \
 --epochs 1 \
 --weight-decay 0.01 \
 --train-log-interval 10 \
 --optimizer-epsilon 1e-06 \
 --bfloat16 \
 --fixed-vocab \
 --train-dataset-type text \
 --train-dataset sample_news_data.txt \