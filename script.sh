TRANSFORMERS_VERBOSITY=info TOKENIZERS_PARALLELISM=true accelerate launch \
     --config_file config.yaml train.py \
     --lr 0.000001 \
     --model meta-llama/Meta-Llama-3-8B \
     --train-batch-size 64 \
     --eval-batch-size 64 \
     --sequence-length 1024 \
     --log-interval 10 \
     --num-epochs 5 \
     --output-dir llama3-finetuned