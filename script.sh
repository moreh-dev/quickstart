TRANSFORMERS_VERBOSITY=info TOKENIZERS_PARALLELISM=true accelerate launch \
     --config_file config.yaml train.py \
     --lr 0.000001 \
     --num-epochs 3 \
     --model baichuan-inc/Baichuan2-13B-Base \
     --output-dir mistral