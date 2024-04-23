#!/bin/bash
echo "Before you use this script, please check your code is well defined"
echo "Is Your path of running this script moreh-quckstart/"
echo "Is the train_model.py is in tutorial?"
echo "Is epoch argument name --epochs"
echo "Is batchsize argument name --batch-size"
echo "Is blocksize argument name --block-size"
echo "Is logging path argument name --log-path"
echo "Please check this points and if you are not setting like above, please fix your code and restart this script"
echo "---------------------------------------------------------------"
PYTHON_SCRIPT="yiko_train.py"

sda='4'
input_batch_size=1024

function change_sda(){
    {
        sleep 0.5
        echo "$sda"
        sleep 0.5
        echo "q"
    } | moreh-switch-model
}

function show_help() {
    echo "Usage: $0 MODEL BATCHSIZE"
    echo "Where:"
    echo "  MODEL - Name of the model to run (e.g., 'mistral', 'gpt')"
    echo "  BATCHSIZE - Max size which uses GPU memory upper 80% in the case of xlarge SDA"
    echo "Example:"
    echo "$0 gpt 64"
}

function run_python() {
    echo "batch size : $input_batch_size, log path : logs/${MODEL}_${sda}_${input_batch_size}.log"
    python $PYTHON_SCRIPT --epochs 1 --batch-size $input_batch_size --block-size 1024 > "logs/${MODEL}_${sda}_${input_batch_size}.log" 2>&1
}

if [ "$#" -ne 2 ]; then
    show_help
    exit 1
fi

MODEL=$1
BATCHSIZE=$2

BATCH_SIZES=($BATCHSIZE $((BATCHSIZE * 2)) $((BATCHSIZE * 4)))

if [ "${MODEL}" == "baichuan" ]; then
    PYTHON_SCRIPT="tutorial/train_baichuan2.py"
    echo "python_script file is $PYTHON_SCRIPT"
elif [ "${MODEL}" == "yiko" ]; then
    PYTHON_SCRIPT="yiko_train.py"
    echo "python_script file is $PYTHON_SCRIPT"
elif [ "${MODEL}" == "mistral" ]; then
    PYTHON_SCRIPT="tutorial/train_mistral.py"
    echo "python_script file is $PYTHON_SCRIPT"
elif [ "${MODEL}" == "gpt" ]; then
    PYTHON_SCRIPT="tutorial/train_gpt.py"
    echo "python_script file is $PYTHON_SCRIPT"
fi

# Proper array expansion in the loop

for sda_num in 1 2 3
do
    change_sda
    sda=$((sda + 2))
    sleep .5
    echo "sda number is $sda"
    for b in "${BATCH_SIZES[@]}"
    do
        input_batch_size=$b
        run_python
        sleep 10
        moreh-smi -r
    done
    sleep 10
done
