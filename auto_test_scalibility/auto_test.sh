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

sda="4"
input_batch_size=1024
input_block_size=1024
input_log_path="logs"
function change_sda() {
    {
        sleep 0.5
        echo "$sda"
        sleep 0.5
        echo "q"
    } | moreh-switch-model
}

function show_help() {
    echo "Usage: $0 FILEPATH"
    echo "Where:"
    echo " FILEPATH - Path of configuration file"
}

function run_python() {
    echo "batch size : $input_batch_size, log path : logs/${model_name}_${sda}_batch${input_batch_size}_block${input_block_size}.log"
    python $PYTHON_SCRIPT --epochs 1 --batch-size $input_batch_size --block-size $input_block_size > "${input_log_path}/${model_name}_${sda}_batch${input_batch_size}_block${input_block_size}.log" 2>&1
}


if [ "$#" -ne 1 ]; then
    show_help
    exit 1
fi

CONFIG_FILE=$1
model_name=$(grep 'model_name' $CONFIG_FILE | awk -F'=' '{print $2}' | tr -d ' "')
model_path=$(grep 'model_path' $CONFIG_FILE | awk -F'=' '{print $2}' | tr -d ' "')
log_path=$(grep 'log_path' $CONFIG_FILE | awk -F'=' '{print $2}' | tr -d ' "')
# Print the model name
echo "Model Name: $model_name"

# Extract the arguments block
model_arguments=$(awk '/model_arguments = \[/{flag=1; next} /\]/{flag=0} flag' $CONFIG_FILE)

echo "Model Arguments:"
echo "$model_arguments"


if [[ -n "$log_path" && -e "$log_path" ]]; then
    input_log_path=$log_path
elif [[ -n "$log_path" ]]; then
    echo "log_path is exist, but not invalid"
    exit 1
fi



if [[ -n "$model_path" && -e "$model_path" ]]; then
    PYTHON_SCRIPT=$model_path
elif [ "${model_name}" == "baichuan" ]; then
    PYTHON_SCRIPT="tutorial/train_baichuan2_13b.py"
    echo "python_script file is $PYTHON_SCRIPT"
elif [ "${model_name}" == "yiko" ]; then
    PYTHON_SCRIPT="yiko_train.py"
    echo "python_script file is $PYTHON_SCRIPT"
elif [ "${model_name}" == "mistral" ]; then
    PYTHON_SCRIPT="tutorial/train_mistral.py"
    echo "python_script file is $PYTHON_SCRIPT"
elif [ "${model_name}" == "gpt" ]; then
    PYTHON_SCRIPT="tutorial/train_gpt.py"
    echo "python_script file is $PYTHON_SCRIPT"
elif [ "${model_name}" == "llama" ]; then
    PYTHON_SCRIPT="tutorial/train_llama2.py"
    echo "python_script file is $PYTHON_SCRIPT"
elif [ "${model_name}" == "qwen" ]; then
    PYTHON_SCRIPT="tutorial/train_qwen.py"
    echo "python_script file is $PYTHON_SCRIPT"
else
    echo "There's no python script file ${model_name}"
    exit 1
fi




# Do while, run python
echo "$model_arguments" | while IFS= read -r line; do
    batch_size=$(echo $line | awk -F'batch_size =' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    block_size=$(echo $line | awk -F'block_size =' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    sda_num=$(echo $line | awk -F'sda_num =' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    echo "$batch_size $block_size $sda_num"
    sda=${sda_num}
    change_sda
    sleep .5
    echo "SDA flavor set ${sda}"
    input_batch_size=${batch_size}
    input_block_size=${block_size}
    run_python
    sleep 10
    moreh-smi -r
    sleep 10
done


