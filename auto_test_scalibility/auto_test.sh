#!/bin/bash
echo "Before you use this script, please check your code is well defined"
echo "Is Your path of running this script moreh-quckstart/"
echo "Is epoch argument name --epochs"
echo "Is batchsize argument name --batch-size"
echo "Is blocksize argument name --block-size"
echo "Please check this points and if you are not setting like above, please fix your code and restart this script"
echo "---------------------------------------------------------------"


sda="4"
input_batch_size=0
input_block_size=1024
log_path="logs"
end_time=60

function change_sda() {
    {
        sleep 0.5
        echo "$sda"
        sleep 0.5
        echo "q"
    } | moreh-switch-model
}


function run_python() {
    echo "batch size : $input_batch_size, log path : logs/${model_name}_${sda}_batch${input_batch_size}_block${input_block_size}.log"
    if [ "default" == "${batch_size}" ] | [ "default" == "${block_size}" ]; then {
            python -u $script_path --model-name-or-path ${model_path} --epochs 1
        } \
            > "${log_path}/${model_name}_sda_${sda}_batch${input_batch_size}_block${input_block_size}.log" 2>&1 &
    else
        nohup python $script_path --model-name-or-path ${model_path} --epochs 1 --batch-size $input_batch_size --block-size $input_block_size > "${log_path}/${model_name}_sda_${sda}_batch${input_batch_size}_block${input_block_size}.log" 2>&1 &
    fi

    PID=$!
    rm -rf "${log_path}/${model_name}_sda_${sda}_batch${input_batch_size}_block${input_block_size}_moreh_smi.log"
    if [[ "false" == "$end_time" ]]; then
        while true; do
            if ps -p $PID > /dev/null; then
                sleep 60
                echo "moreh-smi check ${i} min"
                moreh-smi >> "${log_path}/${model_name}_sda_${sda}_batch${input_batch_size}_block${input_block_size}_moreh_smi.log" &
            else
                break
            fi
        done
    else
        for i in {1..${end_time}}
        do
              sleep 60
            echo "moreh-smi check ${i} min"
            moreh-smi >> "${log_path}/${model_name}_sda_${sda}_batch${input_batch_size}_block${input_block_size}_moreh_smi.log" &
        done

    fi

    moreh-smi -r
    sleep 10
    kill ${PID}
    sleep 10
}


function show_help() {
    echo "Usage: $0 FILEPATH"
    echo "Where:"
    echo " FILEPATH - Path of configuration file"
}


function prepare_dataset() {
    dataset_path="dataset/prepare_${model_name}_dataset.py"
    if [ ! -e $dataset_path ]; then
        echo "Dataset path ${dataset_path} is invalid, Please check your code."
        exit 1
    else
        python ${dataset_path} --model-name-or-path ${model_path}
    fi

    if [ $? -eq 0 ]; then
        echo "Dataset is downloaded"
    else
        echo "Dataset is not download, something error"
        exit 1
    fi
}

function install_packages() {
    requirements_path="requirements/requirements_${model_name}.txt"
    pip install -r ${requirements_path}
}



# Create a temporary config file for testing
config_file=$1

# Initialize variables
model_name=""
script_path=""
model_path=""
log_path=""
in_arguments_block=0
model_arguments=()

# Read the config file and parse each line
while IFS= read -r line || [ -n "$line" ]; do
    # Trim leading and trailing whitespace
    line=$(echo "$line" | sed 's/^[ \t]*//;s/[ \t]*$//')

    # Skip empty lines
    if [[ -z "$line" ]]; then
        continue
    fi

    # Check for model_name
    if [[ $line =~ model_name ]]; then
        model_name=$(echo "$line" | awk -F'=' '{print $2}' | sed 's/[", ]//g')
        echo "Model Name: $model_name"
    fi

    # Check for model_path
    if [[ $line =~ model_path ]]; then
        model_path=$(echo "$line" | awk -F'=' '{print $2}' | sed 's/[", ]//g')
        if [[ ! -e "$model_path" ]]; then
            echo "The model_path '$model_path' is invalid"
            exit 1
        fi
        echo "Model Path: $model_path"
    fi

    # Check for script_path
    if [[ $line =~ script_path ]]; then
        script_path=$(echo "$line" | awk -F'=' '{print $2}' | sed 's/[", ]//g')
        if [[ ! -e "$script_path" ]]; then
            echo "The script_path '$script_path' is invalid"
            exit 1
        fi
        echo "Model Path: $script_path"
    fi

    # Check for end time
    if [[ $line =~ end_time ]]; then
        user_end_time=$(echo "$line" | awk -F'=' '{print $2}' | sed 's/[", ]//g')
        if [[ $script_path =~ ^[0-9]+$ ]]; then
            end_time=user_end_time
        elif [[ "false" == "$user_end_time" ]]; then
            end_time=user_end_time
        else
            echo "The end_time '$user_end_timed' is invalid"
            exit 1
        fi
    fi

    # Check for log_path
    if [[ $line =~ log_path ]]; then
        log_path=$(echo "$line" | awk -F'=' '{print $2}' | sed 's/[", ]//g')
        if [[ ! -d "$log_path" ]]; then
            echo "The log_path '$log_path' is invalid"
            exit 1
        fi
        echo "Log Path: $log_path"
        echo "Model name: ${model_name}, Script path: ${script_path} Model path: ${model_path}, Log path: ${log_path}"
        #Below two lines are dependence on quickstart (morehdocs)
        install_packages
        prepare_dataset
    fi


    # Check for model_arguments
    if [[ $line =~ model_arguments ]]; then
        in_arguments_block=1
        continue
    fi
    # Parse model_arguments
    if [[ $in_arguments_block -eq 1 ]]; then
        # Break when the end of the arguments list is reached
        if [[ $line =~ \] ]]; then
            in_arguments_block=0
            for arg in "${model_arguments[@]}"; do
                IFS=',' read -ra params <<< "$arg"
                for param in "${params[@]}"; do
                    key=$(echo $param | awk -F'=' '{print $1}' | sed 's/[ ]//g')
                    value=$(echo $param | awk -F'=' '{print $2}' | sed 's/[ ]//g')
                    case $key in
                        batch_size) batch_size=$value ;;
                        block_size) block_size=$value ;;
                        sda) sda=$value ;;
                    esac
                done
                change_sda
                sleep 1
                echo "SDA flavor set ${sda}"
                input_batch_size=${batch_size}
                input_block_size=${block_size}
                run_python
            done
            model_arguments=()
        else
            model_arguments+=("$line")
        fi
    fi
done < "$config_file"

# Clean up the temporary config file
rm $config_file


