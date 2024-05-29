import re
import os, sys
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Log Parser")
    parser.add_argument(
        "--log-path",
        type=str,
        default="./logs",
        help="log directory path"
    )
    parser.add_argument(
        "--validation", "-v",
        action='store_true',
        help='Compare throughput and memory_usage with morehdocs\''
    )
    args = parser.parse_args()

    return args


# moreh-smi parse functions
def process_lines_containing_python(dir_path):
    datas = {}
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if 'moreh_smi' in file_path:
            with open(file_path, 'r') as file:
                script_lines = []
                for line in file:
                    # Check if 'python' is in the line
                    if 'python' in line.lower():  # This makes the search case-insensitive
                        script_lines.append(line.strip())
                total_mem_usage = 0
                total_cnt = 0
                for sl in script_lines:
                    tmp = sl.split('|')
                    mem_usage = int(tmp[5].replace('MiB', ''))
                    if mem_usage != 0:
                        total_mem_usage += mem_usage
                        total_cnt += 1

                # calcuate average
                if total_cnt != 0:
                    avg_mem_usage = total_mem_usage/total_cnt
                else:
                    avg_mem_usage = 0

                datas[file_path.replace('./logs/', '')] = avg_mem_usage


    return datas


def print_moreh_smi(datas):
    for k in datas:
        print(f"{k}'s max mem usage is {datas[k]:.2f} MiB({(datas[k]/1024):.2f} GiB)")



# throughput parser functions
def parse_throughput(lines):
    tps = 0
    tps_list = []
    for l in lines:
        match = re.search(r'Throughput\s*:\s*(\d+)', l)
        if match:
            tps_list.append(int(match.group(1)))
    if len(tps_list) == 0 : return 0
    tps_list.remove(max(tps_list))
    tps_list.remove(min(tps_list))
    return round(sum(tps_list)/len(tps_list), 2)


def process_file(file_path):
    with open(file_path, 'r') as file:
        # Process each line in the file
        lines = []
        for line in file:
            if 'throughput' in line.lower():
                lines.append(line.strip())
    return lines

def process_all_files_in_folder(directory):
    datas = {}
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        if 'moreh_smi' in file_path: continue

        if os.path.isfile(file_path):
            lines = process_file(file_path)
            avg_throughput = parse_throughput(lines)
            datas[file_path.replace('./logs/', '')] = f"{avg_throughput} until {len(lines)} steps"
    return datas

def print_data(gpu_usage_datas, throughput_datas, args):
    if args.validation:
        original_datas = {}
        with open('./auto_test_scalibility/moreh_docs_results.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                model_name = line[0]
                tps = line[1]
                gpu_usage = line[3]
                tmp_dict = {'tps':tps, 'gpu_usage':gpu_usage}
                original_datas[model_name] = tmp_dict
        print(original_datas) 
        print("GPU Memory Usage")
        print("-"*20)
        for k in gpu_usage_datas:
            model_name = k.split('_')[0]
            gpu_usage = float(gpu_usage_datas[k])
            base_gpu_usage = float(original_datas[model_name]['gpu_usage'])
            if (base_gpu_usage < 0.9 * gpu_usage) or (base_gpu_usage > 1.1 * gpu_usage):
                print(f"{model_name}'s original gpu_usage is outdated.(under 0.9 * original or over 1.1 * original) original : {base_gpu_usage} MiB test : {gpu_usage_datas[k]:.2f} MiB")
            else:
                print(f"{model_name}'s original gpu_usage is valid. original : {base_gpu_usage} Mib test : {gpu_usage_datas[k]:.2f} MiB")
        print("-"*20)
        print("Throughputs")
        print("-"*20)
        for k in throughput_datas:
            print(f"{k}'s avg throughput is {throughput_datas[k]}")
        print("-"*20)
    else:
        print("GPU Memory Usage")
        print("-"*20)
        for k in gpu_usage_datas:
            print(f"{k}'s avg mem usage is {gpu_usage_datas[k]:.2f} MiB({(gpu_usage_datas[k]/1024):.2f} GiB)")
        print("-"*20)
        print("Throughputs")
        print("-"*20)
        for k in throughput_datas:
            print(f"{k}'s avg throughput is {throughput_datas[k]}")
        print("-"*20)


def main(args):
    gpu_usage_datas = process_lines_containing_python(args.log_path)
    throughput_datas = process_all_files_in_folder(args.log_path)
    print_data(gpu_usage_datas, throughput_datas, args)

if __name__=='__main__':
    args = parse_args()
    main(args)

