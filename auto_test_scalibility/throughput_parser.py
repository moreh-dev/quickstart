import os, sys
import re

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

        if file_path == 'logs/moreh-smi-data.logs': continue
        
        if os.path.isfile(file_path):
            lines = process_file(file_path)
            max_throughput = parse_throughput(lines)
            datas[file_path.replace('logs/', '')] = max_throughput
    return datas

def print_data(datas):
    for k in datas:
        print(f"{k}'s max throughput is {datas[k]}")


# Example usage
if __name__ == "__main__":
    # Path to the directory containing files
    if len(sys.argv) != 2:
        print("argument missing")
        print("python throughput_parser.py directory_path")
        sys.exit()

    directory_path = sys.argv[1]

    # Process all files in the directory
    datas = process_all_files_in_folder(directory_path)
    print_data(datas)
