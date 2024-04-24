#! /bin/bash

echo "Parsing Start"
echo "-----------------------------------------------------------"
echo "GPU memory usages"
python moreh_smi_parser.py
echo "-----------------------------------------------------------"
echo "Max Throughputs"
python throughput_parser.py logs
echo "-----------------------------------------------------------"
echo "Parsing Done"
