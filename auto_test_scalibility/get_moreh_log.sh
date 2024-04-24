#! /bin/bash

while true; do
    moreh-smi >> logs/moreh-smi-data.logs
    sleep 300
done
