#!/bin/bash

declare -a arr=("Dos" "Fuzzy" "GEAR" "RPM")
for i in "${arr[@]}";
do
    echo "Running CAN. Anomaly Class: $i "
    python main.py --dataset CAN --img_size 48 --channels 1 --n_classes 5 --n_abnormal_classes 1 --batch_size 64 --n_epochs 600 --abnormal_class $i --log_path "./log/can_strategy" --pkl_path "./pkl/can_strategy" --is_train_mode true
done
exit 0
