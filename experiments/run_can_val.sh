#!/bin/bash

declare -a arr=("Dos" "Fuzzy" "GEAR" "RPM")
declare -a pkg_arr=("" "" "" "") # need to be modified to your storage model parameter location
for i in $(seq 0 `expr ${#arr[@]} - 1`);
do
    abnormal_class=${arr[i]}
    pkg_path=${pkg_arr[i]}
    echo "Running CAN. Anomaly Class: $abnormal_class "
    python main.py --dataset CAN --img_size 48 --channels 1 --n_classes 5 --n_abnormal_classes 1 --batch_size 64 --n_epochs 600 --abnormal_class $abnormal_class --log_path "./log/can_strategy" --pkl_path $pkg_path
done
exit 0
