#!/bin/sh

## activate the right virtual environment (You need to change this path to your own environment path)
# source ~/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection/hate_detection_env_new/bin/activate
source

# Set the full path for the $DATA variable (You need to change this path to your own path)
# DATA=/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection
DATA=

for basemodel in xlmt_dyn21_en_20000_rs1 xlmt_fou18_en_20000_rs1 xlmt_ken20_en_20000_rs1; do
    for dataset in bas19_es for19_pt; do
        for testpath in $DATA/0_data/main/1_clean/${dataset}/test_*.csv; do
            python finetune_and_test.py \
                --model_name_or_path $DATA/low-resource-hate/english-base-models/${basemodel}/ \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_predict \
                --test_file ${testpath} \
                --test_results_dir $DATA/low-resource-hate/results/${dataset}_$(basename $testpath .csv) \
                --test_results_name ${basemodel}_${dataset}_0_rs1.csv \
                --per_device_eval_batch_size 64 \
                --max_seq_length 128 \
                --output_dir .
        done
    done
done