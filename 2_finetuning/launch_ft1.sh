#!/bin/sh

## activate the right virtual environment (You need to change this path to your own environment path)
# source ~/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection/hate_detection_env_new/bin/activate
source

# Set the full path for the $DATA variable (You need to change this path to your own path)
# DATA=/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection
DATA=

for model in dyn21_en fou18_en ken20_en; do
    for split in 10 20 30 40 50 100 200 300 400 500 1000 2000 3000 4000 5000 10000 20000; do
        for seed in 1 2 3 4 5 6 7 8 9 10; do
            python finetune_and_test.py \
                --model_name_or_path cardiffnlp/twitter-xlm-roberta-base \
                --train_file $DATA/0_data/main/1_clean/${model}/train/train_${split}_rs${seed}.csv \
                --validation_file $DATA/0_data/main/1_clean/${model}/dev_500.csv \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_train \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 16 \
                --per_device_eval_batch_size 1 \
                --eval_accumulation_steps 1 \
                --num_train_epochs 3 \
                --max_seq_length 128 \
                --group_by_length True \
                --gradient_checkpointing True \
                --torch_empty_cache_steps 50 \
                --save_strategy "no" \
                --do_eval \
                --output_dir $DATA/low-resource-hate/english-base-models/xlmt_${model}_${split}_rs${seed} \
                --overwrite_output_dir
        done
    done
done
