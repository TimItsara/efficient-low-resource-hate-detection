#!/bin/sh

## activate the right virtual environment (You need to change this path to your own environment path)
# source ~/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection/hate_detection_env_new/bin/activate
source

# Set the full path for the $DATA variable (You need to change this path to your own path)
# DATA=/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection
# DATA_CN=/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection/new-data-collection/chinese/datasets/chinese_hatesent
DATA=
DATA_CN=

for dataset in chinese; do
    for modelpath in $DATA/low-resource-hate/finetuned-models/random-sample/monolingual-models/chinese-macbert-base_*/; do
        for testpath in $DATA_CN/test_*.csv; do
            python finetune_and_test.py \
                --model_name_or_path ${modelpath} \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_predict \
                --test_file ${testpath} \
                --test_results_dir $TARGET/low-resource-hate/results/${dataset}_$(basename $testpath .csv) \
                --test_results_name $(basename $modelpath).csv \
                --per_device_eval_batch_size 64 \
                --max_seq_length 128 \
                --output_dir .
        done
    done
    for modelpath in $DATA/low-resource-hate/finetuned-models/random-sample/multilingual-models/twitter-xlm-roberta-base_*${dataset}*/; do
        for testpath in $DATA_CN/test_*.csv; do
            python finetune_and_test.py \
                --model_name_or_path ${modelpath} \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_predict \
                --test_file ${testpath} \
                --test_results_dir $TARGET/low-resource-hate/results/${dataset}_$(basename $testpath .csv) \
                --test_results_name $(basename $modelpath).csv \
                --per_device_eval_batch_size 64 \
                --max_seq_length 128 \
                --output_dir .
        done
    done
done

for dataset in chinese; do
    for modelpath in $TARGET/low-resource-hate/finetuned-models/random-sample/multilingual-models/xlmt_dyn21_en_*${dataset}*/; do
        for testpath in $DATA_CN/test_*.csv; do
            python finetune_and_test.py \
                --model_name_or_path ${modelpath} \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_predict \
                --test_file ${testpath} \
                --test_results_dir $TARGET/low-resource-hate/results/${dataset}_$(basename $testpath .csv) \
                --test_results_name $(basename $modelpath).csv \
                --per_device_eval_batch_size 64 \
                --max_seq_length 128 \
                --output_dir .
        done
    done
    for modelpath in $TARGET/low-resource-hate/finetuned-models/random-sample/multilingual-models/xlmt_fou18_en_*${dataset}*/; do
        for testpath in $DATA_CN/test_*.csv; do
            python finetune_and_test.py \
                --model_name_or_path ${modelpath} \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_predict \
                --test_file ${testpath} \
                --test_results_dir $TARGET/low-resource-hate/results/${dataset}_$(basename $testpath .csv) \
                --test_results_name $(basename $modelpath).csv \
                --per_device_eval_batch_size 64 \
                --max_seq_length 128 \
                --output_dir .
        done
    done
    for modelpath in $TARGET/low-resource-hate/finetuned-models/random-sample/multilingual-models/xlmt_ken20_en_*${dataset}*/; do
        for testpath in $DATA_CN/test_*.csv; do
            python finetune_and_test.py \
                --model_name_or_path ${modelpath} \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_predict \
                --test_file ${testpath} \
                --test_results_dir $TARGET/low-resource-hate/results/${dataset}_$(basename $testpath .csv) \
                --test_results_name $(basename $modelpath).csv \
                --per_device_eval_batch_size 64 \
                --max_seq_length 128 \
                --output_dir .
        done
    done
done