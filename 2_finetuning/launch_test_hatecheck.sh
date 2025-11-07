#!/bin/sh

## activate the right virtual environment (You need to change this path to your own environment path)
# source ~/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection/hate_detection_env_new/bin/activate
source

# Set the full path for the $DATA variable (You need to change this path to your own path)
# DATA=/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection
# DATA_CN=/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection/new-data-collection/chinese/datasets/chinese_hatesent
DATA=
DATA_CN=

CHINESE_PREFIX=cn

for base_model in fou18_en dyn21_en ken20_en; do
    for dataset in chinese; do
        for modelpath in $DATA/low-resource-hate/finetuned-models/random-sample/multilingual-models/xlmt_${base_model}_*${dataset}*/; do
            # Validate modelpath
            if [ ! -d "$modelpath" ] || [ ! -f "$modelpath/config.json" ]; then
                echo "Skipping invalid modelpath: $modelpath"
                continue
            fi

            for testpath in $DATA/0_data/hatecheck/*_$CHINESE_PREFIX.csv; do
                # Validate testpath
                if [ ! -f "$testpath" ]; then
                    echo "Skipping invalid testpath: $testpath"
                    echo "$DATA/0_data/hatecheck/*_$CHINESE_PREFIX.csv"
                    continue
                fi

                echo "Running test with model: $modelpath and test file: $testpath"

                python finetune_and_test.py \
                    --model_name_or_path ${modelpath} \
                    --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                    --do_predict \
                    --test_file ${testpath} \
                    --test_results_dir $DATA/low-resource-hate/results/hatecheck_$dataset \
                    --test_results_name $(basename $modelpath).csv \
                    --per_device_eval_batch_size 64 \
                    --max_seq_length 128 \
                    --output_dir .
            done
        done
    done
done


for dataset in chinese; do
    for modelpath in $DATA/low-resource-hate/finetuned-models/random-sample/multilingual-models/twitter-xlm-roberta-base_*${dataset}*/; do
        # Validate modelpath
        if [ ! -d "$modelpath" ] || [ ! -f "$modelpath/config.json" ]; then
            echo "Skipping invalid modelpath: $modelpath"
            continue
        fi

        for testpath in $DATA/0_data/hatecheck/*_$CHINESE_PREFIX.csv; do
            # Validate testpath
            if [ ! -f "$testpath" ]; then
                echo "Skipping invalid testpath: $testpath"
                continue
            fi

            echo "Running test with model: $modelpath and test file: $testpath"

            python finetune_and_test.py \
                --model_name_or_path ${modelpath} \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_predict \
                --test_file ${testpath} \
                --test_results_dir $DATA/low-resource-hate/results/hatecheck_$dataset \
                --test_results_name $(basename $modelpath).csv \
                --per_device_eval_batch_size 64 \
                --max_seq_length 128 \
                --output_dir .
        done
    done
done

for base_model in chinese-macbert-base; do
    for dataset in chinese; do
        for modelpath in $DATA/low-resource-hate/finetuned-models/random-sample/monolingual-models/${base_model}_*/; do
            # Validate modelpath
            if [ ! -d "$modelpath" ] || [ ! -f "$modelpath/config.json" ]; then
                echo "Skipping invalid modelpath: $modelpath"
                continue
            fi

            for testpath in $DATA/0_data/hatecheck/*_$CHINESE_PREFIX.csv; do
                # Validate testpath
                if [ ! -f "$testpath" ]; then
                    echo "Skipping invalid testpath: $testpath"
                    continue
                fi

                echo "Running test with model: $modelpath and test file: $testpath"

                python finetune_and_test.py \
                    --model_name_or_path ${modelpath} \
                    --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                    --do_predict \
                    --test_file ${testpath} \
                    --test_results_dir $DATA/low-resource-hate/results/hatecheck_$dataset \
                    --test_results_name $(basename $modelpath).csv \
                    --per_device_eval_batch_size 64 \
                    --max_seq_length 128 \
                    --output_dir .
            done
        done
    done
done
