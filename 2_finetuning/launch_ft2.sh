#!/bin/sh

## activate the right virtual environment (You need to change this path to your own environment path)
# source ~/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection/hate_detection_env_new/bin/activate
source

# Set the full path for the $DATA variable (You need to change this path to your own path)
# DATA=/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection
# DATA_CN=/Macquarie/session_3/COMP8240/project/efficient-low-resource-hate-detection/new-data-collection/chinese/datasets/chinese_hatesent
DATA=
DATA_CN=

#### bas19_es
for dataset in chinese; do
    for split in 10 20 30 40 50 100 200 300 400 500 1000 2000; do
       for seed in rs1 rs2 rs3 rs4 rs5 rs6 rs7 rs8 rs9 rs10; do
            python finetune_and_test.py \
                --model_name_or_path hfl/chinese-macbert-base \
                --train_file $DATA_CN/train/train_${split}_${seed}.csv \
                --validation_file $DATA_CN/dev_*.csv \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_train \
                --per_device_train_batch_size 16 \
                --num_train_epochs 5 \
                --max_seq_length 128 \
                --do_eval \
                --per_device_eval_batch_size 64 \
                --eval_strategy epoch \
                --save_strategy epoch \
                --save_total_limit 1 \
                --load_best_model_at_end \
                --metric_for_best_model macro_f1 \
                --output_dir $DATA/low-resource-hate/finetuned-models/random-sample/monolingual-models/chinese-macbert-base_${split}_${seed} \
                --overwrite_output_dir

            rm -rf $DATA/low-resource-hate/finetuned-models/random-sample/monolingual-models/chinese-macbert-base_${dataset}_${split}_${seed}/check*
        done
    done
done

for dataset in chinese; do
    for split in 10 20 30 40 50 100 200 300 400 500 1000 2000; do
        for seed in rs1 rs2 rs3 rs4 rs5 rs6 rs7 rs8 rs9 rs10; do 
            python finetune_and_test.py \
                --model_name_or_path cardiffnlp/twitter-xlm-roberta-base \
                --train_file $DATA_CN/train/train_${split}_${seed}.csv \
                --validation_file $DATA_CN/dev_*.csv \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_train \
                --per_device_train_batch_size 16 \
                --num_train_epochs 5 \
                --max_seq_length 128 \
                --do_eval \
                --per_device_eval_batch_size 64 \
                --eval_strategy epoch \
                --save_strategy epoch \
                --save_total_limit 1 \
                --load_best_model_at_end \
                --metric_for_best_model macro_f1 \
                --output_dir $DATA/low-resource-hate/finetuned-models/random-sample/multilingual-models/twitter-xlm-roberta-base_chinese_${split}_${seed} \
                --overwrite_output_dir

            rm -rf $DATA/low-resource-hate/finetuned-models/random-sample/multilingual-models/twitter-xlm-roberta-base_chinese_${split}_${seed}/check*
        done
    done
done

for base_model in xlmt_dyn21_en_20000_rs1 xlmt_fou18_en_20000_rs1 xlmt_ken20_en_20000_rs1; do
    for dataset in chinese; do
        for split in 10 20 30 40 50 100 200 300 400 500 1000 2000; do
            for seed in rs1 rs2 rs3 rs4 rs5 rs6 rs7 rs8 rs9 rs10; do
                python finetune_and_test.py \
                    --model_name_or_path $DATA/low-resource-hate/english-base-models-2/${base_model}/ \
                    --train_file $DATA_CN/train/train_${split}_${seed}.csv \
                    --validation_file $DATA_CN/dev_*.csv \
                    --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                    --do_train \
                    --per_device_train_batch_size 16 \
                    --num_train_epochs 5 \
                    --max_seq_length 128 \
                    --do_eval \
                    --per_device_eval_batch_size 64 \
                    --eval_strategy epoch \
                    --save_strategy epoch \
                    --save_total_limit 1 \
                    --load_best_model_at_end \
                    --metric_for_best_model macro_f1 \
                    --output_dir $DATA/low-resource-hate/finetuned-models/random-sample/multilingual-models/${base_model}_${dataset}_${split}_${seed} \
                    --overwrite_output_dir

                rm -rf $DATA/low-resource-hate/finetuned-models/random-sample/multilingual-models/${base_model}_${dataset}_${split}_${seed}/check*
            done
        done
    done
done
