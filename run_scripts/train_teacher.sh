#!/bin/bash

declare -A datasets

datasets['ECL']="96 192 336 720"
datasets['Solar']="96 192 336 720"
datasets['Traffic']="96 192 336 720"
datasets['ETTm1']="96 192 336 720"
datasets['ETTm2']="96 192 336 720"
datasets['ETTh1']="96 192 336 720"
datasets['ETTh2']="96 192 336 720"
datasets['Weather']="96 192 336 720"

folder=log_teacher_results
mkdir -p ${folder}

method=iTransformer # Select a teacher: [iTransformer ModernTCN TimeMixer PatchTST MICN Fedformer TimesNet Autoformer]
seq_len=720
gpu=0
for dataset in "${!datasets[@]}"; do
    pred_lens="${datasets[$dataset]}"
    for pred_len in $pred_lens; do
        echo "Run method:$method, dataset: $dataset with pred_len: $pred_len and seq_len: $seq_len. Output in $folder/${dataset}_${seq_len}_${pred_len}_${method}.out"
        bash ./scripts/long_term_forecast/${dataset}/${method}.sh ${gpu} ${pred_len} ${seq_len} >> $folder/${dataset}_${seq_len}_${pred_len}_${method}.out
    done
done