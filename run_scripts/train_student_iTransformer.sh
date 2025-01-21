#!/bin/bash

declare -A datasets

datasets['ECL']="96 192 336 720"
datasets['ETTm1']="96 192 336 720"
datasets['ETTm2']="96 192 336 720"
datasets['ETTh1']="96 192 336 720"
datasets['ETTh2']="96 192 336 720"
datasets['Solar']="96 192 336 720"
datasets['Traffic']="96 192 336 720"
datasets['Weather']="96 192 336 720"

declare -A alpha_dist=(
    ["ECL"]=0.1 ["Traffic"]=0.5 ["Solar"]=0.1 
    ["ETTh1"]=2 ["ETTh2"]=1 ["ETTm1"]=1 
    ["ETTm2"]=1 ["Weather"]=0.1
)

declare -A beta_dist=(
    ["ECL"]=0.5 ["Traffic"]=0.5 ["Solar"]=1
    ["ETTh1"]=2 ["ETTh2"]=0.5 ["ETTm1"]=1 
    ["ETTm2"]=0.1 ["Weather"]=0.1
)

declare -A norm_dist=(
    ["ECL"]="non-stationary" ["Traffic"]="revin" ["Weather"]="non-stationary" ["ETTm2"]="non-stationary"
    ["ETTh1"]="non-stationary" ["ETTh2"]="non-stationary" ["ETTm1"]="non-stationary" ["Solar"]="non-stationary"
)

declare -A d_model_dist=(
    ["ECL"]="512" ["Traffic"]="1024" ["Weather"]="512" ["ETTm2"]="512"
    ["ETTh1"]="512" ["ETTh2"]="512" ["ETTm1"]="512" ["Solar"]="512"
)

declare -A train_epochs_dist=(
    ["ECL"]="30" ["Traffic"]="10" ["Weather"]="100" ["ETTm2"]="100"
    ["ETTh1"]="100" ["ETTh2"]="100" ["ETTm1"]="100" ["Solar"]="30"
)

method=DMLP
seq_len=720
learning_rate=0.01
lradj=type1
model_t=iTransformer 
gpu=0

folder=log_student_results
mkdir -p ${folder}

for dataset in "${!datasets[@]}"; do
    pred_lens=${datasets[$dataset]}
    alpha=${alpha_dist[$dataset]}
    beta=${beta_dist[$dataset]}
    norm=${norm_dist[$dataset]}
    d_model=${d_model_dist[$dataset]}
    train_epochs=${train_epochs_dist[$dataset]}
    for pred_len in $pred_lens; do
        echo "Run method:$method, dataset: $dataset with seq_len: $seq_len, pred_len: $pred_len, alpha: $alpha, beta: $beta, learning_rate: $learning_rate, lradj: $lradj, d_model: $d_model, model_t: $model_t."
        echo "Output in ${folder}/${dataset}_${pred_len}_${learning_rate}_${lradj}_${d_model}_${train_epochs}_${norm}_${alpha}_${beta}_${model_t}_${method}.out"
        bash ./scripts/distillation/${dataset}/${method}.sh ${gpu} ${pred_len} ${learning_rate} ${lradj} ${d_model} ${train_epochs} ${norm} ${alpha} ${beta} ${model_t} ${seq_len} >> ${folder}/${dataset}_${pred_len}_${learning_rate}_${lradj}_${d_model}_${train_epochs}_${norm}_${alpha}_${beta}_${model_t}_${method}.out
    done
done
